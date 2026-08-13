"""NAF on Hopper-v5 (Gymnasium/MuJoCo) using jaxtor, rlax, optax, equinox.

Normalized Advantage Functions from Gu et al. (2016) "Continuous Deep
Q-Learning with Model-based Acceleration", arXiv:1603.00748.

Q(s,a) = V_θ(s) - ½ Σᵢ pᵢ(s) (aᵢ - μᵢ(s))²

where p(s) = softplus(net(s)) + ε is a diagonal precision vector.
Exploration: aᵢ = μᵢ(s) + σ / √pᵢ(s) · εᵢ,  ε ~ N(0, I).

Training:
1. Collect large rollout with scaled P⁻¹-shaped exploration noise
2. Compute off-policy Q(λ) returns using target network
3. Train Q via MSE over multiple epochs
4. Polyak-update target network

- jaxtor:
  - rollout sampler: GymEnv -> Mc -> VecMc -> Imc -> Roll
  - evaluation: GymEnv -> Mc -> VecMc -> Imc -> Eval
- equinox: V, μ, p networks (separate MLPs).
- optax: Adam optimizer with gradient clipping and LR annealing.
- rlax: general_off_policy_returns_from_q_and_v.

"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import rlax
import tyro
from chex import dataclass
from rich import print
from rich.progress import track

from jaxtor.env import gymnasium
from jaxtor.eval.mc import Eval as Evaluator
from jaxtor.sampler import EpisodeStats, Imc, Mc, Roll, VecMc
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI.

    Tuned hyperparameters per environment:
        Hopper-v5:      --noise-scale 0.1 --noise-eps 1.0  (3085 @ 100 iters)
        Walker2d-v5:    --noise-scale 0.1 --noise-eps 1.0  (2743 @ 100 iters)
        HalfCheetah-v5: --noise-scale 0.1 --noise-eps 1.0  (4470 @ 100 iters)
        Ant-v5:         --noise-scale 0.1 --noise-eps 1.0  (4194 @ 500 iters)
    """

    env_id: str = "Hopper-v5"
    n_iters: int = 500
    n_envs: int = 16
    seq_len: int = 2048
    max_eps_len: int = 1000
    seed: int = 0

    v_hiddens: tuple[int, ...] = (128, 128)
    mu_hiddens: tuple[int, ...] = (64, 64)
    p_hiddens: tuple[int, ...] = (128, 128)

    lr: float = 3e-4
    lr_schedule: Literal["constant", "linear"] = "linear"
    max_grad_norm: float = 0.5

    gamma: float = 0.99
    target_update_freq: int = 10
    trace_lambda: float = 0.9
    noise_eps: float = 1.0
    noise_scale: float = 0.1

    n_epochs: int = 10
    n_batches: int = 32

    normalize_obs: bool = True
    normalize_reward: bool = True
    layer_norm: bool = True
    eval_freq: int = 5
    eval_envs: int = 10
    async_envs: bool = True
    env_backend: str = (
        "gymnasium"  # "gymnasium" (any env) | "envpool" (fast CPU MuJoCo)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Networks (orthogonal init per Huang et al. 2022)
# ──────────────────────────────────────────────────────────────────────────────


def _ortho_init(key: jax.Array, shape: tuple, gain: float = 1.0) -> jax.Array:
    """Orthogonal weight initialization."""
    return jax.nn.initializers.orthogonal(scale=gain)(key, shape, jnp.float32)


class MLP(eqx.Module):
    """Multi-layer perceptron with tanh activations and orthogonal init."""

    layers: list
    layer_norms: list | None

    def __init__(
        self,
        in_dim: int,
        hiddens: tuple[int, ...],
        out_dim: int,
        *,
        output_gain: float = math.sqrt(2),
        layer_norm: bool = False,
        key: jax.Array,
    ):
        keys = jrd.split(key, len(hiddens) + 1)
        dims = [in_dim, *hiddens, out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            gain = output_gain if i == len(dims) - 2 else math.sqrt(2)
            w = _ortho_init(keys[i], (dims[i], dims[i + 1]), gain)
            b = jnp.zeros(dims[i + 1])
            self.layers.append((w, b))
        if layer_norm:
            self.layer_norms = [(jnp.ones(d), jnp.zeros(d)) for d in hiddens]
        else:
            self.layer_norms = None

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, (w, b) in enumerate(self.layers[:-1]):
            x = x @ w + b
            if self.layer_norms is not None:
                scale, bias = self.layer_norms[i]
                x = scale * jax.nn.standardize(x, axis=-1) + bias
            x = jnp.tanh(x)
        w, b = self.layers[-1]
        return x @ w + b


# ──────────────────────────────────────────────────────────────────────────────
# Agent (deterministic policy with exploration noise)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """Deterministic policy agent for NAF.

    Q(s,a) = V(s) - ½ Σᵢ pᵢ(s) (aᵢ - μᵢ(s))²

    where p(s) = softplus(net(s)) + ε, a diagonal precision vector.

    Action sampling:
        deterministic=False: a = μ(s) + σ / √p(s) · ε,  ε ~ N(0, I).
        deterministic=True:  a = μ(s).
    """

    deterministic: bool = False
    noise_eps: float = 0.1
    noise_scale: float = 0.5
    obs_norm: RunningStats | None = None

    @dataclass
    class State:
        key: jax.Array
        v_net: MLP
        mu_net: MLP
        p_net: MLP
        obs_stats: RunningStats.State

    def _precision(self, obs: jax.Array, state: Agent.State) -> jax.Array:
        """Diagonal precision p = softplus(net(s)) + ε."""
        return jax.nn.softplus(state.p_net(obs)) + self.noise_eps

    def act(
        self,
        obs: jax.Array,
        state: Agent.State,
    ) -> tuple[jax.Array, Agent.State]:
        """Select a deterministic or exploratory NAF action."""
        policy_obs = (
            self.obs_norm.normalize(obs, state.obs_stats)
            if self.obs_norm is not None
            else obs
        )
        mu_raw = state.mu_net(policy_obs)
        if self.deterministic:
            action = jnp.tanh(mu_raw)
        else:
            key, k = jrd.split(state.key)
            p = self._precision(policy_obs, state)
            eps = jrd.normal(k, mu_raw.shape)
            action = jnp.tanh(mu_raw + self.noise_scale * eps / jnp.sqrt(p))
            state = replace(state, key=key)
        return action, state

    def q_val(self, obs: jax.Array, act: jax.Array, state: Agent.State) -> jax.Array:
        """Q(s,a) = V(s) - ½ Σᵢ pᵢ (aᵢ - μᵢ)²."""
        v = state.v_net(obs).squeeze(-1)
        mu = jnp.tanh(state.mu_net(obs))
        p = self._precision(obs, state)
        diff = act - mu
        return v - 0.5 * jnp.sum(p * diff**2, axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training state and step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Batch:
    """Flattened minibatch for NAF updates."""

    obs: jax.Array
    act: jax.Array
    ret: chex.Array


@dataclass
class State:
    """Flat training state threaded immutably through NAF updates."""

    mc: Mc.State
    agent: Agent.State
    stats: EpisodeStats.State
    rew_norm: RewardNorm.State
    opt: optax.OptState


def train_step(
    state: State,
    *,
    agent: Agent,
    roll: Roll,
    stats: EpisodeStats,
    rew_norm: RewardNorm,
    tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[State, dict]:
    """One NAF iteration: collect rollout, compute Q(λ) targets, train Q."""
    # 1. Collect rollout
    imc_state = roll.imc.init(state.mc, state.agent)
    seq, imc_state = roll.sample(imc_state)
    stats_state = stats.update(seq, state.stats)
    sam_metrics, stats_state = stats.drain(stats_state)
    state = replace(
        state,
        mc=imc_state.mc,
        agent=imc_state.agent,
        stats=stats_state,
    )
    dones = jnp.logical_or(seq.term, seq.trun)

    # 2. Obs normalization
    if agent.obs_norm is not None:
        obs_stats = agent.obs_norm.update(
            seq.obs.reshape(-1, seq.obs.shape[-1]),
            state.agent.obs_stats,
        )
        state = replace(state, agent=replace(state.agent, obs_stats=obs_stats))
        obs_n = agent.obs_norm.normalize(seq.obs, state.agent.obs_stats)
        nobs_n = agent.obs_norm.normalize(
            seq.nobs,
            state.agent.obs_stats,
        )
    else:
        obs_n, nobs_n = seq.obs, seq.nobs

    # 3. Reward normalization
    if cfg.normalize_reward:
        rewards, rew_state = rew_norm.update(
            seq.rew,
            dones,
            state.rew_norm,
        )
        state = replace(state, rew_norm=rew_state)
    else:
        rewards = seq.rew

    # 4. Off-policy Q(λ) returns — V_soft = V_θ + const (log det P cancels)
    discount_t = cfg.gamma * (1.0 - seq.term.astype(jnp.float32))
    v_t = jax.vmap(jax.vmap(state.agent.v_net))(nobs_n).squeeze(-1)
    q_t = jax.vmap(jax.vmap(lambda o, a: agent.q_val(o, a, state.agent)))(
        obs_n[:, 1:], seq.act[:, 1:]
    )
    c_t = cfg.trace_lambda * (1.0 - dones[:, :-1].astype(jnp.float32))

    targets = jax.vmap(
        lambda q, v, r, d, c: rlax.general_off_policy_returns_from_q_and_v(
            q, v, r, d, c, stop_target_gradients=True
        )
    )(q_t, v_t, rewards, discount_t, c_t)

    # 5. Flatten batch
    batch_size = cfg.n_envs * cfg.seq_len
    mb_size = batch_size // cfg.n_batches
    batch = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        Batch(obs=obs_n, act=seq.act, ret=targets),
    )

    # 6. Epoch loop: shuffle and minibatch SGD over fixed targets
    def epoch_step(state, _):
        key, perm_key = jrd.split(state.agent.key)
        state = replace(state, agent=replace(state.agent, key=key))
        shuffled = jax.tree.map(
            lambda x: x[jrd.permutation(perm_key, batch_size)], batch
        )

        def minibatch_step(state, start):
            mb = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size),
                shuffled,
            )
            params = (state.agent.v_net, state.agent.mu_net, state.agent.p_net)

            def loss_fn(params):
                v_net, mu_net, p_net = params
                s = replace(state.agent, v_net=v_net, mu_net=mu_net, p_net=p_net)
                q = jax.vmap(lambda o, a: agent.q_val(o, a, s))(mb.obs, mb.act)
                loss = jnp.mean((q - mb.ret) ** 2)
                return loss, dict(
                    q_loss=loss, q_mean=jnp.mean(q), ret_mean=jnp.mean(mb.ret)
                )

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, state.opt, params)
            new_v, new_mu, new_p = eqx.apply_updates(params, updates)
            return replace(
                state,
                agent=replace(
                    state.agent,
                    v_net=new_v,
                    mu_net=new_mu,
                    p_net=new_p,
                ),
                opt=opt_state,
            ), info

        starts = jnp.arange(cfg.n_batches) * mb_size
        state, infos = jax.lax.scan(minibatch_step, state, starts)
        return state, infos

    state, all_infos = jax.lax.scan(epoch_step, state, length=cfg.n_epochs)
    infos = jax.tree.map(jnp.mean, all_infos)
    infos["sam_rew"] = sam_metrics.avg_eps_rew

    # Diagnostics
    infos["act_mean"] = jnp.mean(jnp.abs(seq.act))
    infos["act_max"] = jnp.max(jnp.abs(seq.act))
    mu_vals = jax.vmap(jax.vmap(state.agent.mu_net))(obs_n)
    infos["mu_mean"] = jnp.mean(jnp.abs(mu_vals))
    infos["mu_max"] = jnp.max(jnp.abs(mu_vals))
    infos["v_mean"] = jnp.mean(v_t)
    infos["v_std"] = jnp.std(v_t)
    p_vals = jax.vmap(jax.vmap(lambda o: agent._precision(o, state.agent)))(obs_n)
    infos["p_mean"] = jnp.mean(p_vals)
    infos["ret_mean"] = jnp.mean(targets)
    infos["ret_std"] = jnp.std(targets)
    infos["rew_raw_mean"] = jnp.mean(seq.rew)
    infos["rew_norm_mean"] = jnp.mean(rewards)
    return state, infos


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _make_env(cfg: Config, num_envs: int):
    """Build the rollout env from the configured backend.

    ``gymnasium`` works for any env; ``envpool`` is fast CPU MuJoCo (its
    ``max_episode_steps`` is aligned to ``max_eps_len`` so its truncation
    matches ``Mc``'s).
    """
    if cfg.env_backend == "envpool":
        from jaxtor.env import envpool

        return envpool.make(
            cfg.env_id, num_envs=num_envs, max_episode_steps=cfg.max_eps_len
        )
    return gymnasium.make(cfg.env_id, num_envs=num_envs, async_envs=cfg.async_envs)


def train(cfg: Config) -> State:
    """Train NAF and return the final training state."""
    key = jrd.PRNGKey(cfg.seed)
    key, v_key, mu_key, a_key, env_key, agent_key, eval_key = jrd.split(key, 7)

    env = _make_env(cfg, cfg.n_envs)
    eval_env = _make_env(cfg, cfg.eval_envs)
    (obs_dim,) = env.obs_shape
    (act_dim,) = env.act_shape

    total_updates = cfg.n_iters * cfg.n_epochs * cfg.n_batches
    if cfg.lr_schedule == "linear":
        lr = optax.linear_schedule(cfg.lr, 0.0, total_updates)
    else:
        lr = cfg.lr
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    obs_rs = RunningStats(clip=10.0) if cfg.normalize_obs else None
    rew_norm = RewardNorm(
        gamma=cfg.gamma,
        rms=RunningStats(),
        seq_axis=1,
        clip=10.0,
    )
    agent = Agent(
        deterministic=False,
        noise_eps=cfg.noise_eps,
        noise_scale=cfg.noise_scale,
        obs_norm=obs_rs,
    )
    eval_agent = Agent(deterministic=True, obs_norm=obs_rs)

    train_mc = VecMc(
        mc=Mc(
            max_eps_len=cfg.max_eps_len,
            env=env,
        )
    )
    roll = Roll(
        seq_len=cfg.seq_len,
        seq_axis=1,
        imc=Imc(agent=agent, mc=train_mc),
    )
    stats = EpisodeStats(seq_axis=1)

    eval_mc = VecMc(
        mc=Mc(
            max_eps_len=cfg.max_eps_len,
            env=eval_env,
        )
    )
    evaluator = Evaluator(
        episode_len=cfg.max_eps_len,
        imc=Imc(agent=eval_agent, mc=eval_mc),
    )

    v_net = MLP(
        obs_dim,
        cfg.v_hiddens,
        1,
        output_gain=1.0,
        layer_norm=cfg.layer_norm,
        key=v_key,
    )
    mu_net = MLP(
        obs_dim,
        cfg.mu_hiddens,
        act_dim,
        output_gain=0.01,
        layer_norm=cfg.layer_norm,
        key=mu_key,
    )
    p_net = MLP(
        obs_dim,
        cfg.p_hiddens,
        act_dim,
        output_gain=1.0,
        layer_norm=cfg.layer_norm,
        key=a_key,
    )
    if (
        not isinstance(v_net, MLP)
        or not isinstance(mu_net, MLP)
        or not isinstance(p_net, MLP)
    ):
        raise TypeError("Equinox returned an unexpected module type")

    agent_state = Agent.State(
        key=agent_key,
        v_net=v_net,
        mu_net=mu_net,
        p_net=p_net,
        obs_stats=RunningStats.State(
            mean=jnp.zeros(obs_dim),
            var=jnp.ones(obs_dim),
            count=jnp.float32(1e-4),
        ),
    )

    state = State(
        mc=train_mc.init(jrd.split(key, cfg.n_envs), env.init(env_key)),
        agent=agent_state,
        stats=stats.init(batch_shape=(cfg.n_envs,)),
        rew_norm=rew_norm.init(batch_shape=(cfg.n_envs,)),
        opt=tx.init(eqx.filter((v_net, mu_net, p_net), eqx.is_inexact_array)),
    )

    @jax.jit
    def step(state):
        return train_step(
            state,
            agent=agent,
            roll=roll,
            stats=stats,
            rew_norm=rew_norm,
            tx=tx,
            cfg=cfg,
        )

    @jax.jit
    def evaluate(imc_state):
        return evaluator.evaluate(imc_state)

    print(f"[bold green]{cfg.env_id} NAF[/bold green]")
    t0 = time.time()

    for i in track(range(cfg.n_iters), description="Training"):
        state, metrics = step(state)

        if (i + 1) % cfg.eval_freq == 0:
            eval_key, e_env_key, k = jrd.split(eval_key, 3)
            eval_mc_state = eval_mc.init(
                jrd.split(k, cfg.eval_envs), eval_env.init(e_env_key)
            )
            m, eval_state = evaluate(
                evaluator.imc.init(
                    eval_mc_state,
                    replace(state.agent, key=eval_key),
                )
            )
            steps = (i + 1) * cfg.n_envs * cfg.seq_len
            print(
                f"  iter {i + 1:4d}  q_loss={float(metrics['q_loss']):.4f}"
                f"  sam_rew={float(metrics['sam_rew']):.1f}"
                f"  rew={float(m.avg_eps_rew):.1f}"
                f"\u00b1{float(m.std_eps_rew):.1f}"
                f"  len={float(m.avg_eps_len):.1f}"
                f"  steps={steps:,}"
                f"\n         |a|={float(metrics['act_mean']):.2f}"
                f"  |a|_max={float(metrics['act_max']):.1f}"
                f"  |μ|={float(metrics['mu_mean']):.2f}"
                f"  |μ|_max={float(metrics['mu_max']):.1f}"
                f"  V={float(metrics['v_mean']):.1f}±{float(metrics['v_std']):.1f}"
                f"  P={float(metrics['p_mean']):.2f}"
                f"\n         ret={float(metrics['ret_mean']):.1f}±{float(metrics['ret_std']):.1f}"
                f"  rew_raw={float(metrics['rew_raw_mean']):.3f}"
                f"  rew_norm={float(metrics['rew_norm_mean']):.3f}"
            )
            eval_env.close(eval_state.mc.env)

    elapsed = time.time() - t0
    print(f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s")

    env.close(state.mc.env)
    return state


if __name__ == "__main__":
    train(tyro.cli(Config))
