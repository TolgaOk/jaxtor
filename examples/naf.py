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

import time
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
from jaxtor.sampler import Imc, Mc, Roll, VecMc
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
    seqlen: int = 2048
    max_episode_len: int = 1000
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
    n_minibatches: int = 32

    normalize_obs: bool = True
    normalize_reward: bool = True
    layer_norm: bool = True
    eval_freq: int = 5
    eval_envs: int = 10
    async_envs: bool = True


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
        output_gain: float = jnp.sqrt(2),
        layer_norm: bool = False,
        key: jax.Array,
    ):
        keys = jrd.split(key, len(hiddens) + 1)
        dims = [in_dim, *hiddens, out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            gain = output_gain if i == len(dims) - 2 else jnp.sqrt(2)
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

    def _precision(self, obs: chex.Array, state: Agent.State) -> jax.Array:
        """Diagonal precision p = softplus(net(s)) + ε."""
        return jax.nn.softplus(state.p_net(obs)) + self.noise_eps

    def act(self, obs: chex.Array, state: Agent.State):
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs, state.obs_stats)
        mu_raw = state.mu_net(obs)
        if self.deterministic:
            action = jnp.tanh(mu_raw)
        else:
            key, k = jrd.split(state.key)
            p = self._precision(obs, state)
            eps = jrd.normal(k, mu_raw.shape)
            action = jnp.tanh(mu_raw + self.noise_scale * eps / jnp.sqrt(p))
            state = state.replace(key=key)
        return action, state

    def q_val(
        self, obs: chex.Array, act: chex.Array, state: Agent.State
    ) -> jax.Array:
        """Q(s,a) = V(s) - ½ Σᵢ pᵢ (aᵢ - μᵢ)²."""
        v = state.v_net(obs).squeeze(-1)
        mu = jnp.tanh(state.mu_net(obs))
        p = self._precision(obs, state)
        diff = act - mu
        return v - 0.5 * jnp.sum(p * diff ** 2, axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training state and step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Batch:
    """Flattened minibatch for NAF updates."""

    obs: chex.Array
    act: chex.Array
    ret: chex.Array


@dataclass
class State:
    """Flat training state — single source of truth, updated via .replace()."""

    mc: Mc.State
    agent: Agent.State
    rew_norm: RewardNorm.State
    opt: optax.OptState


def train_step(
    state: State,
    *,
    agent: Agent,
    roll: Roll,
    rew_norm: RewardNorm,
    tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[State, dict]:
    """One NAF iteration: collect rollout, compute Q(λ) targets, train Q."""
    # 1. Collect rollout
    trans, imc_state = roll.sample(Imc.State(mc=state.mc, agent=state.agent))
    sam_metrics, mc_state = roll.imc.mc.metrics(imc_state.mc)
    state = state.replace(mc=mc_state, agent=imc_state.agent)
    dones = jnp.logical_or(trans.term, trans.trun)

    # 2. Obs normalization
    if agent.obs_norm is not None:
        obs_stats = agent.obs_norm.update(
            trans.obs.reshape(-1, trans.obs.shape[-1]), state.agent.obs_stats
        )
        state = state.replace(agent=state.agent.replace(obs_stats=obs_stats))
        obs_n = agent.obs_norm.normalize(trans.obs, state.agent.obs_stats)
        nobs_n = agent.obs_norm.normalize(trans.nobs, state.agent.obs_stats)
    else:
        obs_n, nobs_n = trans.obs, trans.nobs

    # 3. Reward normalization
    if cfg.normalize_reward:
        rewards, rew_state = rew_norm.update(trans.rew, dones, state.rew_norm)
        state = state.replace(rew_norm=rew_state)
    else:
        rewards = trans.rew

    # 4. Off-policy Q(λ) returns — V_soft = V_θ + const (log det P cancels)
    discount_t = cfg.gamma * (1.0 - trans.term.astype(jnp.float32))
    v_t = jax.vmap(jax.vmap(state.agent.v_net))(nobs_n).squeeze(-1)
    q_t = jax.vmap(jax.vmap(
        lambda o, a: agent.q_val(o, a, state.agent)
    ))(obs_n[:, 1:], trans.act[:, 1:])
    c_t = cfg.trace_lambda * (1.0 - dones[:, :-1].astype(jnp.float32))

    targets = jax.vmap(
        lambda q, v, r, d, c: rlax.general_off_policy_returns_from_q_and_v(
            q, v, r, d, c, stop_target_gradients=True
        )
    )(q_t, v_t, rewards, discount_t, c_t)

    # 5. Flatten batch
    batch_size = cfg.n_envs * cfg.seqlen
    mb_size = batch_size // cfg.n_minibatches
    batch = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        Batch(obs=obs_n, act=trans.act, ret=targets),
    )

    # 6. Epoch loop: shuffle and minibatch SGD over fixed targets
    def epoch_step(state, _):
        key, perm_key = jrd.split(state.agent.key)
        state = state.replace(agent=state.agent.replace(key=key))
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
                s = state.agent.replace(v_net=v_net, mu_net=mu_net, p_net=p_net)
                q = jax.vmap(lambda o, a: agent.q_val(o, a, s))(mb.obs, mb.act)
                loss = jnp.mean((q - mb.ret) ** 2)
                return loss, dict(
                    q_loss=loss, q_mean=jnp.mean(q), ret_mean=jnp.mean(mb.ret)
                )

            (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, state.opt, params)
            new_v, new_mu, new_p = eqx.apply_updates(params, updates)
            return state.replace(
                agent=state.agent.replace(v_net=new_v, mu_net=new_mu, p_net=new_p),
                opt=opt_state,
            ), info

        starts = jnp.arange(cfg.n_minibatches) * mb_size
        state, infos = jax.lax.scan(minibatch_step, state, starts)
        return state, infos

    state, all_infos = jax.lax.scan(epoch_step, state, length=cfg.n_epochs)
    infos = jax.tree.map(jnp.mean, all_infos)
    infos["sam_rew"] = sam_metrics.avg_eps_rew

    # Diagnostics
    infos["act_mean"] = jnp.mean(jnp.abs(trans.act))
    infos["act_max"] = jnp.max(jnp.abs(trans.act))
    mu_vals = jax.vmap(jax.vmap(state.agent.mu_net))(obs_n)
    infos["mu_mean"] = jnp.mean(jnp.abs(mu_vals))
    infos["mu_max"] = jnp.max(jnp.abs(mu_vals))
    infos["v_mean"] = jnp.mean(v_t)
    infos["v_std"] = jnp.std(v_t)
    p_vals = jax.vmap(jax.vmap(lambda o: agent._precision(o, state.agent)))(obs_n)
    infos["p_mean"] = jnp.mean(p_vals)
    infos["ret_mean"] = jnp.mean(targets)
    infos["ret_std"] = jnp.std(targets)
    infos["rew_raw_mean"] = jnp.mean(trans.rew)
    infos["rew_norm_mean"] = jnp.mean(rewards)
    return state, infos


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def train(cfg: Config) -> State:
    """Train NAF and return the final training state."""
    key = jrd.PRNGKey(cfg.seed)
    key, v_key, mu_key, a_key, env_key, agent_key, eval_key = jrd.split(key, 7)

    env = gymnasium.make(cfg.env_id, num_envs=cfg.n_envs, async_envs=cfg.async_envs)
    eval_env = gymnasium.make(
        cfg.env_id, num_envs=cfg.eval_envs, async_envs=cfg.async_envs
    )
    (obs_dim,) = env._obs_shape
    (act_dim,) = env._act_shape

    total_updates = cfg.n_iters * cfg.n_epochs * cfg.n_minibatches
    if cfg.lr_schedule == "linear":
        lr = optax.linear_schedule(cfg.lr, 0.0, total_updates)
    else:
        lr = cfg.lr
    tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    obs_rs = RunningStats(clip=10.0) if cfg.normalize_obs else None
    rew_norm = RewardNorm(gamma=cfg.gamma, rms=RunningStats(), clip=10.0)
    agent = Agent(
        deterministic=False,
        noise_eps=cfg.noise_eps,
        noise_scale=cfg.noise_scale,
        obs_norm=obs_rs,
    )
    eval_agent = Agent(deterministic=True, obs_norm=obs_rs)

    roll = Roll(
        seqlen=cfg.seqlen,
        seq_axis=1,
        imc=Imc(
            agent=agent,
            mc=VecMc(
                mc=Mc(
                    max_episode_len=cfg.max_episode_len,
                    queue_size=20,
                    env=env,
                )
            ),
        ),
    )

    evaluator = Evaluator(
        episode_len=cfg.max_episode_len,
        imc=Imc(
            agent=eval_agent,
            mc=VecMc(
                mc=Mc(
                    max_episode_len=cfg.max_episode_len,
                    queue_size=20,
                    env=eval_env,
                )
            ),
        ),
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
        mc=roll.imc.mc.init(jrd.split(key, cfg.n_envs), env.init(env_key)),
        agent=agent_state,
        rew_norm=RewardNorm.State(
            ret=jnp.zeros(cfg.n_envs),
            rms=RunningStats.State(
                mean=jnp.float32(0.0),
                var=jnp.float32(1.0),
                count=jnp.float32(1e-4),
            ),
        ),
        opt=tx.init((v_net, mu_net, p_net)),
    )

    @jax.jit
    def step(state):
        return train_step(
            state,
            agent=agent,
            roll=roll,
            rew_norm=rew_norm,
            tx=tx,
            cfg=cfg,
        )

    @jax.jit
    def evaluate(imc_state):
        return evaluator.metric(imc_state)

    print(f"[bold green]{cfg.env_id} NAF[/bold green]")
    t0 = time.time()

    for i in track(range(cfg.n_iters), description="Training"):
        state, metrics = step(state)

        if (i + 1) % cfg.eval_freq == 0:
            eval_key, e_env_key, k = jrd.split(eval_key, 3)
            eval_mc = evaluator.imc.mc.init(
                jrd.split(k, cfg.eval_envs), eval_env.init(e_env_key)
            )
            m = evaluate(
                Imc.State(mc=eval_mc, agent=state.agent.replace(key=eval_key))
            )
            steps = (i + 1) * cfg.n_envs * cfg.seqlen
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

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"  rew={float(m.avg_eps_rew):.1f}\u00b1{float(m.std_eps_rew):.1f}"
        f"  (over {int(m.n_episodes)} eps)"
    )

    env._vec_env.close()
    eval_env._vec_env.close()
    return state


if __name__ == "__main__":
    train(tyro.cli(Config))
