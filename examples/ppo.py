"""PPO on Hopper-v5 (Gymnasium/MuJoCo) using jaxtor, distrax, rlax, optax, equinox.

Proximal Policy Optimization from Schulman et al. (2017) "Proximal Policy
Optimization Algorithms", arXiv:1707.06347.

Implementation follows the 9 MuJoCo-specific details from Huang et al. (2022)
"The 37 Implementation Details of Proximal Policy Optimization", ICLR Blog:

1. Orthogonal initialization (sqrt(2) hidden, 0.01 actor, 1.0 critic)
2. State-independent log-std initialized to 0
3. Observation normalization with clipping [-10, 10]
4. Reward normalization (divide by std of rolling discounted return)
5. No action clipping in agent (raw Gaussian sample stored for log-prob)
6. Separate actor/critic networks, tanh activations
7. Linear learning rate annealing to 0
8. Adam epsilon = 1e-5
9. Per-minibatch advantage normalization

- jaxtor:
  - rollout sampler: GymEnv -> Mc -> VecMc -> MuImc -> Roll
  - evaluation: GymEnv -> Mc -> VecMc -> Imc -> Eval
- equinox: separate actor and critic MLPs.
- optax: Adam optimizer with gradient clipping and LR annealing.
- distrax: Gaussian policy distribution (sampling, log-prob, entropy).
- rlax: clipped_surrogate_pg_loss, truncated_generalized_advantage_estimation.

"""

from __future__ import annotations

import time

import chex
import distrax
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
from jaxtor.sampler import Imc, Mc, MuImc, Roll, VecMc
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    env_id: str = "Hopper-v5"
    n_iters: int = 500
    n_envs: int = 16
    seqlen: int = 2048
    max_episode_len: int = 1000
    seed: int = 0

    actor_hiddens: tuple[int, ...] = (64, 64)
    critic_hiddens: tuple[int, ...] = (128, 128)

    lr: float = 3e-4
    max_grad_norm: float = 0.5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.0
    vf_coef: float = 0.5

    n_epochs: int = 10
    n_minibatches: int = 32

    normalize_obs: bool = True
    normalize_reward: bool = True
    normalize_adv: bool = False
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
# Agent (unified behavior + eval)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """Gaussian policy agent for continuous control.

    - deterministic=False, satisfies MuImc protocol: act -> (action, log_prob, state).
    - deterministic=True, satisfies Imc protocol: act -> (action, state).
    """

    deterministic: bool = False
    obs_norm: RunningStats | None = None

    @dataclass
    class State:
        key: jax.Array
        actor: MLP
        log_std: jax.Array
        critic: MLP
        obs_stats: RunningStats.State

    def act(self, obs: chex.Array, state: Agent.State):
        if self.obs_norm is not None:
            obs = self.obs_norm.normalize(obs, state.obs_stats)
        mean = state.actor(obs)
        log_std = jnp.clip(state.log_std, -20.0, 2.0)
        if self.deterministic:
            return mean, state
        key, k = jrd.split(state.key)
        dist = distrax.Normal(loc=mean, scale=jnp.exp(log_std))
        action, log_prob = dist.sample_and_log_prob(seed=k)
        return action, log_prob.sum(-1), state.replace(key=key)


# ──────────────────────────────────────────────────────────────────────────────
# PPO loss
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Batch:
    """Flattened minibatch for PPO updates."""

    obs: chex.Array
    act: chex.Array
    old_lp: chex.Array
    adv: chex.Array
    ret: chex.Array


def ppo_loss(
    policy_params: tuple[MLP, jax.Array],
    critic: MLP,
    mb: Batch,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
    normalize_adv: bool = True,
) -> tuple[chex.Numeric, dict]:
    """PPO clipped objective with optional per-minibatch advantage normalization."""
    actor, log_std = policy_params
    mean = jax.vmap(actor)(mb.obs)
    log_std = jnp.clip(log_std, -20.0, 2.0)
    dist = distrax.Normal(loc=mean, scale=jnp.exp(log_std))
    new_lp = dist.log_prob(mb.act).sum(-1)
    entropy = jnp.mean(dist.entropy().sum(-1))

    if normalize_adv:
        adv = (mb.adv - mb.adv.mean()) / (mb.adv.std() + 1e-8)
    else:
        adv = mb.adv
    ratio = jnp.exp(new_lp - mb.old_lp)
    pg_loss = rlax.clipped_surrogate_pg_loss(ratio, adv, clip_eps)

    values = jax.vmap(critic)(mb.obs).squeeze(-1)
    vf_loss = jnp.mean((values - mb.ret) ** 2)

    loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy
    return loss, dict(
        pg_loss=pg_loss,
        vf_loss=vf_loss,
        entropy=entropy,
        clip_frac=jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        approx_kl=jnp.mean((ratio - 1.0) - jnp.log(ratio)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training state and step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class State:
    """Flat training state — single source of truth, updated via .replace()."""

    mc: Mc.State
    agent: Agent.State
    rew_norm: RewardNorm.State
    actor_opt: optax.OptState
    critic_opt: optax.OptState


def train_step(
    state: State,
    *,
    agent: Agent,
    roll: Roll,
    rew_norm: RewardNorm,
    actor_tx: optax.GradientTransformation,
    critic_tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[State, dict]:
    """One PPO iteration: collect rollout, compute GAE, run minibatch updates."""
    # 1. Collect rollout
    trans, imc_state = roll.sample(MuImc.State(mc=state.mc, agent=state.agent))
    state = state.replace(mc=imc_state.mc, agent=imc_state.agent)
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

    # 4. GAE
    values = jax.vmap(jax.vmap(state.agent.critic))(obs_n).squeeze(-1)
    last_val = jax.vmap(state.agent.critic)(nobs_n[:, -1]).squeeze(-1)
    values_tp1 = jnp.concatenate([values, last_val[:, None]], axis=1)
    discount_t = cfg.gamma * (1.0 - trans.term.astype(jnp.float32))

    advantages = jax.vmap(
        lambda r, d, v: rlax.truncated_generalized_advantage_estimation(
            r, d, cfg.gae_lambda, v
        )
    )(rewards, discount_t, values_tp1)
    returns = advantages + values

    # 5. Flatten batch
    batch_size = cfg.n_envs * cfg.seqlen
    mb_size = batch_size // cfg.n_minibatches
    batch = jax.tree.map(
        lambda x: x.reshape(batch_size, *x.shape[2:]),
        Batch(
            obs=obs_n, act=trans.act, old_lp=trans.log_mu, adv=advantages, ret=returns
        ),
    )

    # 6. Minibatch updates
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
            policy_params = (state.agent.actor, state.agent.log_std)
            (loss, info), (p_grad, c_grad) = jax.value_and_grad(
                ppo_loss, argnums=(0, 1), has_aux=True
            )(
                policy_params,
                state.agent.critic,
                mb,
                cfg.clip_eps,
                cfg.vf_coef,
                cfg.entropy_coef,
                cfg.normalize_adv,
            )

            p_upd, actor_opt = actor_tx.update(p_grad, state.actor_opt, policy_params)
            c_upd, critic_opt = critic_tx.update(
                c_grad, state.critic_opt, state.agent.critic
            )
            new_actor, new_log_std = eqx.apply_updates(policy_params, p_upd)
            new_critic = eqx.apply_updates(state.agent.critic, c_upd)
            return state.replace(
                agent=state.agent.replace(
                    actor=new_actor, log_std=new_log_std, critic=new_critic
                ),
                actor_opt=actor_opt,
                critic_opt=critic_opt,
            ), info

        starts = jnp.arange(cfg.n_minibatches) * mb_size
        state, infos = jax.lax.scan(minibatch_step, state, starts)
        return state, infos

    state, all_infos = jax.lax.scan(epoch_step, state, length=cfg.n_epochs)
    return state, jax.tree.map(jnp.mean, all_infos)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _make_env(cfg: Config, num_envs: int):
    """Build the rollout env from the configured backend.

    ``gymnasium`` works for any env; ``envpool`` is fast CPU MuJoCo (its
    ``max_episode_steps`` is aligned to ``max_episode_len`` so its truncation
    matches ``Mc``'s).
    """
    if cfg.env_backend == "envpool":
        from jaxtor.env import envpool

        return envpool.make(
            cfg.env_id, num_envs=num_envs, max_episode_steps=cfg.max_episode_len
        )
    return gymnasium.make(cfg.env_id, num_envs=num_envs, async_envs=cfg.async_envs)


def train(cfg: Config) -> State:
    """Train PPO and return the final training state."""
    key = jrd.PRNGKey(cfg.seed)
    key, actor_key, critic_key, env_key, agent_key, eval_key = jrd.split(key, 6)

    env = _make_env(cfg, cfg.n_envs)
    eval_env = _make_env(cfg, cfg.eval_envs)
    (obs_dim,) = env.obs_shape
    (act_dim,) = env.act_shape

    total_updates = cfg.n_iters * cfg.n_epochs * cfg.n_minibatches
    lr_schedule = optax.linear_schedule(cfg.lr, 0.0, total_updates)
    actor_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )

    obs_rs = RunningStats(clip=10.0) if cfg.normalize_obs else None
    rew_norm = RewardNorm(gamma=cfg.gamma, rms=RunningStats(), clip=10.0)
    agent = Agent(deterministic=False, obs_norm=obs_rs)
    eval_agent = Agent(deterministic=True, obs_norm=obs_rs)

    roll = Roll(
        seqlen=cfg.seqlen,
        seq_axis=1,
        imc=MuImc(
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

    actor = MLP(
        obs_dim,
        cfg.actor_hiddens,
        act_dim,
        output_gain=0.01,
        layer_norm=cfg.layer_norm,
        key=actor_key,
    )
    log_std = jnp.zeros(act_dim)
    critic = MLP(
        obs_dim,
        cfg.critic_hiddens,
        1,
        output_gain=1.0,
        layer_norm=cfg.layer_norm,
        key=critic_key,
    )

    state = State(
        mc=roll.imc.mc.init(jrd.split(key, cfg.n_envs), env.init(env_key)),
        agent=Agent.State(
            key=agent_key,
            actor=actor,
            log_std=log_std,
            critic=critic,
            obs_stats=RunningStats.State(
                mean=jnp.zeros(obs_dim),
                var=jnp.ones(obs_dim),
                count=jnp.float32(1e-4),
            ),
        ),
        rew_norm=RewardNorm.State(
            ret=jnp.zeros(cfg.n_envs),
            rms=RunningStats.State(
                mean=jnp.float32(0.0),
                var=jnp.float32(1.0),
                count=jnp.float32(1e-4),
            ),
        ),
        actor_opt=actor_tx.init((actor, log_std)),
        critic_opt=critic_tx.init(critic),
    )

    @jax.jit
    def step(state):
        return train_step(
            state,
            agent=agent,
            roll=roll,
            rew_norm=rew_norm,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            cfg=cfg,
        )

    @jax.jit
    def evaluate(imc_state):
        return evaluator.metric(imc_state)

    print(f"[bold green]{cfg.env_id} PPO[/bold green]")
    t0 = time.time()

    for i in track(range(cfg.n_iters), description="Training"):
        state, metrics = step(state)

        if (i + 1) % cfg.eval_freq == 0:
            eval_key, e_env_key, k = jrd.split(eval_key, 3)
            eval_mc = evaluator.imc.mc.init(
                jrd.split(k, cfg.eval_envs), eval_env.init(e_env_key)
            )
            m = evaluate(Imc.State(mc=eval_mc, agent=state.agent.replace(key=eval_key)))
            steps = (i + 1) * cfg.n_envs * cfg.seqlen
            print(
                f"  iter {i + 1:4d}  loss={float(metrics['pg_loss']):+.4f}"
                f"  rew={float(m.avg_eps_rew):.1f}"
                f"\u00b1{float(m.std_eps_rew):.1f}"
                f"  len={float(m.avg_eps_len):.1f}"
                f"  steps={steps:,}"
            )
            eval_env.close(eval_mc.env)

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"  rew={float(m.avg_eps_rew):.1f}\u00b1{float(m.std_eps_rew):.1f}"
        f"  (over {int(m.n_episodes)} eps)"
    )

    env.close(state.mc.env)
    return state


if __name__ == "__main__":
    train(tyro.cli(Config))
