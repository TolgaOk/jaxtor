"""PPO on Hopper-v5 (Gymnasium/MuJoCo) using `jaxtor`, `rlax`, `optax`, `equinox`.

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

- `jaxtor`:
  - rollout sampler: `GymEnv -> Mc -> VecMc -> Imc -> Roll` for parallel rollouts.
  - evaluation: `GymEnv -> Mc -> VecMc -> Imc -> Eval` for evaluation.
- `equinox`: separate actor and critic MLPs.
- `optax`: Adam optimizer with gradient clipping and LR annealing.
- `rlax`: `clipped_surrogate_pg_loss`.

"""

from __future__ import annotations

import time
from functools import partial

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


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    n_iters: int = 500
    n_envs: int = 16
    seqlen: int = 128
    max_episode_len: int = 1000
    seed: int = 0

    hidden_dim: int = 64
    n_layers: int = 2

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
    eval_freq: int = 5
    eval_envs: int = 10
    async_envs: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Networks (orthogonal init per Huang et al. 2022)
# ──────────────────────────────────────────────────────────────────────────────


def _ortho_init(key: jax.Array, shape: tuple, gain: float = 1.0) -> jax.Array:
    """Orthogonal weight initialization."""
    return jax.nn.initializers.orthogonal(scale=gain)(key, shape, jnp.float32)


class MLP(eqx.Module):
    """Multi-layer perceptron with tanh activations and orthogonal init."""

    layers: list

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        *,
        output_gain: float = jnp.sqrt(2),
        key: jax.Array,
    ):
        keys = jrd.split(key, n_layers + 1)
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            gain = output_gain if i == len(dims) - 2 else jnp.sqrt(2)
            w = _ortho_init(keys[i], (dims[i], dims[i + 1]), gain)
            b = jnp.zeros(dims[i + 1])
            self.layers.append((w, b))

    def __call__(self, x: jax.Array) -> jax.Array:
        for w, b in self.layers[:-1]:
            x = jnp.tanh(x @ w + b)
        w, b = self.layers[-1]
        return x @ w + b


class Actor(eqx.Module):
    """Gaussian actor with learnable log-std (output gain 0.01)."""

    net: MLP
    log_std: jax.Array

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        n_layers: int,
        *,
        key: jax.Array,
    ):
        self.net = MLP(
            obs_dim, hidden_dim, act_dim, n_layers, output_gain=0.01, key=key
        )
        self.log_std = jnp.zeros(act_dim)

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return (mean, log_std) for the Gaussian policy."""
        mean = self.net(obs)
        log_std = jnp.clip(self.log_std, -20.0, 2.0)
        return mean, log_std


class Critic(eqx.Module):
    """Value function critic (output gain 1.0)."""

    net: MLP

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        n_layers: int,
        *,
        key: jax.Array,
    ):
        self.net = MLP(obs_dim, hidden_dim, 1, n_layers, output_gain=1.0, key=key)

    def __call__(self, obs: jax.Array) -> jax.Array:
        """Return scalar value estimate."""
        return self.net(obs).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Running statistics (Welford online algorithm)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RunningStats:
    """Running mean/var for observation or reward normalization."""

    mean: chex.Array
    var: chex.Array
    count: chex.Numeric


def update_stats(stats: RunningStats, batch: chex.Array) -> RunningStats:
    """Update running mean/var with a batch of samples.

    Args:
        stats: Current running statistics.
        batch: Flat samples, shape (N,) or (N, dim).
    """
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    delta = batch_mean - stats.mean
    total = stats.count + batch_count
    new_mean = stats.mean + delta * batch_count / total
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * stats.count * batch_count / total
    new_var = m2 / total
    return RunningStats(mean=new_mean, var=new_var, count=total)


def normalize_obs(obs: chex.Array, stats: RunningStats) -> chex.Array:
    """Normalize and clip observations to [-10, 10]."""
    normed = (obs - stats.mean) / jnp.sqrt(stats.var + 1e-8)
    return jnp.clip(normed, -10.0, 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# Reward normalization (divide by std of rolling discounted return)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RewardNorm:
    """State for reward normalization via rolling discounted return."""

    ret: chex.Array
    rms: RunningStats


def normalize_rewards(
    rewards: chex.Array,
    dones: chex.Array,
    rew_norm: RewardNorm,
    gamma: float,
) -> tuple[chex.Array, RewardNorm]:
    """Normalize rewards by std of rolling discounted return.

    Args:
        rewards: Shape (n_envs, seqlen).
        dones: Shape (n_envs, seqlen), term | trun.
        rew_norm: Current reward normalization state.
        gamma: Discount factor.

    Returns:
        Normalized rewards and updated RewardNorm.
    """

    def scan_fn(ret, step_data):
        rew, done = step_data
        ret = ret * gamma * (1.0 - done) + rew
        return ret, ret

    rew_t = jnp.transpose(rewards)
    done_t = jnp.transpose(dones.astype(jnp.float32))
    final_ret, all_rets = jax.lax.scan(scan_fn, rew_norm.ret, (rew_t, done_t))

    rms = update_stats(rew_norm.rms, all_rets.reshape(-1))
    norm_rewards = rewards / jnp.sqrt(rms.var + 1e-8)
    norm_rewards = jnp.clip(norm_rewards, -10.0, 10.0)

    return norm_rewards, RewardNorm(ret=final_ret, rms=rms)


# ──────────────────────────────────────────────────────────────────────────────
# Agent (satisfies jaxtor Agent protocol)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """Gaussian policy agent for continuous control."""

    deterministic: bool = False
    use_obs_norm: bool = True

    @dataclass
    class State:
        key: jax.Array
        actor: Actor
        obs_stats: RunningStats

    def act(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[chex.Array, Agent.State]:
        """Sample action from Gaussian policy.

        Returns raw (unclipped) Gaussian sample so that log-probability
        computation in the PPO loss is consistent with the sampling
        distribution. The environment handles action clamping.
        """
        if self.use_obs_norm:
            obs = normalize_obs(obs, state.obs_stats)
        mean, log_std = state.actor(obs)
        if self.deterministic:
            return mean, state
        key, noise_key = jrd.split(state.key)
        std = jnp.exp(log_std)
        action = mean + std * jrd.normal(noise_key, mean.shape)
        return action, state.replace(key=key)


# ──────────────────────────────────────────────────────────────────────────────
# Log-probability helpers
# ──────────────────────────────────────────────────────────────────────────────


def gaussian_log_prob(
    mean: chex.Array, log_std: chex.Array, action: chex.Array
) -> chex.Numeric:
    """Gaussian log-probability, summed over action dims."""
    std = jnp.exp(log_std)
    log_p = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
    return jnp.sum(log_p, axis=-1)


def gaussian_entropy(log_std: chex.Array) -> chex.Numeric:
    """Gaussian entropy, summed over action dims."""
    return jnp.sum(0.5 * (1.0 + jnp.log(2 * jnp.pi) + 2 * log_std), axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────────────────────────────────────


def compute_gae(
    rewards: chex.Array,
    dones: chex.Array,
    values: chex.Array,
    last_value: chex.Numeric,
    gamma: float,
    gae_lambda: float,
) -> tuple[chex.Array, chex.Array]:
    """Compute GAE advantages and returns for a single environment trajectory.

    Uses dones = term | trun to zero both bootstrap and GAE carry at episode
    boundaries (standard PPO approach per CleanRL).

    Args:
        rewards: Shape (seqlen,).
        dones: Shape (seqlen,), episode boundary flags (term | trun).
        values: Shape (seqlen,), value estimates.
        last_value: Bootstrap value for the last observation.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        advantages and returns, both shape (seqlen,).
    """
    not_done = 1.0 - dones.astype(jnp.float32)
    values_with_bootstrap = jnp.concatenate([values, last_value[None]])
    deltas = (
        rewards
        + gamma * not_done * values_with_bootstrap[1:]
        - values_with_bootstrap[:-1]
    )

    def scan_fn(carry, t):
        gae = deltas[t] + gamma * gae_lambda * not_done[t] * carry
        return gae, gae

    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.float32(0.0),
        jnp.arange(rewards.shape[0] - 1, -1, -1),
    )
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


# ──────────────────────────────────────────────────────────────────────────────
# PPO loss
# ──────────────────────────────────────────────────────────────────────────────


def ppo_loss(
    actor: Actor,
    critic: Critic,
    obs: chex.Array,
    actions: chex.Array,
    old_log_probs: chex.Array,
    advantages: chex.Array,
    returns: chex.Array,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
) -> tuple[chex.Numeric, dict]:
    """PPO clipped objective for a minibatch.

    Advantage normalization is applied per-minibatch (not full batch).
    """
    mean, log_std = jax.vmap(actor)(obs)
    new_log_probs = gaussian_log_prob(mean, log_std, actions)
    entropy = jnp.mean(gaussian_entropy(log_std))

    mb_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    pg_loss = rlax.clipped_surrogate_pg_loss(ratio, mb_adv, clip_eps)

    values = jax.vmap(critic)(obs)
    vf_loss = jnp.mean((values - returns) ** 2)

    total_loss = pg_loss + vf_coef * vf_loss - entropy_coef * entropy
    return total_loss, dict(pg_loss=pg_loss, vf_loss=vf_loss, entropy=entropy)


# ──────────────────────────────────────────────────────────────────────────────
# Training state and step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainState:
    """Full training state."""

    imc: Imc.State
    critic: Critic
    obs_stats: RunningStats
    rew_norm: RewardNorm
    actor_opt: optax.OptState
    critic_opt: optax.OptState


def train_step(
    train_state: TrainState,
    *,
    agent: Agent,
    roll: Roll,
    actor_tx: optax.GradientTransformation,
    critic_tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[TrainState, dict]:
    """One PPO iteration: collect rollout, compute GAE, run minibatch updates."""
    imc_state = train_state.imc
    actor = imc_state.agent.actor
    critic = train_state.critic
    obs_stats = train_state.obs_stats
    rew_norm = train_state.rew_norm

    # 1. Collect rollout: (n_envs, seqlen, ...)
    trans, imc_state = roll.sample(imc_state)

    # 2. Episode boundary mask
    dones = jnp.logical_or(trans.term, trans.trun)

    # 3. Update obs stats and normalize
    if agent.use_obs_norm:
        flat_obs = trans.obs.reshape(-1, trans.obs.shape[-1])
        obs_stats = update_stats(obs_stats, flat_obs)
        imc_state = eqx.tree_at(lambda s: s.agent.obs_stats, imc_state, obs_stats)

    norm_fn = (
        partial(normalize_obs, stats=obs_stats) if agent.use_obs_norm else lambda x: x
    )
    obs_normalized = jax.vmap(jax.vmap(norm_fn))(trans.obs)
    nobs_normalized = jax.vmap(jax.vmap(norm_fn))(trans.nobs)

    # 4. Normalize rewards
    rewards = trans.rew
    if cfg.normalize_reward:
        rewards, rew_norm = normalize_rewards(rewards, dones, rew_norm, cfg.gamma)

    # 5. Compute values and GAE
    values = jax.vmap(jax.vmap(critic))(obs_normalized)
    last_values = jax.vmap(critic)(nobs_normalized[:, -1])
    last_dones = dones[:, -1]
    last_values = last_values * (1.0 - last_dones.astype(jnp.float32))

    advantages, returns = jax.vmap(
        partial(compute_gae, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    )(rewards, dones, values, last_values)

    # 6. Old log-probs (from unclipped actions stored in transition)
    old_mean, old_log_std = jax.vmap(jax.vmap(actor))(obs_normalized)
    old_log_probs = gaussian_log_prob(old_mean, old_log_std, trans.act)

    # 7. Flatten for minibatch updates
    batch_size = cfg.n_envs * cfg.seqlen
    flat_obs = obs_normalized.reshape(batch_size, -1)
    flat_actions = trans.act.reshape(batch_size, -1)
    flat_old_lp = old_log_probs.reshape(batch_size)
    flat_adv = advantages.reshape(batch_size)
    flat_ret = returns.reshape(batch_size)

    # 8. Minibatch PPO updates
    mb_size = batch_size // cfg.n_minibatches
    actor_opt_state = train_state.actor_opt
    critic_opt_state = train_state.critic_opt
    key = imc_state.agent.key

    def epoch_step(carry, _):
        actor, critic, actor_opt_state, critic_opt_state, key = carry
        key, perm_key = jrd.split(key)
        perm = jrd.permutation(perm_key, batch_size)
        shuffled = (
            flat_obs[perm],
            flat_actions[perm],
            flat_old_lp[perm],
            flat_adv[perm],
            flat_ret[perm],
        )

        def minibatch_step(carry, start_idx):
            actor, critic, a_opt, c_opt = carry
            mb_obs = jax.lax.dynamic_slice(
                shuffled[0], (start_idx, 0), (mb_size, flat_obs.shape[1])
            )
            mb_act = jax.lax.dynamic_slice(
                shuffled[1], (start_idx, 0), (mb_size, flat_actions.shape[1])
            )
            mb_old_lp = jax.lax.dynamic_slice_in_dim(shuffled[2], start_idx, mb_size)
            mb_adv = jax.lax.dynamic_slice_in_dim(shuffled[3], start_idx, mb_size)
            mb_ret = jax.lax.dynamic_slice_in_dim(shuffled[4], start_idx, mb_size)

            (loss, info), (actor_grads, critic_grads) = jax.value_and_grad(
                ppo_loss, argnums=(0, 1), has_aux=True
            )(
                actor,
                critic,
                mb_obs,
                mb_act,
                mb_old_lp,
                mb_adv,
                mb_ret,
                cfg.clip_eps,
                cfg.vf_coef,
                cfg.entropy_coef,
            )

            a_updates, a_opt = actor_tx.update(actor_grads, a_opt, actor)
            actor = eqx.apply_updates(actor, a_updates)
            c_updates, c_opt = critic_tx.update(critic_grads, c_opt, critic)
            critic = eqx.apply_updates(critic, c_updates)
            return (actor, critic, a_opt, c_opt), loss

        starts = jnp.arange(cfg.n_minibatches) * mb_size
        (actor, critic, actor_opt_state, critic_opt_state), losses = jax.lax.scan(
            minibatch_step,
            (actor, critic, actor_opt_state, critic_opt_state),
            starts,
        )
        return (
            (actor, critic, actor_opt_state, critic_opt_state, key),
            jnp.mean(losses),
        )

    (actor, critic, actor_opt_state, critic_opt_state, key), epoch_losses = (
        jax.lax.scan(
            epoch_step,
            (actor, critic, actor_opt_state, critic_opt_state, key),
            length=cfg.n_epochs,
        )
    )

    imc_state = eqx.tree_at(lambda s: s.agent.actor, imc_state, actor)
    imc_state = eqx.tree_at(lambda s: s.agent.key, imc_state, key)

    new_train_state = TrainState(
        imc=imc_state,
        critic=critic,
        obs_stats=obs_stats,
        rew_norm=rew_norm,
        actor_opt=actor_opt_state,
        critic_opt=critic_opt_state,
    )
    metrics = dict(loss=jnp.mean(epoch_losses))
    return new_train_state, metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    cfg = tyro.cli(Config)

    key = jrd.PRNGKey(cfg.seed)
    key, actor_key, critic_key, env_key, agent_key, eval_key = jrd.split(key, 6)

    obs_dim = 11
    act_dim = 3

    # Networks
    actor = Actor(obs_dim, act_dim, cfg.hidden_dim, cfg.n_layers, key=actor_key)
    critic = Critic(obs_dim, cfg.hidden_dim, cfg.n_layers, key=critic_key)

    # Optimizers with linear LR annealing and Adam eps=1e-5
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
    actor_opt_state = actor_tx.init(actor)
    critic_opt_state = critic_tx.init(critic)

    # Running stats
    obs_stats = RunningStats(
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
        count=jnp.float32(1e-4),
    )
    rew_norm = RewardNorm(
        ret=jnp.zeros(cfg.n_envs),
        rms=RunningStats(
            mean=jnp.float32(0.0),
            var=jnp.float32(1.0),
            count=jnp.float32(1e-4),
        ),
    )

    # Env + sampler
    env = gymnasium.make("Hopper-v5", num_envs=cfg.n_envs, async_envs=cfg.async_envs)
    mc = Mc(max_episode_len=cfg.max_episode_len, queue_size=20, env=env)
    vec_mc = VecMc(mc=mc)
    behavior_agent = Agent(deterministic=False, use_obs_norm=cfg.normalize_obs)
    imc = Imc(agent=behavior_agent, mc=vec_mc)
    roll = Roll(imc=imc, seqlen=cfg.seqlen, seq_axis=1)

    # Eval
    eval_env = gymnasium.make(
        "Hopper-v5", num_envs=cfg.eval_envs, async_envs=cfg.async_envs
    )
    eval_mc = Mc(max_episode_len=cfg.max_episode_len, queue_size=20, env=eval_env)
    eval_vec_mc = VecMc(mc=eval_mc)
    eval_agent = Agent(deterministic=True, use_obs_norm=cfg.normalize_obs)
    eval_imc = Imc(agent=eval_agent, mc=eval_vec_mc)
    evaluator = Evaluator(imc=eval_imc, episode_len=cfg.max_episode_len)

    # Init states
    env_state = env.init(env_key)
    imc_state = Imc.State(
        mc=vec_mc.init(jrd.split(key, cfg.n_envs), env_state),
        agent=Agent.State(key=agent_key, actor=actor, obs_stats=obs_stats),
    )

    train_state = TrainState(
        imc=imc_state,
        critic=critic,
        obs_stats=obs_stats,
        rew_norm=rew_norm,
        actor_opt=actor_opt_state,
        critic_opt=critic_opt_state,
    )

    # JIT compile
    jit_train_step = jax.jit(
        partial(
            train_step,
            agent=behavior_agent,
            roll=roll,
            actor_tx=actor_tx,
            critic_tx=critic_tx,
            cfg=cfg,
        )
    )
    jit_eval = jax.jit(evaluator.metric)

    # Training loop
    print("[bold green]Hopper-v5 PPO[/bold green]")
    t0 = time.time()

    for i in track(range(cfg.n_iters), description="Training"):
        train_state, metrics = jit_train_step(train_state)

        if (i + 1) % cfg.eval_freq == 0:
            eval_key, e_env_key, k = jrd.split(eval_key, 3)
            eval_env_state = eval_env.init(e_env_key)
            eval_imc_state = Imc.State(
                mc=eval_vec_mc.init(jrd.split(k, cfg.eval_envs), eval_env_state),
                agent=Agent.State(
                    key=eval_key,
                    actor=train_state.imc.agent.actor,
                    obs_stats=train_state.obs_stats,
                ),
            )
            m = jit_eval(eval_imc_state)
            steps = (i + 1) * cfg.n_envs * cfg.seqlen
            print(
                f"  iter {i + 1:4d}  loss={float(metrics['loss']):+.4f}"
                f"  rew={float(m.avg_eps_rew):.1f}"
                f"\u00b1{float(m.std_eps_rew):.1f}"
                f"  len={float(m.avg_eps_len):.1f}"
                f"  steps={steps:,}"
            )

    elapsed = time.time() - t0
    print(
        f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
        f"  rew={float(m.avg_eps_rew):.1f}\u00b1{float(m.std_eps_rew):.1f}"
        f"  (over {int(m.n_episodes)} eps)"
    )

    env._vec_env.close()
    eval_env._vec_env.close()
