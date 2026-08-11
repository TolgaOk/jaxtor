"""PPO on Hopper-v5 with an explicit actor-critic component.

Proximal Policy Optimization follows Schulman et al. (2017) and the continuous
control details collected by Huang et al. (2022).

Components
----------
actor_critic: ActorCritic
├── policy: Gaussian MLP + log_std
├── value: MLP
└── obs_norm: ObservationNorm

rollout: Roll
└── Imc
    ├── agent: ActorCritic(stochastic)
    └── VecMc
        └── Mc
            └── train_env

evaluation: Evaluator
└── Imc
    ├── agent: ActorCritic(deterministic)
    └── VecMc
        └── Mc
            └── eval_env

The two trees contain equivalent static ActorCritic configurations and share
one dynamic ActorCritic.State.

State
-----
key: PPO update randomness.
rollout: Dynamic state consumed and returned by Roll.sample.
├── mc: Environment state and episode queues.
├── agent: Policy, value function, sampling key, and observation statistics.
└── dec: Cached next decision.
rew_norm: Rolling discounted-return statistics.
pi_opt, v_opt: Policy and value optimizer states.

Flow
----
trajectory[dec, mc, succ]
-> reward normalization
-> TD(lambda) from dec.value to succ.value
-> PpoBatch
-> ActorCritic.eval_pi[log_pi] + ActorCritic.value
-> PPO update
-> observation normalization
-> refresh cached decision

Axes
----
N: environments, T: rollout length, B: N * T.
"""

from __future__ import annotations

import time
from typing import Protocol

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
from jaxtor.sampler import Imc, Mc, Roll, VecMc
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
# Normalization contracts
# ──────────────────────────────────────────────────────────────────────────────


class ObservationNorm(Protocol):
    """Observation-normalization surface consumed by ActorCritic."""

    class State(Protocol): ...

    def init(self, shape: tuple[int, ...]) -> ObservationNorm.State: ...

    def update(
        self,
        batch: jax.Array,
        state: ObservationNorm.State,
    ) -> ObservationNorm.State: ...

    def norm(
        self,
        obs: jax.Array,
        state: ObservationNorm.State,
    ) -> jax.Array: ...


class RewardNormalizer(Protocol):
    """Reward-normalization surface consumed by the PPO recipe."""

    class State(Protocol): ...

    def init(self, n_envs: int) -> RewardNormalizer.State: ...

    def norm(
        self,
        rewards: jax.Array,
        done: jax.Array,
        state: RewardNormalizer.State,
    ) -> tuple[jax.Array, RewardNormalizer.State]: ...


@dataclass
class RunningObsNorm(RunningStats):
    """Expose ``RunningStats`` through PPO's normalization contract."""

    def init(self, shape: tuple[int, ...]) -> RunningStats.State:
        """Initialize unit statistics for one observation shape."""
        return self.State(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=jnp.float32(1e-4),
        )

    def norm(self, obs: jax.Array, state: RunningStats.State) -> jax.Array:
        """Normalize observations with the running statistics."""
        return self.normalize(obs, state)


@dataclass
class RunningRewNorm(RewardNorm):
    """Expose ``RewardNorm`` through PPO's normalization contract."""

    def init(self, n_envs: int) -> RewardNorm.State:
        """Initialize per-environment returns and scalar reward statistics."""
        return self.State(
            ret=jnp.zeros(n_envs),
            rms=RunningStats.State(
                mean=jnp.float32(0.0),
                var=jnp.float32(1.0),
                count=jnp.float32(1e-4),
            ),
        )

    def norm(
        self,
        rewards: jax.Array,
        done: jax.Array,
        state: RewardNorm.State,
    ) -> tuple[chex.Array, RewardNorm.State]:
        """Normalize rewards and update rolling-return statistics."""
        return self.update(rewards, done, state)


@dataclass
class IdentityObsNorm:
    """Leave observations unchanged without adding array leaves to state."""

    @dataclass
    class State:
        """Empty identity-normalizer state."""

    def init(self, shape: tuple[int, ...]) -> ObservationNorm.State:
        """Return the empty state; shape is accepted for API compatibility."""
        del shape
        return self.State()

    def update(
        self,
        batch: jax.Array,
        state: ObservationNorm.State,
    ) -> ObservationNorm.State:
        """Return the unchanged state."""
        del batch
        return state

    def norm(
        self,
        obs: jax.Array,
        state: ObservationNorm.State,
    ) -> jax.Array:
        """Return observations unchanged."""
        del state
        return obs


@dataclass
class IdentityRewNorm:
    """Leave rewards unchanged without adding array leaves to state."""

    @dataclass
    class State:
        """Empty identity-normalizer state."""

    def init(self, n_envs: int) -> RewardNormalizer.State:
        """Return the empty state; n_envs is accepted for API compatibility."""
        del n_envs
        return self.State()

    def norm(
        self,
        rewards: jax.Array,
        done: jax.Array,
        state: RewardNormalizer.State,
    ) -> tuple[jax.Array, RewardNormalizer.State]:
        """Return rewards and state unchanged."""
        del done
        return rewards, state


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
        output_gain: float = 2**0.5,
        layer_norm: bool = False,
        key: jax.Array,
    ):
        keys = jrd.split(key, len(hiddens) + 1)
        dims = [in_dim, *hiddens, out_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            gain = output_gain if i == len(dims) - 2 else 2**0.5
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
# Actor-critic
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ActorCritic:
    """Gaussian policy and value function used by collection, loss, and eval.

    Public dataclasses:
        State: Policy, value, random-key, and observation-normalizer state.
        Decision: Action, behavior log-probability, and value.
        PolicyEval: Log-probability and entropy for supplied actions.

    Public methods:
        init: Initialize the complete dynamic state.
        decide: Prepare a stochastic or deterministic decision.
        eval_pi: Evaluate policy statistics for PPO updates.
        value: Evaluate the critic.
        update_obs: Update observation statistics from a trajectory.
    """

    obs_dim: int
    act_dim: int
    actor_hiddens: tuple[int, ...]
    critic_hiddens: tuple[int, ...]
    layer_norm: bool
    obs_norm: ObservationNorm
    deterministic: bool = False

    @dataclass
    class PolicyState:
        """Gaussian policy parameters."""

        network: MLP
        log_std: jax.Array

    @dataclass
    class State:
        """Dynamic actor-critic state."""

        key: jax.Array
        policy: ActorCritic.PolicyState
        value: MLP
        obs_norm: ObservationNorm.State

    @dataclass
    class Decision:
        """Actor-critic data cached and collected by ``Imc``."""

        act: chex.Array
        log_mu: jax.Array
        value: jax.Array

    @dataclass
    class PolicyEval:
        """Policy statistics evaluated at supplied actions."""

        logp: jax.Array
        entropy: jax.Array

    def init(self, key: jax.Array) -> ActorCritic.State:
        """Initialize policy, value, RNG, and observation statistics."""
        key, actor_key, critic_key = jrd.split(key, 3)
        policy = self.PolicyState(
            network=MLP(
                self.obs_dim,
                self.actor_hiddens,
                self.act_dim,
                output_gain=0.01,
                layer_norm=self.layer_norm,
                key=actor_key,
            ),
            log_std=jnp.zeros(self.act_dim),
        )
        value = MLP(
            self.obs_dim,
            self.critic_hiddens,
            1,
            output_gain=1.0,
            layer_norm=self.layer_norm,
            key=critic_key,
        )
        return self.State(
            key=key,
            policy=policy,
            value=value,
            obs_norm=self.obs_norm.init((self.obs_dim,)),
        )

    def _norm_obs(
        self,
        obs: jax.Array,
        state: ActorCritic.State,
    ) -> jax.Array:
        """Normalize observations using the configured strategy."""
        return self.obs_norm.norm(obs, state.obs_norm)

    def _dist(
        self,
        obs: jax.Array,
        state: ActorCritic.State,
    ) -> distrax.Normal:
        """Construct the diagonal Gaussian policy distribution."""
        mean = state.policy.network(self._norm_obs(obs, state))
        log_std = jnp.clip(state.policy.log_std, -20.0, 2.0)
        return distrax.Normal(loc=mean, scale=jnp.exp(log_std))

    def decide(
        self,
        obs: jax.Array,
        state: ActorCritic.State,
    ) -> tuple[ActorCritic.Decision, ActorCritic.State]:
        """Prepare the action, behavior log-probability, and value at ``obs``."""
        dist = self._dist(obs, state)
        value = self.value(obs, state)
        if self.deterministic:
            action = dist.mode()
            log_mu = dist.log_prob(action).sum(-1)
            return self.Decision(act=action, log_mu=log_mu, value=value), state

        key, sample_key = jrd.split(state.key)
        action, log_mu = dist.sample_and_log_prob(seed=sample_key)
        dec = self.Decision(
            act=action,
            log_mu=log_mu.sum(-1),
            value=value,
        )
        return dec, state.replace(key=key)

    def eval_pi(
        self,
        obs: jax.Array,
        action: jax.Array,
        state: ActorCritic.State,
    ) -> ActorCritic.PolicyEval:
        """Evaluate log-probability and entropy for supplied actions."""
        dist = self._dist(obs, state)
        return self.PolicyEval(
            logp=dist.log_prob(action).sum(-1),
            entropy=dist.entropy().sum(-1),
        )

    def value(
        self,
        obs: jax.Array,
        state: ActorCritic.State,
    ) -> jax.Array:
        """Evaluate the value function."""
        return state.value(self._norm_obs(obs, state)).squeeze(-1)

    def update_obs(
        self,
        obs: jax.Array,
        state: ActorCritic.State,
    ) -> ActorCritic.State:
        """Update observation statistics from arbitrary leading axes."""
        batch = obs.reshape(-1, obs.shape[-1])
        obs_norm = self.obs_norm.update(batch, state.obs_norm)
        return state.replace(obs_norm=obs_norm)


# ──────────────────────────────────────────────────────────────────────────────
# PPO loss
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PpoBatch:
    """Flattened minibatch for PPO updates."""

    obs: jax.Array
    act: jax.Array
    log_mu: jax.Array
    adv: jax.Array
    ret: jax.Array


@dataclass
class PpoMetrics:
    """Transient metrics produced by one PPO update."""

    pi_loss: jax.Array
    v_loss: jax.Array
    entropy: jax.Array
    clip_frac: jax.Array
    approx_kl: jax.Array


def ppo_loss(
    ac_state: ActorCritic.State,
    *,
    ac: ActorCritic,
    batch: PpoBatch,
    clip_eps: float,
    vf_coef: float,
    entropy_coef: float,
    norm_adv: bool,
) -> tuple[jax.Array, PpoMetrics]:
    """PPO clipped objective with optional per-minibatch advantage normalization."""
    pi = ac.eval_pi(batch.obs, batch.act, ac_state)
    entropy = jnp.mean(pi.entropy)

    adv = batch.adv
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(pi.logp - batch.log_mu)
    pi_loss = rlax.clipped_surrogate_pg_loss(ratio, adv, clip_eps)

    values = ac.value(batch.obs, ac_state)
    v_loss = jnp.mean((values - batch.ret) ** 2)

    loss = pi_loss + vf_coef * v_loss - entropy_coef * entropy
    return loss, PpoMetrics(
        pi_loss=pi_loss,
        v_loss=v_loss,
        entropy=entropy,
        clip_frac=jnp.mean(jnp.abs(ratio - 1.0) > clip_eps),
        approx_kl=jnp.mean((ratio - 1.0) - jnp.log(ratio)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training state and step
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class State:
    """Dynamic state of the complete PPO recipe."""

    key: jax.Array
    rollout: Imc.State
    rew_norm: RewardNormalizer.State
    pi_opt: optax.OptState
    v_opt: optax.OptState


def train_step(
    state: State,
    *,
    rollout: Roll,
    rew_norm: RewardNormalizer,
    pi_tx: optax.GradientTransformation,
    v_tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[PpoMetrics, State]:
    """One PPO iteration: collect rollout, compute GAE, run minibatch updates."""
    ac = rollout.imc.agent
    trajectory, roll_state = rollout.sample(state.rollout)

    done = jnp.logical_or(trajectory.mc.term, trajectory.mc.trun)
    rewards, rew_norm_state = rew_norm.norm(
        trajectory.mc.rew,
        done,
        state.rew_norm,
    )

    values = trajectory.dec.value
    discounts = cfg.gamma * (1.0 - trajectory.mc.term.astype(jnp.float32))
    trace = cfg.gae_lambda * (1.0 - done.astype(jnp.float32))
    chex.assert_equal_shape([rewards, values, discounts, trace, trajectory.succ.value])
    advantages = jax.vmap(rlax.td_lambda)(
        values,
        rewards,
        discounts,
        trajectory.succ.value,
        trace,
    )
    returns = advantages + values

    batch = jax.tree.map(
        lambda x: x.reshape((-1, *x.shape[2:])),
        PpoBatch(
            obs=trajectory.mc.obs,
            act=trajectory.dec.act,
            log_mu=trajectory.dec.log_mu,
            adv=advantages,
            ret=returns,
        ),
    )
    state = state.replace(rollout=roll_state, rew_norm=rew_norm_state)
    metrics, state = update(
        batch,
        state,
        ac=ac,
        pi_tx=pi_tx,
        v_tx=v_tx,
        cfg=cfg,
    )
    agent = ac.update_obs(trajectory.mc.obs, state.rollout.agent)
    rollout_state = rollout.imc.refresh(state.rollout.replace(agent=agent))
    state = state.replace(rollout=rollout_state)
    return metrics, state


def update(
    batch: PpoBatch,
    state: State,
    *,
    ac: ActorCritic,
    pi_tx: optax.GradientTransformation,
    v_tx: optax.GradientTransformation,
    cfg: Config,
) -> tuple[PpoMetrics, State]:
    """Optimize the actor-critic over shuffled epochs and minibatches."""
    batch_size = batch.obs.shape[0]
    if batch_size % cfg.n_minibatches:
        raise ValueError("batch size must be divisible by n_minibatches")
    mb_size = batch_size // cfg.n_minibatches

    def minibatch(
        state: State,
        minibatch: PpoBatch,
    ) -> tuple[State, PpoMetrics]:
        ac_state = state.rollout.agent
        (_, metrics), grads = eqx.filter_value_and_grad(
            ppo_loss,
            has_aux=True,
        )(
            ac_state,
            ac=ac,
            batch=minibatch,
            clip_eps=cfg.clip_eps,
            vf_coef=cfg.vf_coef,
            entropy_coef=cfg.entropy_coef,
            norm_adv=cfg.normalize_adv,
        )
        pi_updates, pi_opt = pi_tx.update(
            grads.policy,
            state.pi_opt,
            ac_state.policy,
        )
        v_updates, v_opt = v_tx.update(
            grads.value,
            state.v_opt,
            ac_state.value,
        )
        ac_state = ac_state.replace(
            policy=eqx.apply_updates(ac_state.policy, pi_updates),
            value=eqx.apply_updates(ac_state.value, v_updates),
        )
        return state.replace(
            rollout=state.rollout.replace(agent=ac_state),
            pi_opt=pi_opt,
            v_opt=v_opt,
        ), metrics

    def epoch(
        state: State,
        key: jax.Array,
    ) -> tuple[State, PpoMetrics]:
        permutation = jrd.permutation(key, batch_size)
        minibatches = jax.tree.map(
            lambda x: x[permutation].reshape(
                cfg.n_minibatches,
                mb_size,
                *x.shape[1:],
            ),
            batch,
        )
        return jax.lax.scan(minibatch, state, minibatches)

    key, epoch_key = jrd.split(state.key)
    state = state.replace(key=key)
    state, metrics = jax.lax.scan(
        epoch,
        state,
        jrd.split(epoch_key, cfg.n_epochs),
    )
    return jax.tree.map(jnp.mean, metrics), state


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
    mc_key, env_key, ac_key, update_key, eval_key = jrd.split(key, 5)

    env = _make_env(cfg, cfg.n_envs)
    eval_env = _make_env(cfg, cfg.eval_envs)
    (obs_dim,) = env.obs_shape
    (act_dim,) = env.act_shape

    total_updates = cfg.n_iters * cfg.n_epochs * cfg.n_minibatches
    lr_schedule = optax.linear_schedule(cfg.lr, 0.0, total_updates)
    pi_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )
    v_tx = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )

    rew_norm: RewardNormalizer = (
        RunningRewNorm(gamma=cfg.gamma, rms=RunningStats(), clip=10.0)
        if cfg.normalize_reward
        else IdentityRewNorm()
    )

    rollout = Roll(
        seqlen=cfg.seqlen,
        seq_axis=1,
        imc=Imc(
            agent=ActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                actor_hiddens=cfg.actor_hiddens,
                critic_hiddens=cfg.critic_hiddens,
                layer_norm=cfg.layer_norm,
                obs_norm=(
                    RunningObsNorm(clip=10.0)
                    if cfg.normalize_obs
                    else IdentityObsNorm()
                ),
                deterministic=False,
            ),
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
            agent=ActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                actor_hiddens=cfg.actor_hiddens,
                critic_hiddens=cfg.critic_hiddens,
                layer_norm=cfg.layer_norm,
                obs_norm=(
                    RunningObsNorm(clip=10.0)
                    if cfg.normalize_obs
                    else IdentityObsNorm()
                ),
                deterministic=True,
            ),
            mc=VecMc(
                mc=Mc(
                    max_episode_len=cfg.max_episode_len,
                    queue_size=20,
                    env=eval_env,
                )
            ),
        ),
    )

    ac = rollout.imc.agent
    ac_state = ac.init(ac_key)
    rollout_state = rollout.imc.init(
        rollout.imc.mc.init(
            jrd.split(mc_key, cfg.n_envs),
            env.init(env_key),
        ),
        ac_state,
    )
    state = State(
        key=update_key,
        rollout=rollout_state,
        rew_norm=rew_norm.init(cfg.n_envs),
        pi_opt=pi_tx.init(rollout_state.agent.policy),
        v_opt=v_tx.init(rollout_state.agent.value),
    )

    @jax.jit
    def step(state):
        return train_step(
            state,
            rollout=rollout,
            rew_norm=rew_norm,
            pi_tx=pi_tx,
            v_tx=v_tx,
            cfg=cfg,
        )

    @jax.jit
    def evaluate(imc_state):
        return evaluator.evaluate(imc_state)

    print(f"[bold green]{cfg.env_id} PPO[/bold green]")
    t0 = time.time()

    for i in track(range(cfg.n_iters), description="Training"):
        metrics, state = step(state)

        if (i + 1) % cfg.eval_freq == 0:
            eval_key, e_env_key, k = jrd.split(eval_key, 3)
            eval_mc_state = evaluator.imc.mc.init(
                jrd.split(k, cfg.eval_envs), eval_env.init(e_env_key)
            )
            eval_metrics, eval_state = evaluate(
                evaluator.imc.init(
                    eval_mc_state,
                    state.rollout.agent.replace(key=eval_key),
                )
            )
            steps = (i + 1) * cfg.n_envs * cfg.seqlen
            print(
                f"  iter {i + 1:4d}  loss={float(metrics.pi_loss):+.4f}"
                f"  rew={float(eval_metrics.avg_eps_rew):.1f}"
                f"\u00b1{float(eval_metrics.std_eps_rew):.1f}"
                f"  len={float(eval_metrics.avg_eps_len):.1f}"
                f"  steps={steps:,}"
            )
            eval_env.close(eval_state.mc.env)

    elapsed = time.time() - t0
    print(f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s")

    env.close(state.rollout.mc.env)
    return state


if __name__ == "__main__":
    train(tyro.cli(Config))
