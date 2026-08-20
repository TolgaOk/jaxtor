"""Normalized Advantage Functions for continuous-action Gymnasium environments.

Each iteration collects a fixed sequence, computes off-policy Q(λ) targets
once, and reuses the fixed batch for several optimization epochs. The
configuration uses neither a replay buffer nor a target network.

The `Quadratic` agent implements a diagonal normalized advantage function:

    Q(s, a) = V(s) - 1/2 sum_i P_i(s) (a_i - mu_i(s))^2
    mu(s) = tanh(mu_net(s))

Collection adds persistent exploration around the deterministic action:

    noise_t = (1 - theta) noise_{t-1} + sigma epsilon_t
    a_t = clip(mu(s_t) + noise_t, -1, 1)

Components::

    Composition
    │
    ├── agent: Quadratic
    │   ├── body: ObsNorm
    │   │   └── RunningStats
    │   ├── value: VHead
    │   │   └── Equinox MLP
    │   ├── loc: tanh-bounded Equinox MLP
    │   └── p: Equinox MLP
    ├── behavior: Ou
    │   └── agent
    ├── mc: VecMc
    │   └── Mc
    │       └── Gymnasium or EnvPool
    ├── roll: Roll
    │   └── Imc
    │       ├── behavior
    │       └── mc
    ├── inference: QNextVInference
    │   └── agent
    ├── rew_norm: RewardNorm
    │   └── RunningStats
    ├── stats: EpisodeStats
    ├── batches: Minibatches
    └── McEval
        └── Imc
            ├── deterministic agent
            └── mc

State::

    TrainState
    ├── roll: imc state (behavior + mc)
    ├── opt: optimizer state
    ├── stats: episode statistics
    └── rew_norm: reward-normalization state

Flow::

    Main loop ↻
    ├→ collect sequence
    │   ├→ update stats
    │   ├→ normalize observations and rewards
    │   └→ freeze Q(λ) targets
    ├→ epochs ↻
    │   ├→ shuffle minibatches
    │   └→ minibatches ↻
    │       └→ Q loss → gradients → update
    └→ periodically
        ├→ report training metrics
        └→ evaluate deterministic agent
"""

from __future__ import annotations

from dataclasses import dataclass as static_dataclass
from dataclasses import replace
from functools import partial
from typing import Any, Literal

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import optax
import rlax
import tyro
from chex import dataclass

from jaxtor.agent import (
    Function,
    Module,
    Ou,
    Quadratic,
    QNextVInference,
    VHead,
    combine,
    partition,
)
from jaxtor.env import gymnasium
from jaxtor.eval import McEval
from jaxtor.sampler import EpisodeStats, Imc, Mc, Roll, VecMc
from jaxtor.util import Minibatches, ObsNorm, RewardNorm, RunningStats


@static_dataclass(frozen=True)
class Config:
    """Command-line NAF configuration."""

    env_id: str = "Hopper-v5"
    env_backend: Literal["gymnasium", "envpool"] = "gymnasium"
    async_envs: bool = True
    n_iters: int = 500
    n_envs: int = 16
    n_eval_envs: int = 10
    seq_len: int = 2048
    max_eps_len: int = 1000
    v_width: int = 128
    v_depth: int = 2
    mu_width: int = 64
    mu_depth: int = 2
    p_width: int = 128
    p_depth: int = 2
    lr: float = 3e-4
    lr_schedule: Literal["constant", "linear"] = "linear"
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    trace_lambda: float = 0.9
    p_eps: float = 1.0
    noise_scale: float = 0.1
    noise_theta: float = 0.15
    n_epochs: int = 5
    n_batches: int = 32
    norm_obs: bool = True
    norm_rew: bool = True
    eval_freq: int = 5
    seed: int = 0


def module(fn: object) -> tuple[Module[jax.Array], Module.State[jax.Array]]:
    """Split an initialized Equinox module into component and state."""
    if not isinstance(fn, Function):
        raise TypeError("an Equinox module must be callable")
    params, static = eqx.partition(fn, eqx.is_array)
    if not isinstance(params, Function) or not isinstance(static, Function):
        raise TypeError("both Equinox partitions must remain callable")
    component: Module[jax.Array] = Module(static=static)
    return component, component.init(params)


@dataclass
class Batch:
    """Fixed observations, actions, and Q(λ) targets."""

    obs: jax.Array
    act: jax.Array
    ret: chex.Array


@dataclass
class Metrics:
    """Transient metrics from one NAF minibatch update."""

    loss: chex.Numeric
    q_mean: chex.Numeric
    ret_mean: chex.Numeric


type NetState = Module.State[jax.Array]
type AgentState = Quadratic.State[
    ObsNorm.State, VHead.State[NetState], NetState, NetState
]
type BehaviorState = Ou.State[AgentState]
type RollState = Imc.State[Mc.State[Any], BehaviorState]


@dataclass
class TrainState:
    """Dynamic state threaded through complete NAF iterations."""

    key: jax.Array
    roll: RollState
    opt: optax.OptState
    stats: EpisodeStats.State
    rew_norm: RewardNorm.State[RunningStats.State]


@dataclass
class LearnState:
    """Parameters and optimizer state carried through NAF learning loops."""

    params: AgentState
    opt: optax.OptState


def make_env(config: Config):
    """Configure one continuous-control environment backend."""
    match config.env_backend:
        case "gymnasium":
            return gymnasium.make(config.env_id, async_envs=config.async_envs)
        case "envpool":
            from jaxtor.env import envpool

            return envpool.make(
                config.env_id,
                max_episode_steps=config.max_eps_len,
            )


cfg = tyro.cli(Config) if __name__ == "__main__" else Config()

(
    agent_key,
    env_key,
    mc_key,
    update_key,
    eval_key,
) = jrd.split(jrd.key(cfg.seed), 5)


env = make_env(cfg)
eval_env = make_env(cfg)
if len(env.obs_shape) != 1 or len(env.act_shape) != 1:
    raise ValueError("NAF requires vector observations and continuous vector actions")
obs_size = env.obs_shape[0]
act_size = env.act_shape[0]


behavior_key, v_key, mu_key, p_key = jrd.split(agent_key, 4)
v, v_state = module(
    eqx.nn.MLP(
        in_size=obs_size,
        out_size=1,
        width_size=cfg.v_width,
        depth=cfg.v_depth,
        activation=jax.nn.tanh,
        key=v_key,
    )
)
mu, mu_state = module(
    eqx.nn.MLP(
        in_size=obs_size,
        out_size=act_size,
        width_size=cfg.mu_width,
        depth=cfg.mu_depth,
        activation=jax.nn.tanh,
        final_activation=jax.nn.tanh,
        key=mu_key,
    )
)
p, p_state = module(
    eqx.nn.MLP(
        in_size=obs_size,
        out_size=act_size,
        width_size=cfg.p_width,
        depth=cfg.p_depth,
        activation=jax.nn.tanh,
        key=p_key,
    )
)
v_head = VHead(net=v)


norm = ObsNorm(
    stats=RunningStats(clip=10.0),
    enabled=cfg.norm_obs,
)
agent = Quadratic(
    act_size=act_size,
    body=norm,
    value=v_head,
    loc=mu,
    p=p,
    eps=cfg.p_eps,
)
agent_state: AgentState = agent.init(
    body=norm.init((obs_size,)),
    v=v_head.init(v_state),
    loc=mu_state,
    p=p_state,
)


mc = VecMc(mc=Mc(max_eps_len=cfg.max_eps_len, env=env))
behavior = Ou(
    agent=agent,
    theta=cfg.noise_theta,
    sigma=cfg.noise_scale,
)
imc = Imc(agent=behavior, mc=mc)
roll = Roll(imc=imc, seq_len=cfg.seq_len, seq_axis=1)
inference = QNextVInference(agent=agent, seq_axis=1)


rew_norm = RewardNorm(
    gamma=cfg.gamma,
    stats=RunningStats(),
    seq_axis=1,
    clip=10.0,
    enabled=cfg.norm_rew,
)
stats = EpisodeStats(seq_axis=1)
batches = Minibatches(count=cfg.n_batches, sample_ndim=2)
total_updates = cfg.n_iters * cfg.n_epochs * cfg.n_batches
learning_rate = (
    optax.linear_schedule(cfg.lr, 0.0, total_updates)
    if cfg.lr_schedule == "linear"
    else cfg.lr
)
optim: Any = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(learning_rate, eps=1e-5),
)


eval_mc = VecMc(mc=Mc(max_eps_len=cfg.max_eps_len, env=eval_env))
eval_imc = Imc(agent=agent, mc=eval_mc)
evaluator = McEval(imc=eval_imc, n_step=cfg.max_eps_len)
evaluate = jax.jit(evaluator.evaluate)
q_lambda = partial(
    rlax.general_off_policy_returns_from_q_and_v,
    stop_target_gradients=True,
)


@jax.jit
def update(state: TrainState) -> tuple[Metrics, TrainState]:
    """Collect one sequence, freeze Q(λ) targets, and optimize NAF."""
    seq, roll_state = roll.sample(state.roll)
    stats_state = stats.update(seq, state.stats)
    done = seq.term | seq.trun
    rew, rew_norm_state = rew_norm.update(seq.rew, done, state.rew_norm)
    seq = replace(seq, rew=rew)
    agent_state = replace(
        roll_state.agent.agent,
        body=norm.update(seq.obs, roll_state.agent.agent.body),
    )
    infer = inference.apply(seq, agent_state)

    ret = jax.vmap(q_lambda)(
        infer.q_t,
        infer.v_t,
        seq.rew,
        cfg.gamma * (~seq.term).astype(seq.rew.dtype),
        cfg.trace_lambda * (~done[:, :-1]).astype(seq.rew.dtype),
    )
    batch = Batch(obs=seq.obs, act=seq.act, ret=ret)
    parts = partition(agent_state)

    def loss(params: AgentState, batch: Batch) -> tuple[chex.Numeric, Metrics]:
        q, _ = agent.q(batch.obs, batch.act, combine(params, parts.frozen))
        q_loss = jnp.mean((q - batch.ret) ** 2)
        return q_loss, Metrics(
            loss=q_loss,
            q_mean=jnp.mean(q),
            ret_mean=jnp.mean(batch.ret),
        )

    def epoch(
        carry: LearnState,
        key: jax.Array,
    ) -> tuple[LearnState, Metrics]:
        def minibatch(
            carry: LearnState,
            batch: Batch,
        ) -> tuple[LearnState, Metrics]:
            (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
                carry.params,
                batch,
            )
            updates, opt = optim.update(grads, carry.opt, carry.params)
            return LearnState(
                params=eqx.apply_updates(carry.params, updates),
                opt=opt,
            ), metrics

        return jax.lax.scan(minibatch, carry, batches.shuffle(key, batch))

    key, epoch_key = jrd.split(state.key)
    learn, metrics = jax.lax.scan(
        epoch,
        LearnState(params=parts.params, opt=state.opt),
        jrd.split(epoch_key, cfg.n_epochs),
    )
    agent_state = combine(learn.params, parts.frozen)
    return jax.tree.map(jnp.mean, metrics), replace(
        state,
        key=key,
        roll=replace(
            roll_state,
            agent=replace(roll_state.agent, agent=agent_state),
        ),
        opt=learn.opt,
        stats=stats_state,
        rew_norm=rew_norm_state,
    )


def train() -> TrainState:
    """Initialize dynamic state and train the configured NAF recipe."""
    state = TrainState(
        key=update_key,
        roll=imc.init(
            mc.init(
                jrd.split(mc_key, cfg.n_envs),
                jax.vmap(env.init)(jrd.split(env_key, cfg.n_envs)),
            ),
            behavior.init(
                behavior_key,
                jnp.zeros((cfg.n_envs, act_size)),
                agent_state,
            ),
        ),
        opt=optim.init(partition(agent_state).params),
        stats=stats.init((cfg.n_envs,)),
        rew_norm=rew_norm.init((cfg.n_envs,)),
    )
    eval_rng = eval_key

    for iteration in range(1, cfg.n_iters + 1):
        metrics, state = update(state)
        if iteration % cfg.eval_freq == 0:
            train_metrics, stats_state = stats.drain(state.stats)
            state = replace(state, stats=stats_state)

            eval_rng, env_rng, mc_rng = jrd.split(eval_rng, 3)
            eval_metrics, eval_state = evaluate(
                eval_imc.init(
                    eval_mc.init(
                        jrd.split(mc_rng, cfg.n_eval_envs),
                        jax.vmap(eval_env.init)(
                            jrd.split(env_rng, cfg.n_eval_envs),
                        ),
                    ),
                    state.roll.agent.agent,
                )
            )
            print(
                f"iter={iteration:4d}  loss={float(metrics.loss):.3f}"
                f"  q={float(metrics.q_mean):.1f}"
                f"  ret={float(metrics.ret_mean):.1f}"
                f"  train={float(train_metrics.avg_eps_rew):.1f}"
                f"  eval={float(eval_metrics.avg_eps_rew):.1f}"
            )
            eval_env.close(eval_state.mc.env)

    env.close(state.roll.mc.env)
    return state


if __name__ == "__main__":
    train()
