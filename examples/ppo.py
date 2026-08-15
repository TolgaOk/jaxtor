"""PPO for discrete-action Gymnax environments with explicit Jaxtor components.

CartPole-v1 is the default. ``--env-id`` may select any Gymnax environment
with one-dimensional observations and a discrete action space. Other tasks may
need different hyperparameters or agent components.

Components::

    Composition
    │
    ├── agent: VPi actor–critic
    │   ├── body: NormModel
    │   │   ├── norm: ObsNorm
    │   │   │   └── RunningStats
    │   │   └── model: Module(Linear → tanh)
    │   ├── v: VHead
    │   │   └── Module(Linear value)
    │   └── pi: CategoricalHead
    │       └── Module(Linear logits)
    ├── mc: VecMc
    │   └── Mc
    │       └── GymnaxEnv
    ├── roll: Roll
    │   └── Imc
    │       ├── agent
    │       └── mc
    ├── inference: VPiNextVInference
    │   └── agent
    ├── rew_norm: RewardNorm
    │   └── RunningStats
    ├── stats: EpisodeStats
    ├── batches: Minibatches
    └── Eval
        └── Imc
            ├── deterministic agent
            └── mc

State::

    TrainState
    ├── roll: imc state (agent + mc)
    ├── opt: optimizer state
    ├── stats: episode statistics
    └── rew_norm: reward-normalization state

Flow::

    Main loop ↻
    ├→ collect sequence
    │   ├→ update stats
    │   ├→ normalize rewards
    │   └→ infer policy + values
    ├→ TD(λ) advantages + returns
    ├→ form batch
    ├→ epochs ↻
    │   ├→ shuffle minibatches
    │   └→ minibatches ↻
    │       └→ loss → gradients → update
    ├→ update obs stats
    └→ periodically
        ├→ report training metrics
        └→ evaluate deterministic agent

The optimization loops and RLax target remain visible because they define PPO.
Replaying the fixed agent trades one batched forward pass for an action-only
sampling interface.
"""

from __future__ import annotations

from dataclasses import dataclass as static_dataclass
from dataclasses import replace
from typing import Any

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
    CategoricalHead,
    Function,
    Module,
    NormModel,
    VHead,
    VPi,
    VPiNextVInference,
    combine,
    partition,
)
from jaxtor.env import gymnax
from jaxtor.eval.mc import Eval
from jaxtor.sampler import EpisodeStats, Imc, Mc, Roll, VecMc
from jaxtor.util import Minibatches, ObsNorm, RewardNorm, RunningStats


@static_dataclass(frozen=True)
class Config:
    """Command-line PPO configuration."""

    env_id: str = "CartPole-v1"
    n_iters: int = 100
    n_envs: int = 16
    n_eval_envs: int = 8
    seq_len: int = 128
    n_epochs: int = 4
    n_batches: int = 4
    hidden_size: int = 64
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_eps_len: int = 500
    eval_freq: int = 10
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
    """Raw observations and fixed PPO training targets."""

    obs: jax.Array
    act: jax.Array
    log_mu: jax.Array
    adv: chex.Array
    ret: chex.Array


@dataclass
class Metrics:
    """Transient metrics from one PPO minibatch update."""

    loss: chex.Numeric
    pi_loss: chex.Numeric
    v_loss: chex.Numeric
    entropy: chex.Numeric


type ModuleState = Module.State[jax.Array]
type BodyState = NormModel.State[ObsNorm.State, ModuleState]
type ValueState = VHead.State[ModuleState]
type PiState = CategoricalHead.State[ModuleState]
type AgentState = VPi.State[BodyState, ValueState, PiState]
type RollState = Imc.State[Mc.State[gymnax.GymnaxEnv.State], AgentState]


@dataclass
class TrainState:
    """Dynamic state threaded through complete PPO iterations."""

    key: jax.Array
    roll: RollState
    opt: optax.OptState
    stats: EpisodeStats.State
    rew_norm: RewardNorm.State[RunningStats.State]


@dataclass
class LearnState:
    """Parameters and optimizer state carried through PPO learning loops."""

    params: AgentState
    opt: optax.OptState


cfg = tyro.cli(Config) if __name__ == "__main__" else Config()

(
    agent_key,
    env_key,
    mc_key,
    update_key,
    eval_env_key,
    eval_mc_key,
) = jrd.split(jrd.key(cfg.seed), 6)


env = gymnax.make(cfg.env_id)
obs_shape = tuple(env.env.observation_space(env.params).shape)
if len(obs_shape) != 1:
    raise ValueError("the lean agent requires vector observations")
act_size = getattr(env.env.action_space(env.params), "n", None)
if act_size is None:
    raise ValueError("the lean agent requires a discrete action space")


act_key, body_key, v_key, pi_key = jrd.split(agent_key, 4)
body_net, body_net_state = module(
    eqx.nn.Sequential(
        [
            eqx.nn.Linear(obs_shape[0], cfg.hidden_size, key=body_key),
            eqx.nn.Lambda(jnp.tanh),
        ]
    )
)
v_net, v_net_state = module(eqx.nn.Linear(cfg.hidden_size, 1, key=v_key))
pi_net, pi_net_state = module(eqx.nn.Linear(cfg.hidden_size, int(act_size), key=pi_key))


norm = ObsNorm(stats=RunningStats(clip=10.0))
body = NormModel(norm=norm, model=body_net)
v = VHead(net=v_net)
pi = CategoricalHead(n_actions=int(act_size), logits=pi_net)
agent = VPi(body=body, v=v, pi=pi)


agent_state = agent.init(
    act_key,
    body=body.init(norm.init(obs_shape), body_net_state),
    v=v.init(v_net_state),
    pi=pi.init(pi_net_state),
)


mc = VecMc(mc=Mc(max_eps_len=cfg.max_eps_len, env=env))
imc = Imc(agent=agent, mc=mc)
roll = Roll(imc=imc, seq_len=cfg.seq_len, seq_axis=1)
inference = VPiNextVInference(agent=agent, seq_axis=1)


rew_norm = RewardNorm(
    gamma=cfg.gamma,
    rms=RunningStats(),
    seq_axis=1,
    clip=10.0,
)
stats = EpisodeStats(seq_axis=1)
batches = Minibatches(count=cfg.n_batches, sample_ndim=2)
optim: Any = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(cfg.lr, eps=1e-5),
)


eval_imc = Imc(agent=replace(agent, deterministic=True), mc=mc)
evaluator = Eval(imc=eval_imc, episode_len=cfg.max_eps_len)
evaluate = jax.jit(evaluator.evaluate)


@jax.jit
def update(state: TrainState) -> tuple[Metrics, TrainState]:
    """Collect one sequence, form targets, and optimize the agent."""
    seq, roll_state = roll.sample(state.roll)
    stats_state = stats.update(seq, state.stats)
    done = seq.term | seq.trun
    rew, rew_norm_state = rew_norm.update(
        seq.rew,
        done,
        state.rew_norm,
    )
    seq = replace(seq, rew=rew)
    infer = inference.apply(seq, roll_state.agent)
    chex.assert_equal_shape([infer.v_tm1, seq.rew, infer.v_t])
    adv = jax.vmap(rlax.td_lambda)(
        infer.v_tm1,
        seq.rew,
        cfg.gamma * (~seq.term).astype(seq.rew.dtype),
        infer.v_t,
        cfg.gae_lambda * (~done).astype(seq.rew.dtype),
    )
    batch = Batch(
        obs=seq.obs,
        act=seq.act,
        log_mu=infer.pi_tm1.evaluate(seq.act).logp,
        adv=(adv - adv.mean()) / (adv.std() + 1e-8),
        ret=adv + infer.v_tm1,
    )
    parts = partition(roll_state.agent)

    def loss(params: AgentState, batch: Batch) -> tuple[chex.Numeric, Metrics]:
        pred, _ = agent.apply(
            batch.obs,
            combine(params, parts.frozen),
        )
        policy = pred.pi.evaluate(batch.act)
        pi_loss = rlax.clipped_surrogate_pg_loss(
            jnp.exp(policy.logp - batch.log_mu),
            batch.adv,
            cfg.clip_eps,
        )
        v_loss = 0.5 * jnp.mean((pred.v - batch.ret) ** 2)
        entropy = jnp.mean(policy.entropy)
        total = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
        return total, Metrics(
            loss=total,
            pi_loss=pi_loss,
            v_loss=v_loss,
            entropy=entropy,
        )

    def minibatch(carry: LearnState, batch: Batch) -> tuple[LearnState, Metrics]:
        (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
            carry.params,
            batch,
        )
        updates, opt = optim.update(grads, carry.opt, carry.params)
        return LearnState(
            params=eqx.apply_updates(carry.params, updates),
            opt=opt,
        ), metrics

    def epoch(carry: LearnState, key: jax.Array) -> tuple[LearnState, Metrics]:
        return jax.lax.scan(
            minibatch,
            carry,
            batches.shuffle(key, batch),
        )

    key, epoch_key = jrd.split(state.key)
    carry, metrics = jax.lax.scan(
        epoch,
        LearnState(params=parts.params, opt=state.opt),
        jrd.split(epoch_key, cfg.n_epochs),
    )
    agent_state: AgentState = combine(carry.params, parts.frozen)
    agent_state = replace(
        agent_state,
        body=body.update(seq.obs, agent_state.body),
    )
    state = replace(
        state,
        key=key,
        roll=replace(roll_state, agent=agent_state),
        opt=carry.opt,
        stats=stats_state,
        rew_norm=rew_norm_state,
    )
    return jax.tree.map(jnp.mean, metrics), state


def train() -> TrainState:
    """Initialize dynamic state and train the configured PPO recipe."""
    state = TrainState(
        key=update_key,
        roll=imc.init(
            mc.init(
                jrd.split(mc_key, cfg.n_envs),
                jax.vmap(env.init)(jrd.split(env_key, cfg.n_envs)),
            ),
            agent_state,
        ),
        opt=optim.init(partition(agent_state).params),
        stats=stats.init((cfg.n_envs,)),
        rew_norm=rew_norm.init((cfg.n_envs,)),
    )
    eval_state = eval_imc.init(
        mc.init(
            jrd.split(eval_mc_key, cfg.n_eval_envs),
            jax.vmap(env.init)(jrd.split(eval_env_key, cfg.n_eval_envs)),
        ),
        agent_state,
    )

    for iteration in range(1, cfg.n_iters + 1):
        metrics, state = update(state)
        if iteration % cfg.eval_freq == 0:
            train_metrics, stats_state = stats.drain(state.stats)
            state = replace(state, stats=stats_state)

            eval_metrics, _ = evaluate(
                replace(eval_state, agent=state.roll.agent),
            )
            print(
                f"iter={iteration:3d}  loss={float(metrics.loss):+.3f}"
                f"  train={float(train_metrics.avg_eps_rew):.1f}"
                f"  eval={float(eval_metrics.avg_eps_rew):.1f}"
            )
    return state


if __name__ == "__main__":
    train()
