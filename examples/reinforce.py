"""REINFORCE for discrete-action Gymnax environments with Jaxtor components.

CartPole-v1 is the default. ``--env-id`` may select any Gymnax environment
with one-dimensional observations and a discrete action space. Fixed-length
sequences may contain several episodes; episode boundaries stop return
propagation, while an unfinished trailing fragment uses a zero bootstrap.

Components::

    Composition
    │
    ├── agent: Pi
    │   ├── body: Module(Linear → tanh)
    │   └── pi: CategoricalHead
    │       └── Module(Linear logits)
    ├── mc: VecMc
    │   └── Mc
    │       └── GymnaxEnv
    ├── roll: Roll
    │   └── Imc
    │       ├── agent
    │       └── mc
    ├── stats: EpisodeStats
    └── Eval
        └── Imc
            ├── deterministic agent
            └── mc

State::

    TrainState
    ├── roll: imc state (agent + mc)
    ├── opt: optimizer state
    └── stats: episode statistics

Flow::

    Main loop ↻
    ├→ collect sequence
    │   ├→ update stats
    │   └→ infer policy
    ├→ discounted returns
    ├→ policy loss → gradients → update
    └→ periodically
        ├→ report training metrics
        └→ evaluate deterministic agent
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
    Pi,
    combine,
    partition,
)
from jaxtor.env import gymnax
from jaxtor.eval.mc import Eval
from jaxtor.sampler import EpisodeStats, Imc, Mc, Roll, VecMc


@static_dataclass(frozen=True)
class Config:
    """Command-line REINFORCE configuration."""

    env_id: str = "CartPole-v1"
    n_iters: int = 200
    n_envs: int = 16
    n_eval_envs: int = 8
    seq_len: int = 200
    hidden_size: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_eps_len: int = 500
    eval_freq: int = 20
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
class Metrics:
    """Transient metrics from one policy update."""

    loss: chex.Numeric
    pi_loss: chex.Numeric
    entropy: chex.Numeric


type ModuleState = Module.State[jax.Array]
type PiState = CategoricalHead.State[ModuleState]
type AgentState = Pi.State[ModuleState, PiState]
type RollState = Imc.State[Mc.State[gymnax.GymnaxEnv.State], AgentState]


@dataclass
class TrainState:
    """Dynamic state threaded through complete REINFORCE iterations."""

    roll: RollState
    opt: optax.OptState
    stats: EpisodeStats.State


cfg = tyro.cli(Config) if __name__ == "__main__" else Config()

(
    agent_key,
    env_key,
    mc_key,
    eval_env_key,
    eval_mc_key,
) = jrd.split(jrd.key(cfg.seed), 5)


env = gymnax.make(cfg.env_id)
obs_shape = tuple(env.env.observation_space(env.params).shape)
if len(obs_shape) != 1:
    raise ValueError("the REINFORCE agent requires vector observations")
act_size = getattr(env.env.action_space(env.params), "n", None)
if act_size is None:
    raise ValueError("the REINFORCE agent requires a discrete action space")


act_key, body_key, pi_key = jrd.split(agent_key, 3)
body_net, body_net_state = module(
    eqx.nn.Sequential(
        [
            eqx.nn.Linear(obs_shape[0], cfg.hidden_size, key=body_key),
            eqx.nn.Lambda(jnp.tanh),
        ]
    )
)
pi_net, pi_net_state = module(eqx.nn.Linear(cfg.hidden_size, int(act_size), key=pi_key))


pi = CategoricalHead(n_actions=int(act_size), logits=pi_net)
agent = Pi(body=body_net, pi=pi)
agent_state = agent.init(
    act_key,
    body=body_net_state,
    pi=pi.init(pi_net_state),
)


mc = VecMc(mc=Mc(max_eps_len=cfg.max_eps_len, env=env))
imc = Imc(agent=agent, mc=mc)
roll = Roll(imc=imc, seq_len=cfg.seq_len, seq_axis=1)


stats = EpisodeStats(seq_axis=1)
optim: Any = optax.chain(
    optax.clip_by_global_norm(cfg.max_grad_norm),
    optax.adam(cfg.lr),
)


eval_imc = Imc(agent=replace(agent, deterministic=True), mc=mc)
evaluator = Eval(imc=eval_imc, episode_len=cfg.max_eps_len)
evaluate = jax.jit(evaluator.evaluate)


@jax.jit
def update(state: TrainState) -> tuple[Metrics, TrainState]:
    """Collect one sequence and apply one REINFORCE update."""
    seq, roll_state = roll.sample(state.roll)
    stats_state = stats.update(seq, state.stats)
    ret = jax.vmap(rlax.discounted_returns, in_axes=(0, 0, None))(
        seq.rew,
        cfg.gamma * (~(seq.term | seq.trun)).astype(seq.rew.dtype),
        jnp.zeros((), dtype=seq.rew.dtype),
    )
    parts = partition(roll_state.agent)

    def loss(params: AgentState) -> tuple[chex.Numeric, Metrics]:
        pred, _ = agent.apply(seq.obs, combine(params, parts.frozen))
        policy = pred.pi.evaluate(seq.act)
        pi_loss = -jnp.mean(policy.logp * jax.lax.stop_gradient(ret))
        entropy = jnp.mean(policy.entropy)
        total = pi_loss - cfg.ent_coef * entropy
        return total, Metrics(loss=total, pi_loss=pi_loss, entropy=entropy)

    (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(parts.params)
    updates, opt = optim.update(grads, state.opt, parts.params)
    agent_state: AgentState = combine(
        eqx.apply_updates(parts.params, updates),
        parts.frozen,
    )
    return metrics, replace(
        state,
        roll=replace(roll_state, agent=agent_state),
        opt=opt,
        stats=stats_state,
    )


def train() -> TrainState:
    """Initialize dynamic state and train the configured REINFORCE recipe."""
    state = TrainState(
        roll=imc.init(
            mc.init(
                jrd.split(mc_key, cfg.n_envs),
                jax.vmap(env.init)(jrd.split(env_key, cfg.n_envs)),
            ),
            agent_state,
        ),
        opt=optim.init(partition(agent_state).params),
        stats=stats.init((cfg.n_envs,)),
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
