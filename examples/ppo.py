"""PPO for discrete-action Gymnax environments with explicit Jaxtor components.

CartPole-v1 is the default. Other tasks with vector observations and discrete
actions may need different hyperparameters. A shared normalized body feeds
categorical policy and value heads. Rollouts supply fixed targets for all PPO
epochs, then observation statistics are updated. Reward normalization retains
its own running state. Evaluation uses a deterministic policy in fixed-start
windows and reports completed episodes.
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
    combine,
    partition,
)
from jaxtor.env import gymnax
from jaxtor.eval import McEval
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
    """Raw observations and fixed PPO training targets.

    Attributes:
        obs: Observations with leading environment and time axes.
        act: Actions sampled by the behavior policy.
        log_mu: Behavior log probabilities of those actions.
        adv: Standardized advantages.
        ret: Unstandardized value targets.
    """

    obs: jax.Array
    act: jax.Array
    log_mu: jax.Array
    adv: chex.Array
    ret: chex.Array


@dataclass
class State:
    """Sampler, optimizer, episode statistics, and shuffle key.

    Attributes:
        sample: Complete interaction state, including the agent and environments.
        opt: Optimizer state for marked agent parameters.
        episodes: Episode statistics accumulated across collection windows.
        rew_norm: Running reward-normalization state.
        key: Key for the next epoch shuffle.
    """

    sample: Imc.State
    opt: optax.OptState
    episodes: EpisodeStats.State
    rew_norm: RewardNorm.State[RunningStats.State]
    key: jax.Array


def train(cfg: Config) -> State:
    """Construct a PPO recipe, train it, and return its final state."""
    agent_key, env_key, mc_key, update_key, eval_env_key, eval_mc_key = jrd.split(
        jrd.key(cfg.seed), 6
    )
    act_key, body_key, v_key, pi_key = jrd.split(agent_key, 4)

    env = gymnax.make(cfg.env_id)
    obs_shape = tuple(env.env.observation_space(env.params).shape)
    if len(obs_shape) != 1:
        raise ValueError("the PPO example requires vector observations")
    act_size = getattr(env.env.action_space(env.params), "n", None)
    if act_size is None:
        raise ValueError("the PPO example requires a discrete action space")

    body_net, body_net_state = module(
        eqx.nn.Sequential(
            [
                eqx.nn.Linear(obs_shape[0], cfg.hidden_size, key=body_key),
                eqx.nn.Lambda(jnp.tanh),
            ]
        )
    )
    v_net, v_net_state = module(eqx.nn.Linear(cfg.hidden_size, 1, key=v_key))
    pi_net, pi_net_state = module(
        eqx.nn.Linear(cfg.hidden_size, int(act_size), key=pi_key)
    )
    norm = ObsNorm(stats=RunningStats(clip=10.0))
    body = NormModel(norm=norm, model=body_net)
    v = VHead(net=v_net)
    pi = CategoricalHead(n_actions=int(act_size), logits=pi_net)
    agent = VPi(body=body, value=v, policy=pi)
    mc = VecMc(mc=Mc(max_eps_len=cfg.max_eps_len, env=env))
    imc = Imc(agent=agent, mc=mc)
    roll = Roll(imc=imc, seq_len=cfg.seq_len, seq_axis=1)
    rew_norm = RewardNorm(gamma=cfg.gamma, stats=RunningStats(), seq_axis=1, clip=10.0)
    episode_stats = EpisodeStats(seq_axis=1)
    batches = Minibatches(count=cfg.n_batches, sample_ndim=2)
    optim: Any = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr, eps=1e-5),
    )
    eval_imc = Imc(agent=replace(agent, deterministic=True), mc=mc)
    evaluator = McEval(imc=eval_imc, n_step=cfg.max_eps_len)
    evaluate = jax.jit(evaluator.evaluate)

    @jax.jit
    def update(state: State):
        """Collect with frozen observation statistics, learn, then commit."""
        seq, sample = roll.sample(state.sample)
        episodes = episode_stats.update(seq, state.episodes)
        rew, rew_norm_state = rew_norm.update(
            seq.rew, seq.term | seq.trun, state.rew_norm
        )
        behavior, _ = agent.vpi(seq.obs, sample.agent)
        nv, _ = agent.v(seq.nobs, sample.agent)
        adv = jax.vmap(rlax.td_lambda)(
            behavior.v,
            rew,
            cfg.gamma * (~seq.term).astype(rew.dtype),
            nv,
            cfg.gae_lambda * (~(seq.term | seq.trun)).astype(rew.dtype),
        )
        batch = Batch(
            obs=seq.obs,
            act=seq.act,
            log_mu=behavior.pi.evaluate(seq.act).logp,
            adv=(adv - adv.mean()) / (adv.std() + 1e-8),
            ret=adv + behavior.v,
        )
        parts = partition(sample.agent)

        def loss(params, b: Batch):
            vp, _ = agent.vpi(b.obs, combine(params, parts.frozen))
            prediction = vp.pi.evaluate(b.act)
            ratio = jnp.exp(prediction.logp - b.log_mu)
            clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            pg = -jnp.mean(jnp.minimum(ratio * b.adv, clipped * b.adv))
            vf = 0.5 * jnp.mean((vp.v - b.ret) ** 2)
            entropy = prediction.entropy.mean()
            total = pg + cfg.vf_coef * vf - cfg.ent_coef * entropy
            return total, dict(loss=total, pg=pg, vf=vf, entropy=entropy)

        def minibatch(carry, b: Batch):
            params, opt = carry
            (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, b)
            updates, opt = optim.update(grads, opt, params)
            return (optax.apply_updates(params, updates), opt), metrics

        def epoch(carry, key):
            return jax.lax.scan(minibatch, carry, batches.shuffle(key, batch))

        key, epoch_key = jrd.split(state.key)
        (params, opt), metrics = jax.lax.scan(
            epoch,
            (parts.params, state.opt),
            jrd.split(epoch_key, cfg.n_epochs),
        )
        learned = combine(params, parts.frozen)
        learned = replace(learned, body=body.update(seq.obs, learned.body))
        state = State(
            sample=replace(sample, agent=learned),
            opt=opt,
            episodes=episodes,
            rew_norm=rew_norm_state,
            key=key,
        )
        return jax.tree.map(jnp.mean, metrics), state

    agent_state = agent.init(
        act_key,
        body=body.init(norm.init(obs_shape), body_net_state),
        v=v.init(v_net_state),
        pi=pi.init(pi_net_state),
    )
    state = State(
        sample=imc.init(
            mc.init(
                jrd.split(mc_key, cfg.n_envs),
                jax.vmap(env.init)(jrd.split(env_key, cfg.n_envs)),
            ),
            agent_state,
        ),
        opt=optim.init(partition(agent_state).params),
        episodes=episode_stats.init((cfg.n_envs,)),
        rew_norm=rew_norm.init((cfg.n_envs,)),
        key=update_key,
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
            train_metrics, episodes = episode_stats.drain(state.episodes)
            state = replace(state, episodes=episodes)
            eval_metrics, _ = evaluate(replace(eval_state, agent=state.sample.agent))
            print(
                f"iter={iteration:3d}  loss={float(metrics['loss']):+.3f}"
                f"  train={float(train_metrics.avg_eps_rew):.1f}"
                f"  eval={float(eval_metrics.avg_eps_rew):.1f}"
            )
    return state


if __name__ == "__main__":
    train(tyro.cli(Config))
