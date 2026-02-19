"""Sampling-based evaluation.

Computes evaluation metrics from environment rollouts using episode statistics
tracked by the sampler.

Classes:
    Eval: Sampling-based evaluator with batched environment support.

Example:
    >>> imc = Imc(agent=agent, mc=mc)
    >>> evaluator = Eval(imc=imc, n_episodes=10, n_envs=4)
    >>> state = evaluator.init(key)
    >>> state, metrics = evaluator.metric(state, agent_state, env_state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from chex import dataclass


class Sampler(Protocol):
    max_episode_len: int

    class State(Protocol):
        last_obs: chex.Array
        last_done: chex.Numeric
        eps_rew_queue: chex.Array
        eps_len_queue: chex.Array

    def init(self, key: chex.PRNGKey, env: chex.ArrayTree) -> State: ...

    def sample(
        self, act: chex.Array, state: Sampler.State
    ) -> tuple[chex.ArrayTree, Sampler.State]: ...


class Agent(Protocol):
    class State(Protocol): ...

    def act(
        self,
        obs: chex.Array,
        state: Agent.State,
    ) -> tuple[chex.Array, Agent.State]: ...


class Imc(Protocol):
    agent: Agent
    mc: Sampler

    class State(Protocol):
        mc: Sampler.State
        agent: Agent.State

    class Transition(Protocol):
        term: chex.Array
        trun: chex.Array

    def sample(
        self, state: Imc.State
    ) -> tuple[Transition, Imc.State]: ...


@dataclass
class Eval:
    """Sampling-based evaluator.

    Rolls out the policy in the environment and aggregates episode statistics
    from the sampler's queues. Supports batched environments via vmap.

    Attributes:
        imc: Induced Markov chain following the Imc protocol.
        n_episodes: Number of episodes to collect per environment.
        n_envs: Number of parallel environments.
        _unroll: Loop unroll factor for jax.lax.scan.
    """

    imc: Imc
    n_episodes: int
    n_envs: int = 1
    _unroll: int = 1

    @dataclass
    class State:
        """State of the evaluator.

        Attributes:
            key: Random key for environment initialization.
            step: Current evaluation iteration count.
        """

        key: chex.PRNGKey
        step: chex.Numeric

    @dataclass
    class Metrics:
        """Episode statistics from evaluation rollouts.

        Attributes:
            avg_eps_rew: Mean episode return.
            avg_eps_len: Mean episode length.
            std_eps_rew: Standard deviation of episode returns.
            min_eps_rew: Minimum episode return.
            max_eps_rew: Maximum episode return.
            n_episodes: Number of completed episodes.
            trun_rate: Fraction of episodes ending by truncation.
            iteration: Current evaluation iteration.
        """

        avg_eps_rew: chex.Numeric
        avg_eps_len: chex.Numeric
        std_eps_rew: chex.Numeric
        min_eps_rew: chex.Numeric
        max_eps_rew: chex.Numeric
        n_episodes: chex.Numeric
        trun_rate: chex.Numeric
        iteration: chex.Numeric

    @dataclass
    class _Carry:
        imc: Imc.State
        done_count: chex.Numeric
        trun_count: chex.Numeric

    def _rollout(
        self,
        imc_state: Imc.State,
    ) -> tuple[Imc.State, chex.Numeric, chex.Numeric]:
        """Rollout a single environment for n_episodes.

        Args:
            imc_state: Imc state for a single environment.

        Returns:
            Updated Imc state, done count, and truncation count.
        """
        rollout_len = self.imc.mc.max_episode_len * self.n_episodes

        def step_fn(carry, _):
            transition, imc_state = self.imc.sample(carry.imc)
            done = jnp.logical_or(transition.term, transition.trun)
            return (
                carry.replace(
                    imc=imc_state,
                    done_count=carry.done_count + done,
                    trun_count=carry.trun_count + transition.trun,
                ),
                None,
            )

        carry, _ = jax.lax.scan(
            step_fn,
            Eval._Carry(
                imc=imc_state,
                done_count=jnp.array(0.0),
                trun_count=jnp.array(0.0),
            ),
            length=rollout_len,
            unroll=self._unroll,
        )

        return carry.imc, carry.done_count, carry.trun_count

    def metric(
        self,
        state: Eval.State,
        agent: Agent.State,
        env: chex.ArrayTree,
    ) -> tuple[Eval.State, Eval.Metrics]:
        """Evaluate agent by rolling out in the environment.

        Args:
            state: Current evaluation state.
            agent: Agent state to evaluate.
            env: Pre-initialized environment state (broadcast across envs).

        Returns:
            Updated evaluator state and computed metrics.
        """
        key, *env_keys = jrd.split(state.key, self.n_envs + 1)
        env_keys = jnp.stack(env_keys)

        sampler_states = jax.vmap(self.imc.mc.init, in_axes=(0, None))(
            env_keys, env
        )
        imc_states = self.imc.State(mc=sampler_states, agent=agent)

        batched_rollout = jax.vmap(self._rollout)
        imc_states, done_counts, trun_counts = batched_rollout(imc_states)

        eps_rew_queues = imc_states.mc.eps_rew_queue
        eps_len_queues = imc_states.mc.eps_len_queue
        total_done = jnp.sum(done_counts)
        total_trun = jnp.sum(trun_counts)

        return (
            state.replace(key=key, step=state.step + 1),
            Eval.Metrics(
                avg_eps_rew=jnp.nanmean(eps_rew_queues),
                avg_eps_len=jnp.nanmean(eps_len_queues),
                std_eps_rew=jnp.nanstd(eps_rew_queues),
                min_eps_rew=jnp.nanmin(eps_rew_queues),
                max_eps_rew=jnp.nanmax(eps_rew_queues),
                n_episodes=jnp.sum(~jnp.isnan(eps_rew_queues)),
                trun_rate=total_trun / jnp.maximum(total_done, 1),
                iteration=state.step + 1,
            ),
        )

    def init(self, key: chex.PRNGKey) -> Eval.State:
        """Initialize the evaluator state.

        Args:
            key: Random key for environment initialization.

        Returns:
            Initialized evaluator state.
        """
        return Eval.State(step=0, key=key)
