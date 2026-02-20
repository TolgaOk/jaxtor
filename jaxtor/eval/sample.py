"""Sampling-based evaluation.

Computes evaluation metrics from environment rollouts using episode statistics
tracked by the sampler.

Classes:
    Eval: Sampling-based evaluator with batched environment support.

Example:
    >>> imc = Imc(agent=agent, mc=mc)
    >>> evaluator = Eval(imc=imc, episode_len=100)
    >>> imc_state = Imc.State(mc=mc.init(keys, env_state), agent=agent_state)
    >>> metrics = evaluator.metric(imc_state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Imc(Protocol):
    class State(Protocol): ...

    class Transition(Protocol):
        term: chex.Array
        trun: chex.Array

    def sample(self, state: Imc.State) -> tuple[Transition, Imc.State]: ...


@dataclass
class Eval:
    """Sampling-based evaluator.

    Rolls out the policy in the environment and aggregates episode statistics
    from the sampler's queues.

    Attributes:
        imc: Induced Markov chain following the Imc protocol.
        episode_len: Number of steps to rollout per evaluation.
        _unroll: Loop unroll factor for jax.lax.scan.
    """

    imc: Imc
    episode_len: int
    _unroll: int = 1

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
        """

        avg_eps_rew: chex.Numeric
        avg_eps_len: chex.Numeric
        std_eps_rew: chex.Numeric
        min_eps_rew: chex.Numeric
        max_eps_rew: chex.Numeric
        n_episodes: chex.Numeric
        trun_rate: chex.Numeric

    @dataclass
    class _Carry:
        imc: Imc.State
        done_count: chex.Numeric
        trun_count: chex.Numeric

    def _rollout(
        self,
        imc_state: Imc.State,
    ) -> tuple[Imc.State, chex.Numeric, chex.Numeric]:
        """Rollout the environment for episode_len steps.

        Args:
            imc_state: Imc state.

        Returns:
            Updated Imc state, done count, and truncation count.
        """

        def step_fn(carry, _):
            transition, imc_state = self.imc.sample(carry.imc)
            chex.assert_equal_shape([transition.term, transition.trun])
            done = jnp.logical_or(transition.term, transition.trun)
            return (
                carry.replace(
                    imc=imc_state,
                    done_count=carry.done_count + jnp.sum(done),
                    trun_count=carry.trun_count + jnp.sum(transition.trun),
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
            length=self.episode_len,
            unroll=self._unroll,
        )

        return carry.imc, carry.done_count, carry.trun_count

    def metric(self, imc_state: Imc.State) -> Eval.Metrics:
        """Evaluate agent by rolling out in the environment.

        Args:
            imc_state: Imc state (vectorized if imc uses VecMc).

        Returns:
            Evaluation metrics.
        """
        imc_states, done_counts, trun_counts = self._rollout(imc_state)

        eps_rew_queues = imc_states.mc.eps_rew_queue
        eps_len_queues = imc_states.mc.eps_len_queue
        total_done = jnp.sum(done_counts)
        total_trun = jnp.sum(trun_counts)

        return Eval.Metrics(
            avg_eps_rew=jnp.nanmean(eps_rew_queues),
            avg_eps_len=jnp.nanmean(eps_len_queues),
            std_eps_rew=jnp.nanstd(eps_rew_queues),
            min_eps_rew=jnp.nanmin(eps_rew_queues),
            max_eps_rew=jnp.nanmax(eps_rew_queues),
            n_episodes=jnp.sum(~jnp.isnan(eps_rew_queues)),
            trun_rate=total_trun / jnp.maximum(total_done, 1),
        )
