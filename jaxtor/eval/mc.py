"""Sampling-based evaluation over an induced Markov chain.

``Eval`` derives episode metrics from public transition fields::

    evaluator = Eval(imc=imc, n_step=500)
    metrics, state = evaluator.evaluate(state)

Each call starts from a fresh episode boundary and advances the sampler state
without exposing its internal structure.
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Transition(Protocol):
    """Transition fields required for episode evaluation."""

    rew: jax.Array
    term: jax.Array
    trun: jax.Array


class Imc[Sample: Transition, S](Protocol):
    """Single-step sampler surface consumed by ``Eval``."""

    def sample(self, state: S) -> tuple[Sample, S]: ...


@dataclass
class Eval[Sample: Transition, S]:
    """Evaluate completed episodes from a fixed-length rollout.

    Attributes:
        imc: Single-step sampler consumed by the evaluator.
        n_step: Number of environment steps per evaluation.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public dataclasses:
        Metrics: Aggregate statistics for completed episodes.

    Public methods:
        evaluate: Evaluate one fresh fixed-step window.
    """

    imc: Imc[Sample, S]
    n_step: int
    _unroll: int = 1

    @dataclass
    class _Accumulator:
        """Scan-local episode and aggregate statistics.

        Attributes:
            episode_rew: Per-chain return of the current episode.
            episode_len: Per-chain length of the current episode.
            sum_eps_rew: Sum of completed episode returns.
            sum_sq_eps_rew: Sum of squared completed episode returns.
            sum_eps_len: Sum of completed episode lengths.
            min_eps_rew: Minimum completed episode return.
            max_eps_rew: Maximum completed episode return.
            n_episodes: Number of completed episodes.
            n_truncated: Number of completed episodes ending by truncation.
        """

        episode_rew: jax.Array
        episode_len: jax.Array
        sum_eps_rew: jax.Array
        sum_sq_eps_rew: jax.Array
        sum_eps_len: jax.Array
        min_eps_rew: jax.Array
        max_eps_rew: jax.Array
        n_episodes: jax.Array
        n_truncated: jax.Array

    @dataclass
    class _Carry[ImcData]:
        """Evaluation scan state.

        Attributes:
            imc: State of the wrapped sampler.
            accumulator: Episode statistics owned by the evaluator.
        """

        imc: ImcData
        accumulator: Eval._Accumulator

    @dataclass
    class Metrics:
        """Aggregate statistics for episodes completed during evaluation.

        Attributes:
            avg_eps_rew: Mean completed episode return.
            avg_eps_len: Mean completed episode length.
            std_eps_rew: Standard deviation of completed episode returns.
            min_eps_rew: Minimum completed episode return.
            max_eps_rew: Maximum completed episode return.
            n_episodes: Number of completed episodes.
            trun_rate: Fraction of completed episodes ending by truncation.
        """

        avg_eps_rew: jax.Array
        avg_eps_len: jax.Array
        std_eps_rew: jax.Array
        min_eps_rew: jax.Array
        max_eps_rew: jax.Array
        n_episodes: jax.Array
        trun_rate: jax.Array

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        if self._unroll < 1:
            raise ValueError("_unroll must be positive")

    @staticmethod
    def _init_accumulator(reward: jax.Array) -> Eval._Accumulator:
        """Initialize statistics with the reward's batch shape."""
        dtype = jnp.result_type(reward.dtype, jnp.float32)
        reward_zero = reward.astype(dtype) * 0
        episode_len = reward.astype(jnp.int32) * 0
        scalar_zero = jnp.sum(reward_zero)
        return Eval._Accumulator(
            episode_rew=reward_zero,
            episode_len=episode_len,
            sum_eps_rew=scalar_zero,
            sum_sq_eps_rew=scalar_zero,
            sum_eps_len=scalar_zero,
            min_eps_rew=scalar_zero + jnp.inf,
            max_eps_rew=scalar_zero - jnp.inf,
            n_episodes=jnp.sum(episode_len),
            n_truncated=jnp.sum(episode_len),
        )

    @staticmethod
    def _accumulate(
        accumulator: Eval._Accumulator,
        transition: Transition,
    ) -> tuple[Eval._Accumulator, None]:
        """Add one timestep of possibly batched transitions to statistics."""
        chex.assert_equal_shape([transition.rew, transition.term, transition.trun])

        reward = transition.rew.astype(accumulator.episode_rew.dtype)
        term = transition.term.astype(jnp.bool_)
        trun = transition.trun.astype(jnp.bool_)
        done = jnp.logical_or(term, trun)

        episode_rew = accumulator.episode_rew + reward
        episode_len = accumulator.episode_len + 1
        completed_rew = jnp.where(done, episode_rew, 0)
        completed_len = jnp.where(done, episode_len, 0)
        upper = episode_rew * 0 + jnp.inf
        lower = episode_rew * 0 - jnp.inf

        return (
            Eval._Accumulator(
                episode_rew=jnp.where(done, 0, episode_rew),
                episode_len=jnp.where(done, 0, episode_len),
                sum_eps_rew=accumulator.sum_eps_rew + jnp.sum(completed_rew),
                sum_sq_eps_rew=(accumulator.sum_sq_eps_rew + jnp.sum(completed_rew**2)),
                sum_eps_len=(accumulator.sum_eps_len + jnp.sum(completed_len)),
                min_eps_rew=jnp.minimum(
                    accumulator.min_eps_rew,
                    jnp.min(jnp.where(done, episode_rew, upper)),
                ),
                max_eps_rew=jnp.maximum(
                    accumulator.max_eps_rew,
                    jnp.max(jnp.where(done, episode_rew, lower)),
                ),
                n_episodes=accumulator.n_episodes + jnp.sum(done),
                n_truncated=(
                    accumulator.n_truncated + jnp.sum(jnp.logical_and(done, trun))
                ),
            ),
            None,
        )

    @staticmethod
    def _summarize(accumulator: Eval._Accumulator) -> Eval.Metrics:
        """Convert aggregate sufficient statistics into evaluation metrics."""
        dtype = accumulator.sum_eps_rew.dtype
        has_episodes = accumulator.n_episodes > 0
        count = jnp.maximum(accumulator.n_episodes, 1).astype(dtype)
        avg_eps_rew = accumulator.sum_eps_rew / count
        variance = jnp.maximum(
            accumulator.sum_sq_eps_rew / count - avg_eps_rew**2,
            0,
        )
        nan = accumulator.sum_eps_rew * 0 + jnp.nan

        return Eval.Metrics(
            avg_eps_rew=jnp.where(has_episodes, avg_eps_rew, nan),
            avg_eps_len=jnp.where(has_episodes, accumulator.sum_eps_len / count, nan),
            std_eps_rew=jnp.where(has_episodes, jnp.sqrt(variance), nan),
            min_eps_rew=jnp.where(has_episodes, accumulator.min_eps_rew, nan),
            max_eps_rew=jnp.where(has_episodes, accumulator.max_eps_rew, nan),
            n_episodes=accumulator.n_episodes,
            trun_rate=accumulator.n_truncated.astype(dtype) / count,
        )

    def _step(self, carry: Eval._Carry[S], unused: None) -> tuple[Eval._Carry[S], None]:
        """Sample and accumulate one evaluation step."""
        del unused
        transition, imc = self.imc.sample(carry.imc)
        accumulator, _ = self._accumulate(carry.accumulator, transition)
        return self._Carry(imc=imc, accumulator=accumulator), None

    def evaluate(self, state: S) -> tuple[Eval.Metrics, S]:
        """Evaluate completed episodes and return the advanced sampler state.

        The input state must begin at an episode boundary. Metrics cover every
        episode completed during this call; incomplete trailing episodes are
        excluded. The returned state may be mid-episode and should not seed a
        later evaluation call.

        Args:
            state: Freshly initialized sampler state.

        Returns:
            Evaluation metrics and the advanced sampler state.
        """
        transition, state = self.imc.sample(state)
        accumulator, _ = self._accumulate(
            self._init_accumulator(transition.rew),
            transition,
        )
        carry, _ = jax.lax.scan(
            self._step,
            self._Carry(imc=state, accumulator=accumulator),
            None,
            length=self.n_step - 1,
            unroll=self._unroll,
        )
        return self._summarize(carry.accumulator), carry.imc
