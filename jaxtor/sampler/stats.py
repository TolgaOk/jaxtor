"""Completed-episode statistics from fixed-length sequences."""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Sequence(Protocol):
    """Sequence fields consumed by ``EpisodeStats``.

    Each field has the same shape and contains at least one sequence axis. A
    scalar rollout uses ``(T,)``. A vector rollout commonly uses ``(N, T)``
    with ``EpisodeStats(seq_axis=1)``.

    Attributes:
        rew: Rewards with minimum shape ``(T,)``.
        term: Termination flags with the same shape as ``rew``.
        trun: Truncation flags with the same shape as ``rew``.
    """

    rew: jax.Array  # Minimum (T,); commonly (N, T) when seq_axis=1.
    term: jax.Array  # Same shape as rew.
    trun: jax.Array  # Same shape as rew.


@dataclass
class EpisodeStats:
    """Accumulate completed-episode statistics across sequence batches.

    ``seq_axis`` identifies the sequence time axis. Every other axis in the
    reward and boundary arrays identifies an independent environment lane.
    Partial episodes remain in state across calls to :meth:`update`.

    Attributes:
        seq_axis: Axis containing consecutive environment transitions.

    Public dataclasses:
        State: Partial episodes and completed-episode accumulators.
        Metrics: Episode-weighted averages since the previous drain.

    Public methods:
        init: Initialize statistics for an environment batch shape.
        update: Accumulate one fixed-length sequence.
        drain: Read completed-episode metrics and clear their accumulators.
    """

    seq_axis: int = 0

    @dataclass
    class State:
        """Dynamic statistics state.

        Attributes:
            eps_rew: Return of each unfinished episode.
            eps_len: Length of each unfinished episode.
            sum_eps_rew: Sum of returns from completed episodes.
            sum_eps_len: Sum of lengths from completed episodes.
            n_episodes: Number of completed episodes.
        """

        eps_rew: jax.Array
        eps_len: jax.Array
        sum_eps_rew: jax.Array
        sum_eps_len: jax.Array
        n_episodes: jax.Array

    @dataclass
    class Metrics:
        """Completed-episode metrics since the previous drain.

        Attributes:
            avg_eps_rew: Mean completed-episode return.
            avg_eps_len: Mean completed-episode length.
            n_episodes: Number of completed episodes.
        """

        avg_eps_rew: jax.Array
        avg_eps_len: jax.Array
        n_episodes: jax.Array

    @dataclass
    class _Step:
        """One time slice reduced across all environment lanes."""

        rew: jax.Array
        done: jax.Array

    def init(self, batch_shape: tuple[int, ...] = ()) -> EpisodeStats.State:
        """Initialize partial episodes for an environment batch.

        Args:
            batch_shape: ``()`` for ``Mc`` or commonly ``(N,)`` for ``VecMc``.
        """
        return self.State(
            eps_rew=jnp.zeros(batch_shape),
            eps_len=jnp.zeros(batch_shape, dtype=jnp.int32),
            sum_eps_rew=jnp.array(0.0),
            sum_eps_len=jnp.array(0, dtype=jnp.int32),
            n_episodes=jnp.array(0, dtype=jnp.int32),
        )

    @staticmethod
    def _update_step(
        state: EpisodeStats.State,
        step: EpisodeStats._Step,
    ) -> tuple[EpisodeStats.State, None]:
        """Accumulate one time slice and reset completed lanes."""
        eps_rew = state.eps_rew + step.rew
        eps_len = state.eps_len + 1
        completed_rew = jnp.where(step.done, eps_rew, 0)
        completed_len = jnp.where(step.done, eps_len, 0)
        state = state.replace(  # type: ignore[reportAttributeAccessIssue]
            eps_rew=jnp.where(step.done, 0, eps_rew),
            eps_len=jnp.where(step.done, 0, eps_len),
            sum_eps_rew=state.sum_eps_rew + jnp.sum(completed_rew),
            sum_eps_len=state.sum_eps_len + jnp.sum(completed_len),
            n_episodes=state.n_episodes + jnp.sum(step.done, dtype=jnp.int32),
        )
        return state, None

    def update(
        self,
        seq: Sequence,
        state: EpisodeStats.State,
    ) -> EpisodeStats.State:
        """Accumulate completed and partial episodes from ``seq``."""
        rew = jnp.moveaxis(jnp.asarray(seq.rew), self.seq_axis, 0)
        term = jnp.moveaxis(
            jnp.asarray(seq.term, dtype=jnp.bool_),
            self.seq_axis,
            0,
        )
        trun = jnp.moveaxis(
            jnp.asarray(seq.trun, dtype=jnp.bool_),
            self.seq_axis,
            0,
        )
        chex.assert_equal_shape([rew, term, trun])
        chex.assert_equal_shape([rew[0], state.eps_rew])

        state, _ = jax.lax.scan(
            self._update_step,
            state,
            self._Step(rew=rew, done=jnp.logical_or(term, trun)),
        )
        return state

    def drain(
        self,
        state: EpisodeStats.State,
    ) -> tuple[EpisodeStats.Metrics, EpisodeStats.State]:
        """Return completed-episode metrics and preserve partial episodes."""
        has_episodes = state.n_episodes > 0
        count = jnp.maximum(state.n_episodes, 1).astype(state.sum_eps_rew.dtype)
        nan = jnp.array(jnp.nan, dtype=state.sum_eps_rew.dtype)
        metrics = self.Metrics(
            avg_eps_rew=jnp.where(has_episodes, state.sum_eps_rew / count, nan),
            avg_eps_len=jnp.where(has_episodes, state.sum_eps_len / count, nan),
            n_episodes=state.n_episodes,
        )
        state = state.replace(  # type: ignore[reportAttributeAccessIssue]
            sum_eps_rew=jnp.zeros_like(state.sum_eps_rew),
            sum_eps_len=jnp.zeros_like(state.sum_eps_len),
            n_episodes=jnp.zeros_like(state.n_episodes),
        )
        return metrics, state
