"""Online mean/variance tracking using the Welford algorithm.

Provides batch-updating running statistics and normalization.

Example:
    >>> rs = RunningStats(clip=10.0)
    >>> state = RunningStats.State(
    ...     mean=jnp.zeros(4), var=jnp.ones(4), count=jnp.float32(1e-4)
    ... )
    >>> state = rs.update(batch, state)
    >>> normalized = rs.normalize(obs, state)
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
from chex import dataclass


@dataclass
class RunningStats:
    """Running mean/variance tracking (Welford algorithm).

    Attributes:
        clip: If not None, clip normalized output to [-clip, clip].
    """

    clip: float | None = None

    @dataclass
    class State:
        """Running statistics state.

        Attributes:
            mean: Running mean, shape matches input feature dims.
            var: Running variance, shape matches input feature dims.
            count: Total number of samples seen.
        """

        mean: chex.Array
        var: chex.Array
        count: chex.Numeric

    def update(
        self, batch: chex.Array, state: RunningStats.State
    ) -> RunningStats.State:
        """Update running mean/variance with a batch of samples.

        Uses the parallel Welford algorithm to merge batch statistics
        with the running statistics.

        Args:
            batch: Flat samples, shape (N,) or (N, dim).
            state: Current running statistics.

        Returns:
            Updated running statistics.
        """
        batch_mean = jnp.mean(batch, axis=0)
        batch_var = jnp.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - state.mean
        total = state.count + batch_count
        new_mean = state.mean + delta * batch_count / total
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * state.count * batch_count / total
        new_var = m2 / total
        return RunningStats.State(mean=new_mean, var=new_var, count=total)

    def normalize(self, x: chex.Array, state: RunningStats.State) -> chex.Array:
        """Normalize by mean/variance, optionally clip per config.

        Args:
            x: Input array to normalize.
            state: Running statistics for normalization.

        Returns:
            Normalized (and optionally clipped) array.
        """
        normed = (x - state.mean) / jnp.sqrt(state.var + 1e-8)
        if self.clip is not None:
            normed = jnp.clip(normed, -self.clip, self.clip)
        return normed
