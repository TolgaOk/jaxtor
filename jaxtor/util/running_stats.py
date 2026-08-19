"""Online mean/variance tracking using the Welford algorithm.

Provides batch-updating running statistics and normalization.

Example:
    >>> stats = RunningStats(clip=10.0)
    >>> state = stats.init((4,))
    >>> state = stats.update(batch, state)
    >>> normalized = stats.normalize(obs, state)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass
class RunningStats:
    """Running mean/variance tracking (Welford algorithm).

    Attributes:
        clip: If not None, clip normalized output to [-clip, clip].

    Public dataclasses:
        State: Mean, variance, and sample count.

    Public methods:
        init: Initialize unit statistics for one feature shape.
        update: Merge a batch into the running statistics.
        normalize: Normalize values with the current statistics.
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

        mean: jax.Array
        var: jax.Array
        count: jax.Array

    def __post_init__(self) -> None:
        """Validate the optional symmetric clipping bound."""
        if self.clip is not None and self.clip < 0:
            raise ValueError("clip must be nonnegative")

    def init(self, shape: tuple[int, ...] = ()) -> RunningStats.State:
        """Initialize unit statistics for one feature shape."""
        return self.State(
            mean=jnp.zeros(shape),
            var=jnp.ones(shape),
            count=jnp.float32(1e-4),
        )

    def update(self, batch: jax.Array, state: RunningStats.State) -> RunningStats.State:
        """Update running mean/variance with a batch of samples.

        Uses the parallel Welford algorithm to merge batch statistics
        with the running statistics.

        Args:
            batch: Flat samples, shape (N,) or (N, dim).
            state: Current running statistics.

        Returns:
            Updated running statistics.
        """
        if batch.ndim < 1 or batch.shape[0] < 1:
            raise ValueError("batch must not be empty")
        if batch.shape[1:] != state.mean.shape:
            raise ValueError(
                f"expected batch feature shape {state.mean.shape}, got {batch.shape[1:]}"
            )

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

    def normalize(self, x: jax.Array, state: RunningStats.State) -> jax.Array:
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
