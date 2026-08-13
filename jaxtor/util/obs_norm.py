"""Observation normalization as a stateful feature transform."""

from __future__ import annotations

import jax
from chex import dataclass

from jaxtor.util.running_stats import RunningStats


@dataclass
class ObsNorm:
    """Normalize observations with explicitly updated running statistics.

    ``apply`` only reads the current statistics. Call :meth:`update` at the
    algorithm boundary where newly collected observations should enter the
    estimate. When disabled, both methods are pass-through operations; the
    static branch is removed during JAX tracing.

    Attributes:
        stats: Running-statistics component used for normalization.
        enabled: Whether normalization and statistics updates are active.

    Public dataclasses:
        State: Running observation statistics.

    Public methods:
        init: Initialize statistics for one observation shape.
        apply: Normalize observations without updating statistics.
        update: Add observations to the running statistics.
    """

    stats: RunningStats
    enabled: bool = True

    @dataclass
    class State:
        """Observation-normalization state.

        Attributes:
            stats: Running observation statistics.
        """

        stats: RunningStats.State

    def init(self, shape: tuple[int, ...]) -> ObsNorm.State:
        """Initialize unit statistics for one observation shape."""
        return self.State(stats=self.stats.init(shape))

    @staticmethod
    def _batch(observations: jax.Array, shape: tuple[int, ...]) -> jax.Array:
        """Flatten leading sample axes while preserving observation axes."""
        if observations.ndim < len(shape):
            raise ValueError("observations have fewer axes than their feature shape")
        if shape and observations.shape[-len(shape) :] != shape:
            raise ValueError(
                f"expected observation suffix {shape}, got {observations.shape}"
            )
        return observations.reshape((-1, *shape))

    def apply(
        self,
        observations: jax.Array,
        state: ObsNorm.State,
    ) -> tuple[jax.Array, ObsNorm.State]:
        """Normalize observations without changing their statistics."""
        if self.enabled:
            observations = self.stats.normalize(observations, state.stats)
        return observations, state

    def update(
        self,
        observations: jax.Array,
        state: ObsNorm.State,
    ) -> ObsNorm.State:
        """Update statistics from arbitrary leading sample axes."""
        if not self.enabled:
            return state
        batch = self._batch(observations, state.stats.mean.shape)
        return self.State(stats=self.stats.update(batch, state.stats))
