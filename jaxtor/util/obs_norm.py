"""Observation normalization as a stateful feature transform."""

from __future__ import annotations

from typing import Protocol

import jax
from chex import dataclass


class Mean(Protocol):
    """Statistics state required by ``ObsNorm``."""

    mean: jax.Array


class Stats[S: Mean](Protocol):
    """Statistics component required by ``ObsNorm``."""

    def init(self, shape: tuple[int, ...] = ()) -> S: ...
    def update(self, batch: jax.Array, state: S) -> S: ...
    def normalize(self, x: jax.Array, state: S) -> jax.Array: ...


@dataclass
class ObsNorm[StatsS: Mean]:
    """Normalize observations with explicitly updated running statistics.

    ``apply`` only reads the current statistics. Call :meth:`update` at the
    algorithm boundary where newly collected observations should enter the
    estimate. When disabled, both methods are pass-through operations; the
    static branch is removed during JAX tracing.

    Required protocols::

        stats.init(shape) -> stats_state
        stats.update(batch, stats_state) -> stats_state
        stats.normalize(value, stats_state) -> normalized_value
        stats_state.mean: jax.Array

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

    stats: Stats[StatsS]
    enabled: bool = True

    @dataclass
    class State[StatsData]:
        """Observation-normalization state.

        Attributes:
            stats: Running observation statistics.
        """

        stats: StatsData

    def init(self, shape: tuple[int, ...]) -> ObsNorm.State[StatsS]:
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
        state: ObsNorm.State[StatsS],
    ) -> tuple[jax.Array, ObsNorm.State[StatsS]]:
        """Normalize observations without changing their statistics."""
        if self.enabled:
            observations = self.stats.normalize(observations, state.stats)
        return observations, state

    def update(
        self,
        observations: jax.Array,
        state: ObsNorm.State[StatsS],
    ) -> ObsNorm.State[StatsS]:
        """Update statistics from arbitrary leading sample axes."""
        if not self.enabled:
            return state
        batch = self._batch(observations, state.stats.mean.shape)
        return self.State(stats=self.stats.update(batch, state.stats))
