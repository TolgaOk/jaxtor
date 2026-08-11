"""Fixed-length trajectory collection from a step sampler."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass


class Sampler[SampleT, StateT](Protocol):
    """Single-step sampler capability required by ``Roll``."""

    def sample(self, state: StateT) -> tuple[SampleT, StateT]: ...


@dataclass
class Roll[SampleT, StateT]:
    """Stack samples from a stateful sampler along one sequence axis.

    Attributes:
        imc: Stateful single-step sampler.
        seqlen: Number of transitions in the trajectory.
        seq_axis: Output axis carrying the temporal sequence.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public methods:
        sample: Stack a fixed number of samples and advance child state.
    """

    imc: Sampler[SampleT, StateT]
    seqlen: int
    seq_axis: int = 0
    _unroll: int = 1

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.seqlen < 1:
            raise ValueError("seqlen must be positive")
        if self._unroll < 1:
            raise ValueError("_unroll must be positive")

    def _advance(
        self,
        state: StateT,
        unused: None,
    ) -> tuple[StateT, SampleT]:
        """Collect one sample for :func:`jax.lax.scan`."""
        del unused
        sample, state = self.imc.sample(state)
        return state, sample

    def sample(
        self,
        state: StateT,
    ) -> tuple[SampleT, StateT]:
        """Stack ``seqlen`` samples and advance the sampler state."""
        state, samples = jax.lax.scan(
            self._advance,
            state,
            xs=None,
            length=self.seqlen,
            unroll=self._unroll,
        )
        if self.seq_axis != 0:
            samples = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis),
                samples,
            )
        return samples, state
