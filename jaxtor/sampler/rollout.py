"""Fixed-length sequence collection from a stateful sampler.

``Roll`` scans one sampler without owning additional state::

    roll = Roll(imc=imc, seq_len=128, seq_axis=1)
    seq, state = roll.sample(state)

The sampler state is threaded through the scan. Each sampled pytree leaf is
stacked along ``seq_axis``.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass


class Sampler[Sample, S](Protocol):
    """Single-step sampler capability required by ``Roll``."""

    def sample(self, state: S) -> tuple[Sample, S]: ...


@dataclass
class Roll[Sample, S]:
    """Stack samples from a stateful sampler along one sequence axis.

    Attributes:
        imc: Stateful single-step sampler.
        seq_len: Number of samples in the sequence.
        seq_axis: Output axis carrying the temporal sequence.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public methods:
        sample: Stack a fixed number of samples and advance child state.
    """

    imc: Sampler[Sample, S]
    seq_len: int
    seq_axis: int = 0
    _unroll: int = 1

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if self._unroll < 1:
            raise ValueError("_unroll must be positive")

    def _advance(self, state: S, unused: None) -> tuple[S, Sample]:
        """Collect one sample for :func:`jax.lax.scan`."""
        del unused
        sample, state = self.imc.sample(state)
        return state, sample

    def sample(self, state: S) -> tuple[Sample, S]:
        """Stack ``seq_len`` samples and advance the sampler state."""
        state, samples = jax.lax.scan(
            self._advance,
            state,
            xs=None,
            length=self.seq_len,
            unroll=self._unroll,
        )
        if self.seq_axis != 0:
            samples = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis),
                samples,
            )
        return samples, state
