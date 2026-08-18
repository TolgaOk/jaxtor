"""Shuffled minibatches from array pytrees."""

from __future__ import annotations

import math

import jax
import jax.random as jrd
from chex import dataclass


@dataclass
class Minibatches:
    """Shuffle and split a pytree along its leading batch axis.

    Attributes:
        count: Number of equal-size minibatches.
        sample_ndim: Number of leading sample axes collapsed before shuffling.

    Public methods:
        shuffle: Shuffle aligned leaves and add a leading minibatch axis.
    """

    count: int
    sample_ndim: int = 1

    def __post_init__(self) -> None:
        """Validate the static batching configuration."""
        if self.count < 1:
            raise ValueError("count must be positive")
        if self.sample_ndim < 1:
            raise ValueError("sample_ndim must be positive")

    def shuffle[BatchT](self, key: jax.Array, batch: BatchT) -> BatchT:
        """Collapse sample axes and return ``(count, size, ...)`` leaves."""
        leaves = jax.tree.leaves(batch)
        if not leaves:
            raise ValueError("batch must contain at least one array")
        if any(leaf.ndim < self.sample_ndim for leaf in leaves):
            raise ValueError("batch leaves have fewer axes than sample_ndim")

        sample_shape = leaves[0].shape[: self.sample_ndim]
        if any(leaf.shape[: self.sample_ndim] != sample_shape for leaf in leaves):
            raise ValueError("batch leaves must share their leading sample axes")

        batch_size = math.prod(sample_shape)
        if batch_size < 1:
            raise ValueError("batch must not be empty")
        if batch_size % self.count:
            raise ValueError("batch size must be divisible by count")

        order = jrd.permutation(key, batch_size)
        size = batch_size // self.count
        return jax.tree.map(
            lambda leaf: leaf.reshape(
                batch_size,
                *leaf.shape[self.sample_ndim :],
            )[order].reshape(self.count, size, *leaf.shape[self.sample_ndim :]),
            batch,
        )
