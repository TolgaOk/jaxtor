"""Expected value propagation for tabular MDPs.

Implements n-step value propagation operators using exact/expected transition
dynamics. Provides matrix-vector operations for backward (value) and forward
(distribution) propagation.
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Mdp(Protocol):
    """Tabular dynamics consumed by ``ExpSweep``."""

    transition: jax.Array


@dataclass
class ExpSweep:
    """Propagate action-state arrays through exact finite-MDP dynamics.

    Required protocols::

        mdp.transition: jax.Array

    Attributes:
        n_step: Number of arrays in the returned propagation sequence.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public methods:
        backward: Propagate value-like arrays backward through the dynamics.
        forward: Propagate occupancy-like arrays forward through the dynamics.
    """

    n_step: int
    _unroll: int = 1

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        if self._unroll < 1:
            raise ValueError("_unroll must be positive")

    def backward(self, q: jax.Array, mdp: Mdp, policy: jax.Array) -> jax.Array:
        """Return zero through ``n_step - 1`` backward propagations.

        Args:
            q: Initial action-state values with shape ``(A, S)``.
            mdp: Tabular dynamics with transition shape ``(A, S_next, S)``.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Propagation sequence with shape ``(n_step, A, S)``.
        """

        chex.assert_rank([q, policy], 2)
        chex.assert_rank(mdp.transition, 3)
        chex.assert_equal_shape([q, policy])
        chex.assert_axis_dimension(mdp.transition, 0, q.shape[0])
        chex.assert_axis_dimension(mdp.transition, 1, q.shape[1])
        chex.assert_axis_dimension(mdp.transition, 2, q.shape[1])

        def _scan_body(carry, _):
            propagated = jnp.einsum(
                "axs,ux,ux->as",
                mdp.transition,
                policy,
                carry,
            )
            return propagated, propagated

        _, sequence = jax.lax.scan(
            _scan_body,
            q,
            length=self.n_step - 1,
            unroll=self._unroll,
        )
        return jnp.concatenate([q[None], sequence], axis=0)

    def forward(
        self,
        occupancy: jax.Array,
        mdp: Mdp,
        policy: jax.Array,
    ) -> jax.Array:
        """Return zero through ``n_step - 1`` forward propagations.

        Args:
            occupancy: Initial action-state occupancy with shape ``(A, S)``.
            mdp: Tabular dynamics with transition shape ``(A, S_next, S)``.
            policy: Action probabilities with shape ``(A, S)``.

        Returns:
            Propagation sequence with shape ``(n_step, A, S)``.
        """

        chex.assert_rank([occupancy, policy], 2)
        chex.assert_rank(mdp.transition, 3)
        chex.assert_equal_shape([occupancy, policy])
        chex.assert_axis_dimension(mdp.transition, 0, occupancy.shape[0])
        chex.assert_axis_dimension(mdp.transition, 1, occupancy.shape[1])
        chex.assert_axis_dimension(mdp.transition, 2, occupancy.shape[1])

        def _scan_body(carry, _):
            propagated = jnp.einsum(
                "as,axs,ux->ux",
                carry,
                mdp.transition,
                policy,
            )
            return propagated, propagated

        _, sequence = jax.lax.scan(
            _scan_body,
            occupancy,
            length=self.n_step - 1,
            unroll=self._unroll,
        )
        return jnp.concatenate([occupancy[None], sequence], axis=0)
