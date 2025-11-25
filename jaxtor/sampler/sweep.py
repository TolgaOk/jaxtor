"""MDP state-space sweeping utilities.

Implements n-step value propagation operators for tabular MDPs using exact
transition dynamics. Provides matrix-vector operations.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


@dataclass
class TabularMDP:
    transition: chex.Array


@dataclass
class Sweep:
    """N-step propagation for tabular MDPs.

    Attributes:
        n_step: Number of propagation steps to perform.
        _unroll: Number of loop iterations to unroll in scan (default: 1).
    """

    n_step: int
    _unroll: int = 1

    def backward(
        self, q_arr: chex.Array, mdp: TabularMDP, mu: chex.Array
    ) -> chex.Array:
        """Apply n-step backward propagation.

        Iteratively propagates value-like arrays backward through transition dynamics under
        policy μ, returning the sequence from initial to most propagated values.

        N-step backward propagation trajectory:
            Q_prop(a,s) = Σ_s' P(s'|s,a) · [Σ_u μ(u|s') · Q(u,s')]  (single propagation)
            Q^(0) = q_arr  (initial, no propagation)
            Q^(k) = (P^μ)^k q_arr  for k = 1, ..., n_step-1
            Q^(n_step-1) = (P^μ)^(n_step-1) q_arr  (most propagated)

        Returns: [Q^(0), Q^(1), ..., Q^(n_step-1)]

        Args:
            q_arr: Initial values (Q-values, returns, etc.).
                Shape: (A, S)
            mdp: TabularMDP with transition matrix P(s'|s,a).
                Shape: (A, S', S) where A=actions, S=states, S'=next_states
            mu: Policy distribution μ(a|s).
                Shape: (A, S)

        Returns:
            Trajectory of values from initial to most propagated.
            Shape: (n_step, A, S)
        """

        def _scan_body(carry, _):
            prop_arr = jnp.einsum("axs,ux,ux->as", mdp.transition, mu, carry)
            return prop_arr, prop_arr

        _, seq = jax.lax.scan(
            _scan_body,
            q_arr,
            length=self.n_step - 1,
            unroll=self._unroll,
        )
        return jnp.concatenate([q_arr[None], seq], axis=0)

    def forward(
        self, pi_arr: chex.Array, mdp: TabularMDP, mu: chex.Array
    ) -> chex.Array:
        """Apply n-step forward propagation.

        Iteratively propagates distribution-like arrays forward through transition dynamics
        under policy μ, returning the sequence of arrays at each step starting from the initial input.

        N-step forward propagation trajectory:
            π^(0) = pi_arr (initial)
            π^(k+1)(a',s') = Σ_s Σ_a π^(k)(a,s) · P(s'|s,a) · μ(a'|s')  for k = 0, ..., n_step-2

        Returns: [π^(0), π^(1), ..., π^(n_step-1)]

        Args:
            pi_arr: Initial distribution (state-action occupancy, etc.).
                Shape: (A, S)
            mdp: TabularMDP with transition matrix P(s'|s,a).
                Shape: (A, S', S) where A=actions, S=states, S'=next_states
            mu: Policy distribution μ(a|s).
                Shape: (A, S)

        Returns:
            Trajectory of distributions from step 0 to step n_step-1.
            Shape: (n_step, A, S)
        """

        def _scan_body(carry, _):
            prop_arr = jnp.einsum("as,axs,ux->ux", carry, mdp.transition, mu)
            return prop_arr, prop_arr

        _, seq = jax.lax.scan(
            _scan_body, pi_arr, length=self.n_step - 1, unroll=self._unroll
        )
        return jnp.concatenate([pi_arr[None], seq], axis=0)
