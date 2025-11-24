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
    """

    n_step: int

    def q(self, q_arr: chex.Array, mdp: TabularMDP, mu: chex.Array) -> chex.Array:
        """Apply n-step propagation.

        Iteratively propagates value-like arrays through transition dynamics under policy μ,
        returning the sequence of arrays at each step starting from the initial input.

        N-step propagation trajectory:
            Q^(0) = q_arr (initial)
            Q^(k+1)(a,s) = Σ_s' P(s'|s,a) · [Σ_u μ(u|s') · Q^(k)(u,s')]  for k = 0, ..., n_step-2

        Returns: [Q^(0), Q^(1), ..., Q^(n_step-1)]

        Args:
            q_arr: Initial values (Q-values, returns, etc.).
                Shape: (A, S)
            mdp: TabularMDP with transition matrix P(s'|s,a).
                Shape: (A, S', S) where A=actions, S=states, S'=next_states
            mu: Policy distribution μ(a|s).
                Shape: (A, S)

        Returns:
            Trajectory of values from step 0 to step n_step-1.
            Shape: (n_step, A, S)
        """

        def _scan_body(carry, _):
            prop_arr = jnp.einsum("axs,ux,ux->as", mdp.transition, mu, carry)
            return prop_arr, prop_arr

        _, seq = jax.lax.scan(_scan_body, q_arr, length=self.n_step - 1)
        return jnp.concatenate([q_arr[None], seq], axis=0)

    def reverse(self):
        pass
