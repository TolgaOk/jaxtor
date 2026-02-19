"""N-step trajectory collection.

Collects fixed-length trajectories using jax.lax.scan over any Imc-compatible
sampler.

Example:
    >>> imc = Imc(agent=agent, mc=mc_sampler)
    >>> roll = Roll(imc=imc, seqlen=20)
    >>> state = Imc.State(mc=mc_state, agent=agent_state)
    >>> transitions, state = roll.sample(state)
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from chex import dataclass

Transition = TypeVar("Transition")
ImcState = TypeVar("ImcState")


class Imc(Protocol[Transition, ImcState]):
    def sample(self, state: ImcState) -> tuple[Transition, ImcState]: ...


@dataclass
class Roll(Generic[Transition, ImcState]):
    """N-step trajectory collector.

    Attributes:
        imc: Single-step sampler following the Imc protocol.
        seqlen: Number of steps to collect per trajectory.
        seq_axis: Axis for the sequence dimension in output transitions.
        _unroll: Loop unroll factor for jax.lax.scan.
    """

    imc: Imc
    seqlen: int
    seq_axis: int = 0
    _unroll: int = 1

    def sample(self, state: ImcState) -> tuple[Transition, ImcState]:
        """Collect seqlen transitions.

        Args:
            state: Current Imc state.

        Returns:
            Stacked transitions and updated state. Sequence dimension is
            placed at seq_axis (default 0). With VecMc, set seq_axis=1
            to get (n_env, seqlen, ...) instead of (seqlen, n_env, ...).
        """

        def step(state, _):
            transition, state = self.imc.sample(state)
            return state, transition

        state, transitions = jax.lax.scan(
            step, state, length=self.seqlen, unroll=self._unroll
        )
        if self.seq_axis != 0:
            transitions = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis), transitions
            )
        return transitions, state
