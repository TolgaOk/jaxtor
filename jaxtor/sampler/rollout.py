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
        _unroll: Loop unroll factor for jax.lax.scan.
    """

    imc: Imc
    seqlen: int
    _unroll: int = 1

    def sample(self, state: ImcState) -> tuple[Transition, ImcState]:
        """Collect seqlen transitions.

        Args:
            state: Current Imc state.

        Returns:
            Stacked transitions with shape (seqlen, ...) and updated state.
        """

        def step(state, _):
            transition, state = self.imc.sample(state)
            return state, transition

        state, transitions = jax.lax.scan(
            step, state, length=self.seqlen, unroll=self._unroll
        )
        return transitions, state
