"""Fixed-length decision, Markov-chain, and successor collection."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass


class ImcSample[McT, DecT](Protocol):
    """Single-step sample surface consumed by ``Roll``."""

    mc: McT
    succ: DecT


class ObservedImc[DecT, McT, StateT](Protocol):
    """Observed single-step sampler surface consumed by ``Roll``."""

    def observe(self, state: StateT) -> DecT: ...

    def sample(self, state: StateT) -> tuple[ImcSample[McT, DecT], StateT]: ...


@dataclass
class Roll[DecT, McT, StateT]:
    """Collect aligned decisions, Markov-chain transitions, and successors.

    Attributes:
        imc: Single-step sampler exposing ``observe`` and ``sample``.
        seqlen: Number of transitions in the trajectory.
        seq_axis: Output axis carrying the temporal sequence.
        _unroll: Loop-unroll factor passed to ``jax.lax.scan``.

    Public dataclasses:
        Trajectory: Decisions, MC transitions, and successors, each of length T.

    Public methods:
        sample: Collect one fixed-length trajectory and advance child state.
    """

    imc: ObservedImc[DecT, McT, StateT]
    seqlen: int
    seq_axis: int = 0
    _unroll: int = 1

    @dataclass
    class Trajectory[DecDataT, McDataT]:
        """Aligned decision, Markov-chain, and successor pytrees.

        Attributes:
            dec: Agent decisions whose actions were consumed.
            mc: Markov-chain transitions produced by those actions.
            succ: Decisions at true successors, including across autoresets.
        """

        dec: DecDataT
        mc: McDataT
        succ: DecDataT

    def sample(
        self,
        state: StateT,
    ) -> tuple[Roll.Trajectory[DecT, McT], StateT]:
        """Collect ``seqlen`` aligned decision-MC-successor triples."""

        def advance(
            state: StateT,
            unused: None,
        ) -> tuple[StateT, tuple[DecT, McT, DecT]]:
            del unused
            dec = self.imc.observe(state)
            sample, state = self.imc.sample(state)
            return state, (dec, sample.mc, sample.succ)

        state, (dec, mc, succ) = jax.lax.scan(
            advance,
            state,
            xs=None,
            length=self.seqlen,
            unroll=self._unroll,
        )
        if self.seq_axis != 0:
            dec, mc, succ = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis),
                (dec, mc, succ),
            )
        return self.Trajectory(dec=dec, mc=mc, succ=succ), state
