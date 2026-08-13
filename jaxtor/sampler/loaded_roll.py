"""Fixed-length sequences with agent outputs at transition endpoints."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass


class AgentOutput(Protocol):
    """Minimum agent output consumed by ``LoadedRoll``."""

    act: jax.Array


class Agent[OutputT: AgentOutput, StateT](Protocol):
    """Agent application capability required by ``LoadedRoll``."""

    def apply(
        self,
        obs: jax.Array,
        state: StateT,
    ) -> tuple[OutputT, StateT]: ...


class McTransition(Protocol):
    """Markov-chain transition fields required by ``LoadedRoll``."""

    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


class MarkovChain[McT: McTransition, StateT](Protocol):
    """Open Markov-chain capability required by ``LoadedRoll``."""

    def observe(self, state: StateT) -> jax.Array: ...

    def sample(
        self,
        act: jax.Array,
        state: StateT,
    ) -> tuple[McT, StateT]: ...


@dataclass
class LoadedRoll[
    OutputT: AgentOutput,
    AgentStateT,
    McT: McTransition,
    McStateT,
]:
    """Collect rich agent outputs before and after each MC transition.

    Agent output at a normal successor is reused as the next decision. At a
    terminal or truncated boundary, the true successor remains in the sequence
    and a separate decision is predicted from the reset observation.

    Reuse is confined to the private scan carry. The returned :class:`State`
    contains no derived output, so changing its agent state cannot leave stale
    agent output behind. The final decision state is discarded because its
    action was not consumed; the next call computes it again from
    the then-current agent state.

    Attributes:
        agent: Agent producing a typed output that includes an action.
        mc: Open Markov-chain sampler advanced by agent actions.
        seq_len: Number of transitions in the sequence.
        seq_axis: Output axis carrying the temporal sequence.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public dataclasses:
        State: Persistent Markov-chain and agent states.
        Sequence: Agent outputs and MC transitions aligned over T steps.

    Public methods:
        init: Combine initialized child states.
        sample: Collect one fixed-length information-loaded sequence.
    """

    agent: Agent[OutputT, AgentStateT]
    mc: MarkovChain[McT, McStateT]
    seq_len: int
    seq_axis: int = 0
    _unroll: int = 1

    @dataclass
    class State[McDataT, AgentDataT]:
        """Persistent state threaded between loaded sequences.

        Attributes:
            mc: State of the open Markov chain.
            agent: Agent state after every action actually consumed by ``mc``.
        """

        mc: McDataT
        agent: AgentDataT

    @dataclass
    class Sequence[OutputDataT, McDataT]:
        """Time-aligned decision, transition, and successor pytrees.

        Attributes:
            dec: Agent decisions whose actions produced the transitions.
            mc: Markov-chain transitions produced by those actions.
            succ: Agent outputs predicted at each true successor observation.
        """

        dec: OutputDataT
        mc: McDataT
        succ: OutputDataT

    @dataclass
    class _Carry[McDataT, AgentDataT, OutputDataT]:
        """Scan-local state with one loaded decision.

        Attributes:
            mc: Current Markov-chain state.
            agent_before_dec: Agent state from which ``dec`` was computed.
            agent_after_dec: Agent state after computing ``dec``.
            dec: Loaded decision for the current observation.
        """

        mc: McDataT
        agent_before_dec: AgentDataT
        agent_after_dec: AgentDataT
        dec: OutputDataT

    @dataclass
    class _Next[McDataT, AgentDataT, OutputDataT]:
        """Data used to choose the next loaded decision.

        Attributes:
            mc: Markov-chain state after the transition and possible reset.
            agent: State after the current decision application.
            succ: Output computed at the true successor.
            succ_agent: Agent state returned with ``succ``.
        """

        mc: McDataT
        agent: AgentDataT
        succ: OutputDataT
        succ_agent: AgentDataT

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.seq_len < 1:
            raise ValueError("seq_len must be positive")
        if self._unroll < 1:
            raise ValueError("_unroll must be positive")

    def init(
        self,
        mc: McStateT,
        agent: AgentStateT,
    ) -> LoadedRoll.State[McStateT, AgentStateT]:
        """Combine initialized Markov-chain and agent states."""
        return self.State(mc=mc, agent=agent)

    def _start(
        self,
        state: LoadedRoll.State[McStateT, AgentStateT],
    ) -> LoadedRoll._Carry[McStateT, AgentStateT, OutputT]:
        """Predict the first decision and create the scan carry."""
        dec, agent_after_dec = self.agent.apply(
            self.mc.observe(state.mc),
            state.agent,
        )
        return self._Carry(
            mc=state.mc,
            agent_before_dec=state.agent,
            agent_after_dec=agent_after_dec,
            dec=dec,
        )

    def _reset_dec(
        self,
        next_data: LoadedRoll._Next[McStateT, AgentStateT, OutputT],
    ) -> tuple[OutputT, AgentStateT]:
        """Predict the next decision from the possibly reset MC state."""
        return self.agent.apply(
            self.mc.observe(next_data.mc),
            next_data.agent,
        )

    @staticmethod
    def _reuse_succ(
        next_data: LoadedRoll._Next[McStateT, AgentStateT, OutputT],
    ) -> tuple[OutputT, AgentStateT]:
        """Reuse the true successor as the next decision."""
        return next_data.succ, next_data.succ_agent

    def _advance(
        self,
        carry: LoadedRoll._Carry[McStateT, AgentStateT, OutputT],
        unused: None,
    ) -> tuple[
        LoadedRoll._Carry[McStateT, AgentStateT, OutputT],
        LoadedRoll.Sequence[OutputT, McT],
    ]:
        """Advance once and retain a valid decision for the next step."""
        del unused
        transition, mc = self.mc.sample(carry.dec.act, carry.mc)
        succ, succ_agent = self.agent.apply(
            transition.nobs,
            carry.agent_after_dec,
        )
        next_data = self._Next(
            mc=mc,
            agent=carry.agent_after_dec,
            succ=succ,
            succ_agent=succ_agent,
        )
        boundary = jnp.any(jnp.logical_or(transition.term, transition.trun))
        dec, agent_after_dec = jax.lax.cond(
            boundary,
            self._reset_dec,
            self._reuse_succ,
            next_data,
        )
        return (
            self._Carry(
                mc=mc,
                agent_before_dec=carry.agent_after_dec,
                agent_after_dec=agent_after_dec,
                dec=dec,
            ),
            self.Sequence(dec=carry.dec, mc=transition, succ=succ),
        )

    def sample(
        self,
        state: LoadedRoll.State[McStateT, AgentStateT],
    ) -> tuple[
        LoadedRoll.Sequence[OutputT, McT],
        LoadedRoll.State[McStateT, AgentStateT],
    ]:
        """Collect a sequence while reusing endpoint predictions only locally."""
        carry, seq = jax.lax.scan(
            self._advance,
            self._start(state),
            xs=None,
            length=self.seq_len,
            unroll=self._unroll,
        )
        if self.seq_axis != 0:
            seq = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis),
                seq,
            )
        state = self.State(mc=carry.mc, agent=carry.agent_before_dec)
        return seq, state
