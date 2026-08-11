"""Fixed-length rollout with agent outputs at transition endpoints."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from chex import dataclass


class AgentOutput(Protocol):
    """Minimum agent output consumed by ``LoadedRoll``."""

    act: jax.Array


class Agent[OutputT: AgentOutput, StateT](Protocol):
    """Inference capability required by ``LoadedRoll``."""

    def infer(
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

    Agent output at a normal successor is reused as the next predecessor
    output. At a terminal or truncated boundary, the true successor remains in
    the trajectory and a separate output is inferred from the reset
    observation.

    Reuse is confined to the private scan carry. The returned :class:`State`
    contains no derived output, so changing its agent state cannot leave stale
    inference behind. The final inferred predecessor state is discarded
    because its action was not consumed; the next call infers it again from
    the then-current agent state.

    Attributes:
        agent: Agent producing a typed output that includes an action.
        mc: Open Markov-chain sampler advanced by agent actions.
        seqlen: Number of transitions in the trajectory.
        seq_axis: Output axis carrying the temporal sequence.
        _unroll: Loop-unroll factor passed to :func:`jax.lax.scan`.

    Public dataclasses:
        State: Persistent Markov-chain and agent states.
        Trajectory: Agent outputs and MC transitions aligned over T steps.

    Public methods:
        init: Combine initialized child states.
        sample: Collect one fixed-length information-loaded trajectory.
    """

    agent: Agent[OutputT, AgentStateT]
    mc: MarkovChain[McT, McStateT]
    seqlen: int
    seq_axis: int = 0
    _unroll: int = 1

    @dataclass
    class State[McDataT, AgentDataT]:
        """Persistent state threaded between loaded rollouts.

        Attributes:
            mc: State of the open Markov chain.
            agent: Agent state after every action actually consumed by ``mc``.
        """

        mc: McDataT
        agent: AgentDataT

    @dataclass
    class Trajectory[OutputDataT, McDataT]:
        """Time-aligned predecessor, transition, and successor pytrees.

        Attributes:
            pre: Agent outputs whose actions produced the transitions.
            mc: Markov-chain transitions produced by those actions.
            succ: Agent outputs inferred at each true successor observation.
        """

        pre: OutputDataT
        mc: McDataT
        succ: OutputDataT

    @dataclass
    class _Carry[McDataT, AgentDataT, OutputDataT]:
        """Scan-local state with one loaded predecessor output.

        Attributes:
            mc: Current Markov-chain state.
            agent_before_pre: Agent state from which ``pre`` was inferred.
            agent_after_pre: Agent state after inferring ``pre``.
            pre: Loaded output for the current predecessor observation.
        """

        mc: McDataT
        agent_before_pre: AgentDataT
        agent_after_pre: AgentDataT
        pre: OutputDataT

    @dataclass
    class _Next[McDataT, AgentDataT, OutputDataT]:
        """Data used to choose the next loaded predecessor.

        Attributes:
            mc: Markov-chain state after the transition and possible reset.
            agent: State after the current predecessor inference.
            succ: Output inferred at the true successor.
            succ_agent: Agent state returned with ``succ``.
        """

        mc: McDataT
        agent: AgentDataT
        succ: OutputDataT
        succ_agent: AgentDataT

    def __post_init__(self) -> None:
        """Validate the static scan configuration."""
        if self.seqlen < 1:
            raise ValueError("seqlen must be positive")
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
        """Infer the first predecessor and create the scan carry."""
        pre, agent_after_pre = self.agent.infer(
            self.mc.observe(state.mc),
            state.agent,
        )
        return self._Carry(
            mc=state.mc,
            agent_before_pre=state.agent,
            agent_after_pre=agent_after_pre,
            pre=pre,
        )

    def _reset_pre(
        self,
        next_data: LoadedRoll._Next[McStateT, AgentStateT, OutputT],
    ) -> tuple[OutputT, AgentStateT]:
        """Infer the next predecessor from the possibly reset MC state."""
        return self.agent.infer(
            self.mc.observe(next_data.mc),
            next_data.agent,
        )

    @staticmethod
    def _reuse_succ(
        next_data: LoadedRoll._Next[McStateT, AgentStateT, OutputT],
    ) -> tuple[OutputT, AgentStateT]:
        """Reuse the true successor as the next predecessor."""
        return next_data.succ, next_data.succ_agent

    def _advance(
        self,
        carry: LoadedRoll._Carry[McStateT, AgentStateT, OutputT],
        unused: None,
    ) -> tuple[
        LoadedRoll._Carry[McStateT, AgentStateT, OutputT],
        LoadedRoll.Trajectory[OutputT, McT],
    ]:
        """Advance once and retain a valid predecessor for the next step."""
        del unused
        transition, mc = self.mc.sample(carry.pre.act, carry.mc)
        succ, succ_agent = self.agent.infer(
            transition.nobs,
            carry.agent_after_pre,
        )
        next_data = self._Next(
            mc=mc,
            agent=carry.agent_after_pre,
            succ=succ,
            succ_agent=succ_agent,
        )
        boundary = jnp.any(jnp.logical_or(transition.term, transition.trun))
        pre, agent_after_pre = jax.lax.cond(
            boundary,
            self._reset_pre,
            self._reuse_succ,
            next_data,
        )
        return (
            self._Carry(
                mc=mc,
                agent_before_pre=carry.agent_after_pre,
                agent_after_pre=agent_after_pre,
                pre=pre,
            ),
            self.Trajectory(pre=carry.pre, mc=transition, succ=succ),
        )

    def sample(
        self,
        state: LoadedRoll.State[McStateT, AgentStateT],
    ) -> tuple[
        LoadedRoll.Trajectory[OutputT, McT],
        LoadedRoll.State[McStateT, AgentStateT],
    ]:
        """Collect a rollout while reusing endpoint inference only locally."""
        carry, trajectory = jax.lax.scan(
            self._advance,
            self._start(state),
            xs=None,
            length=self.seqlen,
            unroll=self._unroll,
        )
        if self.seq_axis != 0:
            trajectory = jax.tree.map(
                lambda x: jnp.moveaxis(x, 0, self.seq_axis),
                trajectory,
            )
        state = self.State(mc=carry.mc, agent=carry.agent_before_pre)
        return trajectory, state
