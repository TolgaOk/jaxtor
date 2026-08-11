"""Agent-induced Markov-chain sampling.

``Imc`` caches the current decision, advances an open Markov chain
with its action, and exposes the Markov-chain transition with agent data for
its true successor. Agent-defined fields such as behavior log-probabilities
or values therefore flow through sampling without specialized IMC variants.
"""

from __future__ import annotations

from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class McTransition(Protocol):
    """Environment transition fields required by ``Imc``."""

    obs: jax.Array
    act: jax.Array
    rew: jax.Array
    term: jax.Array
    trun: jax.Array
    nobs: jax.Array


class MarkovChain[McT: McTransition, StateT](Protocol):
    """Open Markov-chain capability required by ``Imc``."""

    def observe(self, state: StateT) -> jax.Array: ...

    def sample(
        self,
        act: chex.Array,
        state: StateT,
    ) -> tuple[McT, StateT]: ...


class Decision(Protocol):
    """Minimum agent decision consumed by ``Imc``."""

    act: chex.Array


class Agent[DecT: Decision, StateT](Protocol):
    """Decision capability required by ``Imc``."""

    def decide(
        self,
        obs: jax.Array,
        state: StateT,
    ) -> tuple[DecT, StateT]: ...


@dataclass
class Imc[
    DecT: Decision,
    AgentStateT,
    McT: McTransition,
    McStateT,
]:
    """Join an agent to a Markov chain for single-step sampling.

    The cached decision contains the agent data consumed by the next
    Markov-chain transition. A sampled successor is always computed at ``mc.nobs``,
    including at terminal or truncated boundaries. The cached next decision
    uses the reset observation at those boundaries.

    Attributes:
        agent: Agent that computes decisions from observations.
        mc: Open Markov-chain sampler.

    Public dataclasses:
        State: Markov-chain state, agent state, and cached decision.
        Sample: One Markov-chain transition and its true successor data.

    Public methods:
        init: Compute the first decision from initialized child states.
        observe: Read the cached decision without advancing state.
        sample: Consume the cached action and prepare its successor.
        refresh: Recompute the decision after externally changing agent state.
    """

    agent: Agent[DecT, AgentStateT]
    mc: MarkovChain[McT, McStateT]

    @dataclass
    class State[DecDataT: Decision, AgentDataT, McDataT]:
        """Dynamic state threaded by ``Imc``.

        Attributes:
            mc: State of the open Markov chain.
            agent: State used to prepare decisions.
            dec: Cached agent decision for the current observation.
        """

        mc: McDataT
        agent: AgentDataT
        dec: DecDataT

    @dataclass
    class Sample[McDataT: McTransition, DecDataT: Decision]:
        """One Markov-chain transition and its true successor decision.

        Attributes:
            mc: Transition returned by the underlying Markov chain.
            succ: Decision computed at ``mc.nobs``.
        """

        mc: McDataT
        succ: DecDataT

    def init(
        self,
        mc: McStateT,
        agent: AgentStateT,
    ) -> Imc.State[DecT, AgentStateT, McStateT]:
        """Initialize the IMC and cache its first decision."""
        dec, agent = self.agent.decide(self.mc.observe(mc), agent)
        return self.State(mc=mc, agent=agent, dec=dec)

    def observe(
        self,
        state: Imc.State[DecT, AgentStateT, McStateT],
    ) -> DecT:
        """Read the cached decision without advancing state."""
        return state.dec

    @staticmethod
    def _select_boundary(
        boundary: chex.Array,
        reset: DecT,
        succ: DecT,
    ) -> DecT:
        """Select reset-decision leaves for boundary lanes."""

        def select(reset_leaf: chex.Array, succ_leaf: chex.Array) -> chex.Array:
            extra_dims = succ_leaf.ndim - boundary.ndim
            if extra_dims < 0:
                raise ValueError("decision leaves must include the boundary batch axes")
            mask = jnp.reshape(boundary, (*boundary.shape, *(1,) * extra_dims))
            return jnp.where(mask, reset_leaf, succ_leaf)

        return cast(DecT, jax.tree.map(select, reset, succ))

    def _next(
        self,
        boundary: chex.Array,
        mc: McStateT,
        succ: DecT,
        agent: AgentStateT,
    ) -> tuple[DecT, AgentStateT]:
        """Reuse a successor normally and prepare reset decisions at boundaries."""

        def reset(_: None) -> tuple[DecT, AgentStateT]:
            reset_dec, reset_agent = self.agent.decide(
                self.mc.observe(mc),
                agent,
            )
            return self._select_boundary(boundary, reset_dec, succ), reset_agent

        def continue_(_: None) -> tuple[DecT, AgentStateT]:
            return succ, agent

        return jax.lax.cond(jnp.any(boundary), reset, continue_, operand=None)

    def sample(
        self,
        state: Imc.State[DecT, AgentStateT, McStateT],
    ) -> tuple[
        Imc.Sample[McT, DecT],
        Imc.State[DecT, AgentStateT, McStateT],
    ]:
        """Advance the Markov chain and compute its successor decision."""
        transition, mc = self.mc.sample(state.dec.act, state.mc)
        succ, agent = self.agent.decide(transition.nobs, state.agent)
        boundary = jnp.logical_or(transition.term, transition.trun)
        dec, agent = self._next(boundary, mc, succ, agent)
        return self.Sample(mc=transition, succ=succ), self.State(
            mc=mc,
            agent=agent,
            dec=dec,
        )

    def refresh(
        self,
        state: Imc.State[DecT, AgentStateT, McStateT],
    ) -> Imc.State[DecT, AgentStateT, McStateT]:
        """Recompute the cached decision after externally changing agent state."""
        dec, agent = self.agent.decide(self.mc.observe(state.mc), state.agent)
        return self.State(mc=state.mc, agent=agent, dec=dec)
