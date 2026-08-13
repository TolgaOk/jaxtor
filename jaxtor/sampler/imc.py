"""Minimal agent-induced Markov-chain sampling.

``Imc`` asks an agent for one action and returns the transition produced by an
open Markov chain. Rich agent predictions belong to ``LoadedRoll``.
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import jax
from chex import dataclass


class MarkovChain[McT, StateT](Protocol):
    """Open Markov-chain capability required by ``Imc``."""

    def observe(self, state: StateT) -> jax.Array: ...

    def sample(
        self,
        act: jax.Array,
        state: StateT,
    ) -> tuple[McT, StateT]: ...


class Agent[StateT](Protocol):
    """Action-selection capability required by ``Imc``."""

    def act(
        self,
        obs: jax.Array,
        state: StateT,
    ) -> tuple[jax.Array, StateT]: ...


AgentT = TypeVar("AgentT", covariant=True)
MarkovChainT = TypeVar("MarkovChainT", covariant=True)


@dataclass
class Imc(Generic[AgentT, MarkovChainT]):
    """Join an action-selecting agent to a Markov chain for one step.

    Attributes:
        agent: Agent that selects an action from the current observation.
        mc: Open Markov-chain sampler advanced by that action.

    Public dataclasses:
        State: Markov-chain and agent states.

    Public methods:
        init: Combine initialized child states.
        sample: Select one action and advance the Markov chain once.
    """

    agent: AgentT
    mc: MarkovChainT

    @dataclass
    class State[McDataT, AgentDataT]:
        """Dynamic state threaded through ``Imc``.

        Attributes:
            mc: State of the open Markov chain.
            agent: State of the action-selecting agent.
        """

        mc: McDataT
        agent: AgentDataT

    def init[AgentStateT, McT, McStateT](
        self: Imc[Agent[AgentStateT], MarkovChain[McT, McStateT]],
        mc: McStateT,
        agent: AgentStateT,
    ) -> Imc.State[McStateT, AgentStateT]:
        """Combine initialized Markov-chain and agent states."""
        return self.State(mc=mc, agent=agent)

    def sample[AgentStateT, McT, McStateT](
        self: Imc[Agent[AgentStateT], MarkovChain[McT, McStateT]],
        state: Imc.State[McStateT, AgentStateT],
    ) -> tuple[
        McT,
        Imc.State[McStateT, AgentStateT],
    ]:
        """Select one action and advance the Markov chain once."""
        act, agent = self.agent.act(self.mc.observe(state.mc), state.agent)
        transition, mc = self.mc.sample(act, state.mc)
        return transition, self.State(
            mc=mc,
            agent=agent,
        )
