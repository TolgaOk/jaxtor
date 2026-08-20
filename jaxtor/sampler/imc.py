"""Minimal agent-induced Markov-chain sampling.

``Imc`` joins action selection and one Markov-chain step::

    imc = Imc(agent=agent, mc=mc)
    state = imc.init(mc_state, agent_state)
    transition, state = imc.sample(state)

The returned sample is produced by the Markov chain. Agent predictions needed
for learning are replayed by a separate inference component.
"""

from __future__ import annotations

from typing import Protocol

from chex import dataclass


class MarkovChain[Obs, Act, Sample, S](Protocol):
    """Open Markov-chain capability required by ``Imc``."""

    def observe(self, state: S) -> Obs: ...
    def sample(self, act: Act, state: S) -> tuple[Sample, S]: ...


class Agent[Obs, Act, S](Protocol):
    """Action-selection capability required by ``Imc``."""

    def act(self, obs: Obs, state: S) -> tuple[Act, S]: ...


@dataclass
class Imc[Obs, Act, Sample, AgentS, McS]:
    """Join an action-selecting agent to a Markov chain for one step.

    Required protocols::

        agent.act(observation, agent_state) -> (action, agent_state)
        mc.observe(mc_state) -> observation
        mc.sample(action, mc_state) -> (transition, mc_state)

    Attributes:
        agent: Agent that selects an action from the current observation.
        mc: Open Markov-chain sampler advanced by that action.

    Public dataclasses:
        State: Markov-chain and agent states.

    Public methods:
        init: Combine initialized child states.
        sample: Select one action and advance the Markov chain once.
    """

    agent: Agent[Obs, Act, AgentS]
    mc: MarkovChain[Obs, Act, Sample, McS]

    @dataclass
    class State[McData, AgentData]:
        """Dynamic state threaded through ``Imc``.

        Attributes:
            mc: State of the open Markov chain.
            agent: State of the action-selecting agent.
        """

        mc: McData
        agent: AgentData

    def init(self, mc: McS, agent: AgentS) -> Imc.State[McS, AgentS]:
        """Combine initialized Markov-chain and agent states."""
        return self.State(mc=mc, agent=agent)

    def sample(
        self,
        state: Imc.State[McS, AgentS],
    ) -> tuple[Sample, Imc.State[McS, AgentS]]:
        """Select one action and advance the Markov chain once."""
        act, agent = self.agent.act(self.mc.observe(state.mc), state.agent)
        transition, mc = self.mc.sample(act, state.mc)
        return transition, self.State(
            mc=mc,
            agent=agent,
        )
