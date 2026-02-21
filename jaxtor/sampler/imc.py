"""Induced Markov Chain sampling.

Wires agent action selection to environment stepping, creating the Markov chain
induced by the agent-environment interaction.

Example:
    >>> mc = Mc(max_episode_len=100, queue_size=10, env=env)
    >>> imc = Imc(agent=agent, mc=mc)
    >>> state = Imc.State(mc=mc.init(key, env_state), agent=agent_state)
    >>> transition, state = imc.sample(state)
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import chex
from chex import dataclass

Transition = TypeVar("Transition")


class MC(Protocol[Transition]):
    class State(Protocol):
        last_obs: chex.Array

    def sample(
        self, act: chex.Array, state: MC.State
    ) -> tuple[Transition, MC.State]: ...


class Agent(Protocol):
    class State(Protocol): ...

    def act(
        self,
        obs: chex.Array,
        state: Agent.State,
    ) -> tuple[chex.Array, Agent.State]: ...


@dataclass
class Imc:
    """Induced Markov Chain - single-step agent-MC interaction.

    Wires: obs -> agent.act -> action -> mc.sample -> transition

    Attributes:
        agent: Agent following the Agent protocol.
        mc: Markov chain sampler following the MC protocol.
    """

    agent: Agent
    mc: MC

    @dataclass
    class State:
        """State of the induced Markov chain.

        Attributes:
            mc: Underlying Markov chain state.
            agent: Agent state.
        """

        mc: MC.State
        agent: Agent.State

    def init(self, mc: MC.State, agent: Agent.State) -> Imc.State:
        """Initialize the induced Markov chain state.

        Args:
            mc: Pre-initialized Markov chain state.
            agent: Pre-initialized agent state.

        Returns:
            Initialized Imc state.
        """
        return self.State(mc=mc, agent=agent)

    def sample(
        self,
        state: Imc.State,
    ) -> tuple[Transition, Imc.State]:
        """Execute one step of agent-MC interaction.

        Args:
            state: Current Imc state.

        Returns:
            Transition and updated state.
        """
        act, agent_state = self.agent.act(state.mc.last_obs, state.agent)
        transition, mc_state = self.mc.sample(act, state.mc)
        return transition, state.replace(mc=mc_state, agent=agent_state)  # type: ignore[unresolved-attribute]
