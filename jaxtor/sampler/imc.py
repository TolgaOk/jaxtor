"""Induced Markov chain sampling utilities.

Wires agent action selection to environment stepping, creating the Markov chain
induced by the agent-environment interaction.

Classes:
    InducedMarkovChain: Single-step agent-MC wiring.

Example:
    >>> mc_sampler = MarkovChain(max_episode_len=100, queue_size=10, env=env)
    >>> imc = InducedMarkovChain(agent=agent, mc=mc_sampler)
    >>> env_state = env.init(key)
    >>> mc_state = mc_sampler.init(key, env_state)
    >>> state = imc.init(key, mc_state, agent_state)
    >>> transition, state = imc.sample(state)

    >>> vec_mc = VecMC(mc=mc_sampler, n_env=4)
    >>> imc = InducedMarkovChain(agent=batched_agent, mc=vec_mc)
    >>> mc_state = vec_mc.init(key, env_state)
    >>> state = imc.init(key, mc_state, agent_state)
    >>> transition, state = imc.sample(state)
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import jax.random as jrd
import chex
from chex import dataclass

EnvState = TypeVar("EnvState")
Transition = TypeVar("Transition")


class MC(Protocol[EnvState, Transition]):
    class State(Protocol):
        last_obs: chex.Array

    def init(self, key: chex.PRNGKey, env: EnvState) -> MC.State: ...

    def sample(
        self, act: chex.Array, state: MC.State
    ) -> tuple[Transition, MC.State]: ...


class Agent(Protocol):
    class State(Protocol): ...

    def act(
        self,
        key: chex.PRNGKey,
        obs: chex.Array,
        state: Agent.State,
    ) -> tuple[chex.Array, Agent.State]: ...


@dataclass
class InducedMarkovChain:
    """Single-step agent-MC interaction.

    Receives agent and mc as dependencies (doesn't own them).
    Wires: obs -> agent.act -> action -> mc.sample -> transition

    Implements the IMC protocol from rollout module, allowing it to be
    used with Rollout for N-step trajectory collection.

    Attributes:
        agent: Agent instance following the Agent protocol for action selection.
        mc: Markov chain sampler following the MC protocol.
    """

    agent: Agent
    mc: MC

    @dataclass
    class State:
        """State of the induced Markov chain sampler.

        Attributes:
            mc: Underlying Markov chain sampler state.
            agent: Agent state for action selection.
        """

        key: chex.PRNGKey
        mc: MC.State
        agent: Agent.State

    def init(
        self, key: chex.PRNGKey, mc: MC.State, agent: Agent.State
    ) -> InducedMarkovChain.State:
        """Initialize the induced Markov chain sampler state.

        Args:
            mc: Pre-initialized MC sampler state.
            agent: Pre-initialized agent state.

        Returns:
            Initialized sampler state.
        """
        return InducedMarkovChain.State(key=key, mc=mc, agent=agent)

    def sample(
        self,
        state: InducedMarkovChain.State,
    ) -> tuple[Transition, InducedMarkovChain.State]:
        """Execute one step of agent-MC interaction.

        Args:
            state: Current state of the induced Markov chain sampler.

        Returns:
            Single transition and updated sampler state.
        """
        key, act_key = jrd.split(state.key, 2)
        act, agent_state = self.agent.act(act_key, state.mc.last_obs, state.agent)
        transition, mc_state = self.mc.sample(act, state.mc)
        return transition, state.replace(key=key, mc=mc_state, agent=agent_state)  # type: ignore[unresolved-attribute]
