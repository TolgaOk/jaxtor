"""Induced Markov chain sampling utilities.

Implements single-step trajectory collection by following a policy (agent) in an
environment, sampling from the Markov chain induced by the agent-environment
interaction.

Example:
    >>> import jax
    >>> from jaxtor.sampler import imc, mc, rollout
    >>>
    >>> # Assuming env follows mc.Env protocol and agent follows imc.Agent protocol
    >>> mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=100, env=env)
    >>> imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    >>>
    >>> # Get metrics
    >>> metrics, state = imc_step.metrics(state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax.numpy as jnp
from chex import dataclass


class MC(Protocol):
    class State(Protocol):
        last_obs: chex.Array
        last_done: chex.Numeric
        eps_rew_queue: chex.Array
        eps_len_queue: chex.Array

    class Transition(Protocol):
        obs: chex.Array
        act: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array
        nobs: chex.Array

    def init(self, key: chex.PRNGKey) -> MC.State: ...

    def sample(
        self,
        act: chex.Array,
        state: MC.State,
    ) -> tuple[Transition, MC.State]: ...

    def refresh_queues(self, state: MC.State) -> MC.State: ...


class Agent(Protocol):
    class State(Protocol): ...

    def act(
        self,
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

        mc: MC.State
        agent: Agent.State

    @dataclass
    class Metrics:
        """Episode statistics metrics.

        Attributes:
            avg_eps_rew: Average episode return from the statistics queue.
            avg_eps_len: Average episode length from the statistics queue.
        """

        avg_eps_rew: chex.Numeric
        avg_eps_len: chex.Numeric

    def init(self, mc: MC.State, agent: Agent.State) -> InducedMarkovChain.State:
        """Initialize the induced Markov chain sampler state.

        Args:
            mc: Pre-initialized MC sampler state.
            agent: Pre-initialized agent state.

        Returns:
            Initialized sampler state.
        """
        return InducedMarkovChain.State(mc=mc, agent=agent)

    def sample(
        self,
        state: InducedMarkovChain.State,
    ) -> tuple[MC.Transition, InducedMarkovChain.State]:
        """Execute one step of agent-MC interaction.

        Args:
            state: Current state of the induced Markov chain sampler.

        Returns:
            Single transition and updated sampler state.
        """
        act, agent_state = self.agent.act(state.mc.last_obs, state.agent)
        transition, mc_state = self.mc.sample(act, state.mc)
        return transition, state.replace(mc=mc_state, agent=agent_state)

    def metrics(
        self, state: InducedMarkovChain.State
    ) -> tuple[InducedMarkovChain.Metrics, InducedMarkovChain.State]:
        """Compute metrics from the episode statistics queues and refresh them.

        Args:
            state: Current state containing episode statistics queues.

        Returns:
            Computed metrics and updated state with refreshed queues.
        """
        avg_eps_rew = jnp.nanmean(state.mc.eps_rew_queue)
        avg_eps_len = jnp.nanmean(state.mc.eps_len_queue)
        return (
            InducedMarkovChain.Metrics(
                avg_eps_rew=avg_eps_rew, avg_eps_len=avg_eps_len
            ),
            state.replace(mc=self.mc.refresh_queues(state.mc)),
        )
