"""Induced Markov chain sampling utilities.

Implements multi-step trajectory collection by following a policy (agent) in an
environment, sampling from the Markov chain induced by the agent-environment
interaction.

Example:
    >>> import jax
    >>> from jaxtor.sampler import imc, mc
    >>>
    >>> # Assuming env follows mc.Env protocol and agent follows imc.Agent protocol
    >>> mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=100, env=env)
    >>> rollout_sampler = imc.InducedMarkovChain(agent=agent, mc=mc_sampler, seqlen=8)
    >>>
    >>> # Initialize state
    >>> key = jax.random.PRNGKey(0)
    >>> state = rollout_sampler.init(mc=mc_state, agent=agent_state)
    >>>
    >>> # Collect a trajectory
    >>> rollout, state = rollout_sampler.sample(state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
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
    """Induced Markov chain sampler for multi-step trajectory collection.

    Collects sequences of transitions by repeatedly applying a policy (agent) in
    an environment, sampling from the Markov chain induced by the agent-environment
    interaction.

    Attributes:
        agent: Agent instance following the Agent protocol for action selection.
        mc: Markov chain sampler following the MC protocol.
        seqlen: Number of steps to collect per trajectory.
        _unroll: Number of loop iterations to unroll in scan (default: 1).
    """

    agent: Agent
    mc: MC
    seqlen: int
    _unroll: int = 1

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

    def sample(
        self,
        state: InducedMarkovChain.State,
    ) -> tuple[MC.Transition, InducedMarkovChain.State]:
        """Collect a trajectory by rolling out the agent in the environment.

        Args:
            state: Current state of the induced Markov chain sampler.

        Returns:
            Stacked transitions with shape (seqlen, ...) and updated sampler state.
        """

        def step(carry, _):
            ag_state, mc_state = carry
            act, ag_state = self.agent.act(mc_state.last_obs, ag_state)
            transition, mc_state = self.mc.sample(act, mc_state)
            return (ag_state, mc_state), transition

        (agent_state, mc_state), transitions = jax.lax.scan(
            step, (state.agent, state.mc), length=self.seqlen, unroll=self._unroll
        )
        return transitions, state.replace(mc=mc_state, agent=agent_state)

    def init(self, mc: MC.State, agent: Agent.State) -> InducedMarkovChain.State:
        """Initialize the induced Markov chain sampler state.

        Args:
            mc: Pre-initialized MC sampler state.
            agent: Pre-initialized agent state.

        Returns:
            Initialized sampler state.
        """
        return InducedMarkovChain.State(mc=mc, agent=agent)

    def metrics(
        self, state: InducedMarkovChain.State
    ) -> tuple[InducedMarkovChain.Metrics, InducedMarkovChain.State]:
        """Compute metrics from the episode statistics queues and refresh them.

        Args:
            state: Current state containing episode statistics queues.

        Returns:
            Computed metrics and updated state with refreshed queues.
        """
        avg_eps_rew = jax.numpy.nanmean(state.mc.eps_rew_queue)
        avg_eps_len = jax.numpy.nanmean(state.mc.eps_len_queue)
        return (
            InducedMarkovChain.Metrics(
                avg_eps_rew=avg_eps_rew, avg_eps_len=avg_eps_len
            ),
            state.replace(mc=self.mc.refresh_queues(state.mc)),
        )
