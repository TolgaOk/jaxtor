"""Induced Markov Chain sampling with behavior log-probability tracking.

Like Imc, wires agent action selection to environment stepping. Additionally
tracks log π_μ(a|s) — the log-probability under the behavior policy at
collection time — useful for importance-weighted methods like PPO.

Example:
    >>> mc = Mc(max_episode_len=100, queue_size=10, env=env)
    >>> mu_imc = MuImc(agent=agent, mc=mc)
    >>> state = MuImc.State(mc=mc.init(key, env_state), agent=agent_state)
    >>> transition, state = mu_imc.sample(state)
    >>> transition.log_mu  # log π_μ(a|s) at collection time
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import chex
from chex import dataclass

Transition = TypeVar("Transition")
Act = TypeVar("Act", bound=chex.Array)
LogProb = TypeVar("LogProb", bound=chex.Numeric)


class MC(Protocol[Transition]):
    class State(Protocol):
        last_obs: chex.Array

    def sample(self, act: Act, state: MC.State) -> tuple[Transition, MC.State]: ...


class Agent(Protocol[Act, LogProb]):
    class State(Protocol): ...

    def act(
        self,
        obs: chex.Array,
        state: Agent.State,
    ) -> tuple[Act, LogProb, Agent.State]: ...


@dataclass
class MuImc:
    """Induced Markov Chain with behavior log-probability.

    Wires: obs -> agent.act -> (action, log_mu) -> mc.sample -> transition

    The agent protocol returns (action, log_mu, state) instead of
    (action, state), and the resulting Transition includes the log_mu field.

    Attributes:
        agent: Agent following the Agent protocol (act returns log_mu).
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

    @dataclass
    class Transition:
        """Transition with behavior log-probability.

        Attributes:
            obs: Current observation.
            act: Action taken.
            rew: Reward received.
            term: Terminal flag.
            trun: Truncated flag.
            nobs: Next observation.
            log_mu: Log-probability under behavior policy.
        """

        obs: chex.Array
        act: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array
        nobs: chex.Array
        log_mu: chex.Array

    def init(self, mc: MC.State, agent: Agent.State) -> MuImc.State:
        """Initialize the induced Markov chain state.

        Args:
            mc: Pre-initialized Markov chain state.
            agent: Pre-initialized agent state.

        Returns:
            Initialized MuImc state.
        """
        return self.State(mc=mc, agent=agent)

    def sample(
        self,
        state: MuImc.State,
    ) -> tuple[MuImc.Transition, MuImc.State]:
        """Execute one step of agent-MC interaction.

        Args:
            state: Current MuImc state.

        Returns:
            Transition (with log_mu) and updated state.
        """
        act, log_mu, agent_state = self.agent.act(state.mc.last_obs, state.agent)
        mc_trans, mc_state = self.mc.sample(act, state.mc)
        transition = self.Transition(
            obs=mc_trans.obs,
            act=mc_trans.act,
            rew=mc_trans.rew,
            term=mc_trans.term,
            trun=mc_trans.trun,
            nobs=mc_trans.nobs,
            log_mu=log_mu,
        )
        return transition, state.replace(mc=mc_state, agent=agent_state)  # type: ignore[unresolved-attribute]
