"""Tabular MDP environment.

Index-based interface for jaxdp tabular MDPs (no one-hot encoding).

Example:
    >>> import jax
    >>> from jaxtor.env import tabular
    >>> key = jax.random.PRNGKey(0)
    >>> config = tabular.garnet.Config(state_size=50, action_size=10)
    >>> env = tabular.garnet.make(config)
    >>> init_key, reset_key = jax.random.split(key)
    >>> state = env.init(init_key)
    >>> obs, state = env.reset(reset_key, state)
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from jaxdp.mdp import MDP as JaxdpMDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp as jaxdp_graph_mdp
from jaxdp.mdp.grid_world import grid_world


def _sample_transition(
    key: chex.PRNGKey, mdp: JaxdpMDP, s: chex.Numeric, a: chex.Numeric
) -> tuple[chex.Numeric, chex.Numeric, chex.Numeric]:
    """Sample transition from MDP using indices.

    Args:
        key: Random key.
        mdp: MDP with transition[A, S', S] and reward[A, S, S'].
        s: Current state index.
        a: Action index.

    Returns:
        (next_state, reward, terminal) tuple.
    """
    probs = mdp.transition[a, :, s]
    s_next = jrd.choice(key, mdp.state_size, p=probs)
    rew = mdp.reward[a, s, s_next]
    term = mdp.terminal[s_next]
    return s_next, rew, term


class ConfigProtocol(Protocol):
    """Protocol for tabular MDP configurations."""

    max_episode_len: int

    def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP: ...


@dataclass
class TabularEnv:
    """Index-based tabular MDP environment.

    Attributes:
        config: MDP configuration following ConfigProtocol.
    """

    @dataclass
    class State:
        """Environment state.

        Attributes:
            mdp: Underlying jaxdp MDP instance.
            s: Current state index.
            step: Current step within the episode.
            max_episode_len: Maximum episode length before truncation.
        """

        mdp: JaxdpMDP
        s: chex.Numeric
        step: chex.Numeric
        max_episode_len: chex.Numeric

    @dataclass
    class Step:
        """Single-step transition result.

        Attributes:
            nobs: Next observation (state index).
            rew: Reward.
            term: Natural termination flag.
            trun: Truncation flag.
        """

        nobs: chex.Numeric
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    config: ConfigProtocol

    def step(
        self, key: chex.PRNGKey, act: chex.Numeric, state: State
    ) -> tuple[Step, State]:
        """Step the environment.

        Args:
            key: Random key.
            act: Action index.
            state: Current state.

        Returns:
            Step result and next state.
        """
        s_next, rew, term = _sample_transition(key, state.mdp, state.s, act)
        trun = state.step >= state.max_episode_len - 1
        new_state = state.replace(s=s_next, step=state.step + 1)  # type: ignore[attr-defined]
        return (
            TabularEnv.Step(nobs=s_next, rew=rew, term=term, trun=trun),
            new_state,
        )

    def init(self, key: chex.PRNGKey) -> TabularEnv.State:
        """Initialize the environment state.

        Args:
            key: Random key for MDP initialization.

        Returns:
            Initialized state.
        """
        mdp = self.config.init_mdp(key)
        return TabularEnv.State(
            mdp=mdp,
            s=jnp.array(-1),
            step=jnp.array(0),
            max_episode_len=jnp.array(self.config.max_episode_len),
        )

    def obs(self, state: TabularEnv.State) -> chex.Numeric:
        """Get observation from state.

        Args:
            state: Current state.

        Returns:
            State index.
        """
        return state.s

    def reset(
        self, key: chex.PRNGKey, state: TabularEnv.State
    ) -> tuple[chex.Numeric, TabularEnv.State]:
        """Reset to a new episode.

        Args:
            key: Random key for sampling initial state.
            state: Current state.

        Returns:
            Initial observation and reset state.
        """
        s = jrd.choice(key, state.mdp.state_size, p=state.mdp.initial)
        new_state = state.replace(s=s, step=jnp.array(0))  # type: ignore[attr-defined]
        return (s, new_state)


class garnet:
    """Garnet MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a Garnet MDP.

        Attributes:
            state_size: Number of states in the MDP.
            action_size: Number of actions available.
            branch_size: Number of successor states per state-action pair.
            min_reward: Minimum reward value.
            max_reward: Maximum reward value.
            max_episode_len: Maximum episode length before truncation.
        """

        state_size: int = 50
        action_size: int = 10
        branch_size: int = 5
        min_reward: float = 0.0
        max_reward: float = 1.0
        max_episode_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a Garnet MDP from this config."""
            return garnet_mdp(
                state_size=self.state_size,
                action_size=self.action_size,
                branch_size=self.branch_size,
                min_reward=self.min_reward,
                max_reward=self.max_reward,
                key=key,
            )

    @staticmethod
    def make(config: garnet.Config) -> TabularEnv:
        """Create a Garnet tabular environment.

        Args:
            config: Garnet MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)


class graph:
    """Graph MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a Graph MDP.

        The graph MDP from 'Fastest Convergence for Q-Learning' paper.
        This is a fixed 6-state graph with predefined edge structure.

        Attributes:
            max_episode_len: Maximum episode length before truncation.
        """

        max_episode_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a Graph MDP from this config."""
            return jaxdp_graph_mdp()

    @staticmethod
    def make(config: graph.Config) -> TabularEnv:
        """Create a Graph tabular environment.

        Args:
            config: Graph MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)


class gridworld:
    """GridWorld MDP namespace."""

    @dataclass
    class Config:
        """Configuration for creating a GridWorld MDP.

        Board characters:
            '#': Impassable wall
            'P': Initial agent position
            '@': Terminal/goal state (positive reward)
            '=': Absorbing state (positive reward)
            '+': Positive reward cell
            'X': Penalty cell
            ' ': Regular passable space

        Attributes:
            board: List of strings representing the 2D grid layout.
            p_slip: Probability of slipping to unintended action.
            max_episode_len: Maximum episode length before truncation.

        Example:
            >>> config = gridworld.Config(
            ...     board=[
            ...         "#####",
            ...         "#  @#",
            ...         "# #X#",
            ...         "#P  #",
            ...         "#####"
            ...     ],
            ...     p_slip=0.1
            ... )
        """

        board: list[str]
        p_slip: float = 0.0
        max_episode_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a GridWorld MDP from this config."""
            return grid_world(board=self.board, p_slip=self.p_slip)

    @staticmethod
    def make(config: gridworld.Config) -> TabularEnv:
        """Create a GridWorld tabular environment.

        Args:
            config: GridWorld MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)
