"""Tabular MDP environment.

Index-based interface for jaxdp tabular MDPs (no one-hot encoding).

Example:
    >>> import jax
    >>> from jaxtor.env import tabular
    >>> key = jax.random.key(0)
    >>> env = tabular.make(tabular.garnet.Config())
    >>> init_key, reset_key = jax.random.split(key)
    >>> state = env.init(init_key)
    >>> obs, state = env.reset(reset_key, state)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from jaxdp.mdp import MDP as JaxdpMdp
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp as jaxdp_graph_mdp
from jaxdp.mdp.grid_world import grid_world


@dataclass
class Mdp:
    """Array-only tabular Markov decision process.

    Attributes:
        transition: Transition probabilities with shape ``(A, S_next, S)``.
        reward: Transition rewards with shape ``(A, S, S_next)``.
        initial: Initial-state distribution with shape ``(S,)``.
        terminal: Terminal-state indicators with shape ``(S,)``.
    """

    transition: jax.Array
    reward: jax.Array
    initial: jax.Array
    terminal: jax.Array

    @property
    def state_size(self) -> int:
        """Number of states."""
        return self.transition.shape[-1]

    @property
    def action_size(self) -> int:
        """Number of actions."""
        return self.transition.shape[-3]


def _adapt(mdp: JaxdpMdp) -> Mdp:
    """Copy a generated jaxdp MDP into Jaxtor's array-only representation."""
    return Mdp(
        transition=jnp.asarray(mdp.transition),
        reward=jnp.asarray(mdp.reward),
        initial=jnp.asarray(mdp.initial),
        terminal=jnp.asarray(mdp.terminal),
    )


def _sample_transition(
    key: chex.PRNGKey,
    mdp: Mdp,
    s: jax.Array,
    a: chex.Numeric,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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


class MdpConfig(Protocol):
    """Tabular MDP configuration consumed by ``TabularEnv``."""

    @property
    def max_eps_len(self) -> int: ...
    def init_mdp(self, key: chex.PRNGKey) -> Mdp: ...


@dataclass
class TabularEnv:
    """Index-based tabular MDP environment.

    Required protocols::

        config.max_eps_len: int
        config.init_mdp(key) -> mdp

    Attributes:
        config: MDP configuration used to initialize dynamics and truncation.

    Public dataclasses:
        State: MDP arrays, current state index, and episode step.
        Step: One environment transition.

    Public methods:
        init: Initialize an MDP and environment state.
        reset: Sample an initial state.
        step: Advance the MDP once.
        obs: Read the current state index.
    """

    @dataclass
    class State:
        """Environment state.

        Attributes:
            mdp: Underlying jaxdp MDP instance.
            s: Current state index.
            step: Current step within the episode.
        """

        mdp: Mdp
        s: jax.Array
        step: jax.Array

    @dataclass
    class Step:
        """Single-step transition result.

        Attributes:
            nobs: Next observation (state index).
            rew: Reward.
            term: Natural termination flag.
            trun: Truncation flag.
        """

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    config: MdpConfig

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
        trun = state.step >= self.config.max_eps_len - 1
        new_state = replace(state, s=s_next, step=state.step + 1)
        return (
            TabularEnv.Step(
                nobs=jnp.asarray(s_next),
                rew=jnp.asarray(rew),
                term=jnp.asarray(term),
                trun=jnp.asarray(trun),
            ),
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
        )

    def obs(self, state: TabularEnv.State) -> jax.Array:
        """Get observation from state.

        Args:
            state: Current state.

        Returns:
            State index.
        """
        return state.s

    def reset(
        self, key: chex.PRNGKey, state: TabularEnv.State
    ) -> tuple[jax.Array, TabularEnv.State]:
        """Reset to a new episode.

        Args:
            key: Random key for sampling initial state.
            state: Current state.

        Returns:
            Initial observation and reset state.
        """
        s = jrd.choice(
            key,
            state.mdp.state_size,
            p=state.mdp.initial,
        )
        new_state = replace(state, s=s, step=jnp.array(0))
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
            max_eps_len: Maximum episode length before truncation.
        """

        state_size: int = 50
        action_size: int = 10
        branch_size: int = 5
        min_reward: float = 0.0
        max_reward: float = 1.0
        max_eps_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> Mdp:
            """Initialize a Garnet MDP from this config."""
            return _adapt(
                garnet_mdp(
                    state_size=self.state_size,
                    action_size=self.action_size,
                    branch_size=self.branch_size,
                    min_reward=self.min_reward,
                    max_reward=self.max_reward,
                    key=key,
                )
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
            max_eps_len: Maximum episode length before truncation.
        """

        max_eps_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> Mdp:
            """Initialize a Graph MDP from this config."""
            return _adapt(jaxdp_graph_mdp())

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
            board: Sequence of strings representing the 2D grid layout.
            p_slip: Probability of slipping to unintended action.
            max_eps_len: Maximum episode length before truncation.

        Example:
            >>> config = gridworld.Config(
            ...     board=(
            ...         "#####",
            ...         "#  @#",
            ...         "# #X#",
            ...         "#P  #",
            ...         "#####",
            ...     ),
            ...     p_slip=0.1,
            ... )
        """

        board: Sequence[str] = (
            "#######",
            "#     #",
            "#  #  #",
            "#P # @#",
            "#  #  #",
            "#     #",
            "#######",
        )
        p_slip: float = 0.0
        max_eps_len: int = 1000

        def init_mdp(self, key: chex.PRNGKey) -> Mdp:
            """Initialize a GridWorld MDP from this config."""
            return _adapt(grid_world(board=list(self.board), p_slip=self.p_slip))

    @staticmethod
    def make(config: gridworld.Config) -> TabularEnv:
        """Create a GridWorld tabular environment.

        Args:
            config: GridWorld MDP configuration.

        Returns:
            TabularEnv instance.
        """
        return TabularEnv(config=config)


_ENVS: dict[str, MdpConfig] = {
    "cliffworld": gridworld.Config(
        board=(
            "#########",
            "#     @X#",
            "#      X#",
            "#      X#",
            "#      X#",
            "#P     X#",
            "#########",
        ),
    ),
    "cliff-walking": gridworld.Config(
        board=(
            "##############",
            "#            #",
            "#            #",
            "#            #",
            "#PXXXXXXXXXX@#",
            "##############",
        ),
    ),
    "four-rooms": gridworld.Config(
        board=(
            "#############",
            "#     #    @#",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#     #     #",
            "## #### #####",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#P    #     #",
            "#############",
        ),
    ),
    "frozen-lake": gridworld.Config(
        board=(
            "######",
            "#P   #",
            "# X X#",
            "#   X#",
            "#X  @#",
            "######",
        ),
        p_slip=1 / 3,
    ),
    "mid-garnet": garnet.Config(state_size=50, action_size=10, branch_size=5),
    "graph": graph.Config(),
}


def make(name: str) -> TabularEnv:
    """Create a pre-defined tabular environment by name.

    For custom configurations, use the namespace make functions directly
    (e.g. ``tabular.garnet.make(config)``).

    Args:
        name: Environment name ("cliffworld", "cliff-walking", "four-rooms",
            "frozen-lake", "mid-garnet", "graph").

    Returns:
        TabularEnv instance.

    Raises:
        ValueError: If name is not recognized.

    Example:
        >>> env = make("cliffworld")
        >>> env = make("mid-garnet")
    """
    if name not in _ENVS:
        raise ValueError(f"Unknown env {name!r}, choose from {list(_ENVS)}")
    return TabularEnv(config=_ENVS[name])
