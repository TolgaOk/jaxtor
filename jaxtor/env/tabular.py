"""Jaxdp tabular environment utilities.

Provides an interface for jaxdp tabular MDP environments.


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
import chex
from chex import dataclass
import jax
from jaxdp.mdp import MDP as JaxdpMDP
from jaxdp.mdp.garnet import garnet_mdp
from jaxdp.mdp.simple_graph import graph_mdp as jaxdp_graph_mdp
from jaxdp.mdp.grid_world import grid_world
from jaxdp import async_sample_step


@dataclass
class TabularSpace:
    shape: tuple[int]
    low: chex.Array
    high: chex.Array


@dataclass
class TabularState:
    mdp: JaxdpMDP
    last_state: chex.Array
    step: chex.Numeric
    max_episode_len: chex.Numeric


@dataclass
class Step:
    nobs: chex.Numeric
    rew: chex.Numeric
    term: chex.Numeric
    trun: chex.Numeric


class ConfigProtocol(Protocol):
    """Protocol for tabular MDP configurations."""

    max_episode_len: int

    def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP: ...


@dataclass
class TabularEnv:
    obs_space: TabularSpace
    act_space: TabularSpace
    config: ConfigProtocol

    def step(
        self, key: chex.PRNGKey, act: chex.Numeric, state: TabularState
    ) -> tuple[Step, TabularState]:
        """Step the tabular environment.

        Args:
            key: JAX random key.
            act: One-hot encoded action.
            state: Current tabular state.

        Returns:
            Step and next state.
        """
        chex.assert_rank(act, 0)
        (
            next_obs,
            reward,
            terminal,
            timeout,
            last_state,
            new_eps_step,
        ) = async_sample_step(
            state.mdp,
            jax.nn.one_hot(act, state.mdp.action_size),
            state.last_state,
            state.step,  # type: ignore[arg-type]
            state.max_episode_len,  # type: ignore[arg-type]
            key,
        )
        return (
            Step(
                nobs=jnp.argmax(next_obs),
                rew=reward,
                term=terminal,
                trun=timeout,
            ),
            state.replace(  # type: ignore[attr-defined]
                last_state=last_state,
                step=new_eps_step,
            ),
        )

    def init(self, key: chex.PRNGKey) -> TabularState:
        """Initialize the tabular environment state.

        Args:
            key: JAX random key for MDP initialization.

        Returns:
            Tabular state with initialized MDP.
        """
        mdp = self.config.init_mdp(key)
        return TabularState(
            mdp=mdp,
            last_state=jnp.full(mdp.state_size, jnp.nan),
            step=jnp.array(0),
            max_episode_len=jnp.array(self.config.max_episode_len),
        )

    def reset(
        self, key: chex.PRNGKey, state: TabularState
    ) -> tuple[chex.Numeric, TabularState]:
        """Reset the environment to start a new episode.

        Args:
            key: JAX random key for sampling initial state.
            state: Current tabular state.

        Returns:
            Initial observation and reset state.
        """
        initial_state = state.mdp.init_state(key)
        return (
            jnp.argmax(initial_state),
            state.replace(
                last_state=initial_state,
                step=jnp.array(0),
            ),
        )


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
        return TabularEnv(
            obs_space=TabularSpace(
                shape=(config.state_size,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            act_space=TabularSpace(
                shape=(config.action_size,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            config=config,
        )


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
        return TabularEnv(
            obs_space=TabularSpace(
                shape=(6,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            act_space=TabularSpace(
                shape=(6,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            config=config,
        )


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
        temp_mdp = grid_world(board=config.board, p_slip=config.p_slip)

        return TabularEnv(
            obs_space=TabularSpace(
                shape=(temp_mdp.state_size,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            act_space=TabularSpace(
                shape=(temp_mdp.action_size,),
                low=jnp.array(0.0),
                high=jnp.array(1.0),
            ),
            config=config,
        )
