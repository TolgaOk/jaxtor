"""Jaxdp tabular environment utilities.

Provides an interface for jaxdp tabular MDP environments.


>>> import jax
>>> from jaxtor.env import tabular
>>> key = jax.random.PRNGKey(0)
>>> config = tabular.garnet.Config(state_size=50, action_size=10)
>>> env = tabular.garnet.make(config)
>>> state = env.init(key)

"""

from __future__ import annotations

from typing import Protocol
from dataclasses import asdict
import jax.numpy as jnp
import chex
from chex import dataclass
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
    episode_length: chex.Numeric


@dataclass
class Step:
    nobs: chex.Array
    rew: chex.Numeric
    term: chex.Numeric
    trun: chex.Numeric


class ConfigProtocol(Protocol):
    """Protocol for tabular MDP configurations."""

    def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP: ...


@dataclass
class TabularEnv:
    obs_space: TabularSpace
    act_space: TabularSpace
    config: ConfigProtocol

    def step(
        self, key: chex.PRNGKey, act: chex.Array, state: TabularState
    ) -> tuple[Step, TabularState]:
        """Step the tabular environment.

        Args:
            key: JAX random key.
            act: One-hot encoded action.
            state: Current tabular state.

        Returns:
            Step and next state.
        """
        (
            next_obs,
            reward,
            terminal,
            timeout,
            last_obs,
            new_eps_step,
        ) = async_sample_step(
            state.mdp,
            act,
            state.last_state,
            state.step,  # type: ignore[arg-type]
            state.episode_length,  # type: ignore[arg-type]
            key,
        )
        return (
            Step(
                nobs=next_obs,
                rew=reward,
                term=terminal,
                trun=timeout,
            ),
            state.replace(  # type: ignore[attr-defined]
                last_state=next_obs,
                step=new_eps_step,
            ),
        )

    def init(self, key: chex.PRNGKey) -> TabularState:
        """Initialize the tabular environment.

        Args:
            key: JAX random key for initialization.

        Returns:
            Initial tabular state.
        """
        mdp = self.config.init_mdp(key)
        initial_state = mdp.init_state(key)
        return TabularState(
            mdp=mdp,
            last_state=initial_state,
            step=jnp.array(0),
            episode_length=jnp.array(1000),
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
        """

        state_size: int = 50
        action_size: int = 10
        branch_size: int = 5
        min_reward: float = 0.0
        max_reward: float = 1.0

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a Garnet MDP from this config."""
            return garnet_mdp(**asdict(self), key=key)

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
        """

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

        def init_mdp(self, key: chex.PRNGKey) -> JaxdpMDP:
            """Initialize a GridWorld MDP from this config."""
            return grid_world(**asdict(self))

    @staticmethod
    def make(config: gridworld.Config) -> TabularEnv:
        """Create a GridWorld tabular environment.

        Args:
            config: GridWorld MDP configuration.

        Returns:
            TabularEnv instance.
        """
        temp_mdp = grid_world(**asdict(config))

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
