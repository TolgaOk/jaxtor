"""Gymnax environment adapter.

Wraps gymnax environments to conform to jaxtor's Env protocol (no auto-reset).

Example:
    >>> import jax
    >>> from jaxtor.env import gymnax
    >>> key = jax.random.PRNGKey(0)
    >>> env = gymnax.make("CartPole-v1")
    >>> state = env.init(key)
    >>> obs, state = env.reset(key, state)
"""

from __future__ import annotations

import gymnax as _gymnax
import jax.numpy as jnp
import chex
from chex import dataclass
from gymnax.environments.environment import Environment, EnvParams, EnvState


@dataclass
class GymnaxEnv:
    """Adapter for gymnax environments.

    Attributes:
        env: Gymnax environment instance.
        params: Environment parameters.
    """

    env: Environment
    params: EnvParams

    @dataclass
    class State:
        """Environment state.

        Attributes:
            env: Inner gymnax environment state.
        """

        env: EnvState

    @dataclass
    class Step:
        """Single-step transition result.

        Attributes:
            nobs: Next observation.
            rew: Reward.
            term: Natural termination flag.
            trun: Truncation flag.
        """

        nobs: chex.Array
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    def init(self, key: chex.PRNGKey) -> GymnaxEnv.State:
        """Initialize the environment state.

        Args:
            key: Random key.

        Returns:
            Initialized state.
        """
        _, env_state = self.env.reset(key, self.params)
        return self.State(env=env_state)

    def step(
        self, key: chex.PRNGKey, act: chex.Numeric, state: State
    ) -> tuple[Step, State]:
        """Step the environment without auto-reset.

        Args:
            key: Random key.
            act: Action.
            state: Current state.

        Returns:
            Step result and next state.
        """
        obs, env_state, rew, done, info = self.env.step_env(
            key, state.env, act, self.params
        )
        trun = env_state.time >= self.params.max_steps_in_episode
        term = jnp.logical_and(done, jnp.logical_not(trun))
        return (
            self.Step(nobs=obs, rew=rew, term=term, trun=trun),
            state.replace(env=env_state),
        )

    def reset(self, key: chex.PRNGKey, state: State) -> tuple[chex.Array, State]:
        """Reset to a new episode.

        Args:
            key: Random key.
            state: Current state (preserved structure, replaced contents).

        Returns:
            Initial observation and reset state.
        """
        obs, env_state = self.env.reset(key, self.params)
        return obs, state.replace(env=env_state)

    def obs(self, state: State) -> chex.Array:
        """Get observation from state.

        Args:
            state: Current state.

        Returns:
            Current observation.
        """
        return self.env.get_obs(state.env, self.params)


def make(name: str, **kwargs) -> GymnaxEnv:
    """Create a gymnax environment adapter.

    Args:
        name: Gymnax environment name (e.g. "CartPole-v1").
        **kwargs: Overrides for environment parameters.

    Returns:
        GymnaxEnv instance.
    """
    env, params = _gymnax.make(name)
    if kwargs:
        params = params.replace(**kwargs)
    return GymnaxEnv(env=env, params=params)
