"""Markovian step sampling utilities.

Implements functions for collecting individual transitions from environments
and managing episode statistics.

Example:
    >>> import jax
    >>> from jaxtor.sampler import mc
    >>>
    >>> # Assuming env follows mc.Env protocol
    >>> sampler = mc.MarkovChain(max_episode_len=1000, queue_size=100, env=env)
    >>>
    >>> # Initialize sampler state
    >>> key = jax.random.PRNGKey(0)
    >>> state = sampler.init(key)
    >>>
    >>> # Sample a transition
    >>> key, act_key = jax.random.split(key)
    >>> act = jax.random.uniform(act_key, (10,))
    >>> transition, state = sampler.sample(act, state)
"""

from __future__ import annotations

from typing import Protocol
import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass


class Env(Protocol):
    class Step(Protocol):
        nobs: chex.Array
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    def init(self, key: chex.PRNGKey) -> tuple[chex.Array, chex.PyTreeDef]: ...  # type: ignore[reportInvalidTypeForm]

    def step(
        self, key: chex.PRNGKey, act: chex.Array, env_state: chex.PyTreeDef
    ) -> tuple[Env.Step, chex.PyTreeDef]: ...  # type: ignore[reportInvalidTypeForm]


@dataclass
class MarkovChain:
    """Markov chain sampler for collecting transitions from environments.

    Provides a uniform interface for interacting with environments and tracking
    episode statistics through rolling queues.

    Attributes:
        max_episode_len: Maximum length of an episode before truncation.
        queue_size: Size of rolling queues for episode statistics.
        env: Environment instance following the Env protocol.
    """

    max_episode_len: int
    queue_size: int
    env: Env

    @dataclass
    class State:
        """State of the Markov chain sampler.

        Attributes:
            key: Random key for sampling.
            env: Environment state.
            last_obs: Last observation from environment.
            last_done: Whether last transition was terminal.
            eps_idx: Current step index in episode.
            eps_rew: Cumulative reward in current episode.
            eps_rew_queue: Rolling queue of episode returns.
            eps_len_queue: Rolling queue of episode lengths.
        """

        key: chex.PRNGKey
        env: chex.PyTreeDef  # type: ignore[reportInvalidTypeForm]
        last_obs: chex.Array
        last_done: chex.Numeric
        eps_idx: chex.Numeric
        eps_rew: chex.Numeric
        eps_rew_queue: chex.Array
        eps_len_queue: chex.Array

    @dataclass
    class Transition:
        """Environment transition sample.

        Attributes:
            obs: Current observation.
            act: Action taken.
            rew: Reward received.
            term: Terminal flag (episode ended naturally).
            trun: Truncated flag (episode ended due to time limit).
            nobs: Next observation.
        """

        obs: chex.Array
        act: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array
        nobs: chex.Array

    def sample(self, act: chex.Array, state: State) -> tuple[Transition, State]:
        """Sample a transition from the environment.

        Args:
            act: Action to take in the environment.
            state: Current state of the sampler.

        Returns:
            Sampled transition and updated sampler state.
        """
        key, step_key, reset_key = jrd.split(state.key, 3)
        result, env_state = self.env.step(step_key, act, state.env)
        trun = jnp.logical_or(result.trun, state.eps_idx == self.max_episode_len - 1)

        transition = self.Transition(
            obs=state.last_obs,
            act=act,
            rew=result.rew,  # type: ignore[reportArgumentType]
            term=result.term,  # type: ignore[reportArgumentType]
            trun=trun,
            nobs=result.nobs,
        )

        done = jnp.logical_or(transition.term, transition.trun)

        # Compute reset values (used only if done)
        reset_obs, reset_env = self.env.init(reset_key)

        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            # If done: reset and update statistics
            state.replace(
                key=key,
                eps_idx=state.eps_idx * 0,
                eps_rew=state.eps_rew * 0,
                eps_rew_queue=(
                    jnp.roll(state.eps_rew_queue, shift=1)
                    .at[0]
                    .set(state.eps_rew + transition.rew)
                ),
                eps_len_queue=(
                    jnp.roll(state.eps_len_queue, shift=1).at[0].set(state.eps_idx + 1)
                ),
                last_done=done,
                last_obs=reset_obs,
                env=reset_env,
            ),
            # If not done: continue episode
            state.replace(
                key=key,
                eps_idx=state.eps_idx + 1,
                eps_rew=state.eps_rew + transition.rew,
                eps_rew_queue=state.eps_rew_queue,
                eps_len_queue=state.eps_len_queue,
                env=env_state,
                last_obs=result.nobs,
                last_done=done,
            ),
        )
        return transition, state

    def refresh_queues(self, state: State) -> State:
        """Reset the episode statistics queues.

        Args:
            state: Current state of the sampler.

        Returns:
            Updated state with cleared queue statistics.
        """
        return state.replace(
            eps_rew_queue=jnp.full_like(state.eps_rew_queue, jnp.nan),
            eps_len_queue=jnp.full_like(state.eps_len_queue, jnp.nan),
        )

    def init(self, key: chex.PRNGKey) -> State:
        """Initialize the state of the Markov chain sampler.

        Args:
            key: Random key for initialization.

        Returns:
            Initialized sampler state.
        """
        key, reset_key = jrd.split(key, 2)
        last_obs, env_state = self.env.init(reset_key)

        return self.State(
            key=key,
            env=env_state,
            last_obs=last_obs,
            last_done=True,
            eps_idx=0,
            eps_rew=0.0,
            eps_rew_queue=jnp.full(self.queue_size, jnp.nan),
            eps_len_queue=jnp.full(self.queue_size, jnp.nan),
        )
