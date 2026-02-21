"""Markov chain sampling utilities.

Provides environment wrappers for transition collection with episode statistics.

Classes:
    Mc: Single-environment sampler with episode tracking.
    VecMc: Vectorized sampler for multiple parallel environments.

Example:
    >>> mc_sampler = Mc(max_episode_len=100, queue_size=10, env=env)
    >>> env_state = env.init(key)
    >>> state = mc_sampler.init(key, env_state)
    >>> transition, state = mc_sampler.sample(action, state)

    >>> vec_mc = VecMc(mc=mc_sampler)
    >>> keys = jax.random.split(key, 4)
    >>> state = vec_mc.init(keys, env_state)
    >>> transition, state = vec_mc.sample(batched_actions, state)
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass

EnvState = TypeVar("EnvState")


class Env(Protocol[EnvState]):
    class Step(Protocol):
        nobs: chex.Array
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    def reset(
        self, key: chex.PRNGKey, env: EnvState
    ) -> tuple[chex.Array, EnvState]: ...

    def step(
        self, key: chex.PRNGKey, act: chex.Array, env: EnvState
    ) -> tuple[Step, EnvState]: ...


@dataclass
class Mc(Generic[EnvState]):
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
        env: EnvState
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

    @dataclass
    class Metrics:
        avg_eps_rew: chex.Numeric
        avg_eps_len: chex.Numeric

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

        chex.assert_rank([result.rew, result.term, result.trun], 0)
        chex.assert_equal_shape([state.last_obs, result.nobs])

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
        reset_obs, reset_env = self.env.reset(reset_key, env_state)

        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            # If done: reset and update statistics
            state.replace(  # type: ignore[reportAttributeAccessIssue]
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
            state.replace(  # type: ignore[reportAttributeAccessIssue]
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

    def _refresh_queues(self, state: State) -> State:
        """Reset the episode statistics queues.

        Args:
            state: Current state of the sampler.

        Returns:
            Updated state with cleared queue statistics.
        """
        return state.replace(  # type: ignore[reportAttributeAccessIssue]
            eps_rew_queue=jnp.full_like(state.eps_rew_queue, jnp.nan),
            eps_len_queue=jnp.full_like(state.eps_len_queue, jnp.nan),
        )

    def metrics(self, state: State) -> tuple[Metrics, State]:
        """Compute metrics from episode statistics queues and refresh them.

        Args:
            state: Current state containing episode statistics queues.

        Returns:
            Computed metrics and updated state with refreshed queues.
        """
        avg_eps_rew = jnp.nanmean(state.eps_rew_queue)
        avg_eps_len = jnp.nanmean(state.eps_len_queue)
        return (
            self.Metrics(avg_eps_rew=avg_eps_rew, avg_eps_len=avg_eps_len),
            self._refresh_queues(state),
        )

    def init(self, key: chex.PRNGKey, env: EnvState) -> State:
        """Initialize the state of the Markov chain sampler.

        Args:
            key: Random key for initialization.
            env: Pre-initialized environment state.

        Returns:
            Initialized sampler state.
        """
        key, reset_key = jrd.split(key, 2)
        last_obs, env_state = self.env.reset(reset_key, env)

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


@dataclass
class VecMc(Generic[EnvState]):
    """Vectorized Markov chain sampler.

    Wraps Mc with internal vmap for parallel environment sampling.
    Plugs directly into Imc and Roll without external vmap.

    Attributes:
        mc: Single-environment Mc sampler.
    """

    mc: Mc

    def init(self, keys: chex.Array, env: EnvState) -> Mc.State:
        """Initialize parallel sampler states.

        Args:
            keys: Batch of random keys with shape (n_env, ...).
            env: Pre-initialized environment state (broadcast to all envs).

        Returns:
            Batched Mc.State with leading dimension n_env.
        """
        return jax.vmap(self.mc.init, in_axes=(0, None))(keys, env)

    def sample(
        self, act: chex.Array, state: Mc.State
    ) -> tuple[Mc.Transition, Mc.State]:
        """Sample transitions from all environments in parallel.

        Args:
            act: Batched actions with shape (n_env, ...).
            state: Batched Mc.State.

        Returns:
            Batched transitions and updated batched state.
        """
        chex.assert_equal_shape_prefix([act, state.key], 1)
        return jax.vmap(self.mc.sample)(act, state)

    def metrics(self, state: Mc.State) -> tuple[Mc.Metrics, Mc.State]:
        """Aggregate metrics across all environments and refresh queues.

        Args:
            state: Batched Mc.State.

        Returns:
            Aggregated metrics (scalar) and updated batched state.
        """
        per_env, state = jax.vmap(self.mc.metrics)(state)
        return (
            Mc.Metrics(
                avg_eps_rew=jnp.nanmean(per_env.avg_eps_rew),
                avg_eps_len=jnp.nanmean(per_env.avg_eps_len),
            ),
            state,
        )
