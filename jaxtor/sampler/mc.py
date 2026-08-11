"""Open Markov-chain sampling with episode lifecycle statistics.

``Mc`` turns an environment into a transition sampler. ``VecMc`` applies the
same interface to a batch of independent environment states.
"""

from __future__ import annotations

from typing import Protocol, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from chex import dataclass


class EnvStep(Protocol):
    """Environment output fields consumed by ``Mc``."""

    nobs: chex.Array
    rew: chex.Array
    term: chex.Array
    trun: chex.Array


class Env[StateT, StepT: EnvStep](Protocol):
    """Environment capability required by ``Mc``."""

    def reset(
        self,
        key: jax.Array,
        state: StateT,
    ) -> tuple[chex.Numeric, StateT]: ...

    def step(
        self,
        key: jax.Array,
        act: jax.Array,
        state: StateT,
    ) -> tuple[StepT, StateT]: ...


@dataclass
class Mc[EnvStateT, EnvStepT: EnvStep]:
    """Sample an open Markov chain and track completed episodes.

    ``Mc`` owns the episode limit and rolling metric queues. Its state keeps
    the current environment observation so :meth:`observe` does not inspect
    backend-specific environment state.

    Attributes:
        max_episode_len: Maximum number of transitions in one episode.
        queue_size: Number of completed episodes retained for metrics.
        env: Environment implementing ``reset`` and non-autoreset ``step``.

    Public dataclasses:
        State: Environment, RNG, current observation, and episode statistics.
        Transition: One aligned ``obs, act, rew, term, trun, nobs`` sample.
        Metrics: Mean return and length of queued completed episodes.

    Public methods:
        init: Initialize state from a backend-specific environment state.
        observe: Read the current observation without advancing state.
        sample: Advance once and autoreset at an episode boundary.
        metrics: Read aggregate episode metrics and clear their queues.
    """

    max_episode_len: int
    queue_size: int
    env: Env[EnvStateT, EnvStepT]

    @dataclass
    class State[EnvDataT]:
        """Dynamic state threaded through ``Mc``.

        Attributes:
            key: Random key used by environment steps and resets.
            env: Backend-specific environment state.
            last_obs: Observation from which the next action is taken.
            eps_idx: Number of transitions completed in the current episode.
            eps_rew: Return accumulated in the current episode.
            eps_rew_queue: Returns of recently completed episodes.
            eps_len_queue: Lengths of recently completed episodes.
        """

        key: jax.Array
        env: EnvDataT
        last_obs: jax.Array
        eps_idx: jax.Array
        eps_rew: jax.Array
        eps_rew_queue: jax.Array
        eps_len_queue: jax.Array

    @dataclass
    class Transition:
        """One environment transition.

        Attributes:
            obs: Observation from which the action was taken.
            act: Action supplied to the environment.
            rew: Scalar reward produced by the action.
            term: Whether the episode terminated naturally.
            trun: Whether the episode was truncated.
            nobs: True observation reached by the action.
        """

        obs: jax.Array
        act: jax.Array
        rew: jax.Array
        term: jax.Array
        trun: jax.Array
        nobs: jax.Array

    @dataclass
    class Metrics:
        """Aggregate statistics over queued completed episodes.

        Attributes:
            avg_eps_rew: Mean completed-episode return.
            avg_eps_len: Mean completed-episode length.
        """

        avg_eps_rew: jax.Array
        avg_eps_len: jax.Array

    @dataclass
    class _Advance[EnvDataT]:
        """Data required to finish one environment advance."""

        state: Mc.State[EnvDataT]
        env: EnvDataT
        transition: Mc.Transition
        key: jax.Array
        reset_key: jax.Array

    def __post_init__(self) -> None:
        """Validate static episode and queue configuration."""
        if self.max_episode_len < 1:
            raise ValueError("max_episode_len must be positive")
        if self.queue_size < 1:
            raise ValueError("queue_size must be positive")

    def init(
        self,
        key: jax.Array,
        env: EnvStateT,
    ) -> Mc.State[EnvStateT]:
        """Initialize the Markov chain from an environment state."""
        key, reset_key = jrd.split(key)
        obs, env = self.env.reset(reset_key, env)
        return self.State(
            key=key,
            env=env,
            last_obs=jnp.asarray(obs),
            eps_idx=jnp.array(0, dtype=jnp.int32),
            eps_rew=jnp.array(0.0),
            eps_rew_queue=jnp.full(self.queue_size, jnp.nan),
            eps_len_queue=jnp.full(self.queue_size, jnp.nan),
        )

    def observe(self, state: Mc.State[EnvStateT]) -> jax.Array:
        """Read the current observation without advancing state."""
        return state.last_obs

    def sample(
        self,
        act: chex.Array,
        state: Mc.State[EnvStateT],
    ) -> tuple[Mc.Transition, Mc.State[EnvStateT]]:
        """Advance once and reset only when the episode reaches a boundary."""
        transition, advance = self._advance(act, state)
        done = jnp.logical_or(transition.term, transition.trun)
        state = jax.lax.cond(
            done,
            self._reset_episode,
            self._continue_episode,
            advance,
        )
        return transition, state

    def metrics(
        self,
        state: Mc.State[EnvStateT],
    ) -> tuple[Mc.Metrics, Mc.State[EnvStateT]]:
        """Return queued episode metrics and clear both metric queues."""
        metrics = self.Metrics(
            avg_eps_rew=jnp.nanmean(state.eps_rew_queue),
            avg_eps_len=jnp.nanmean(state.eps_len_queue),
        )
        return metrics, self._clear_metric_queues(state)

    def _advance(
        self,
        act: chex.Array,
        state: Mc.State[EnvStateT],
    ) -> tuple[Mc.Transition, Mc._Advance[EnvStateT]]:
        """Step the environment and retain data needed for boundary handling."""
        key, step_key, reset_key = jrd.split(state.key, 3)
        act = jnp.asarray(act)
        result, env = self.env.step(step_key, act, state.env)
        nobs = jnp.asarray(result.nobs)
        rew = jnp.asarray(result.rew)
        term = jnp.asarray(result.term, dtype=jnp.bool_)
        env_trun = jnp.asarray(result.trun, dtype=jnp.bool_)

        chex.assert_rank([rew, term, env_trun], 0)
        chex.assert_equal_shape([state.last_obs, nobs])

        reached_limit = state.eps_idx + 1 >= self.max_episode_len
        transition = self.Transition(
            obs=state.last_obs,
            act=act,
            rew=rew,
            term=term,
            trun=jnp.logical_or(env_trun, reached_limit),
            nobs=nobs,
        )
        return transition, self._Advance(
            state=state,
            env=env,
            transition=transition,
            key=key,
            reset_key=reset_key,
        )

    def _reset_episode(
        self,
        advance: Mc._Advance[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Reset after a boundary and record the completed episode."""
        state = advance.state
        transition = advance.transition
        obs, env = self.env.reset(advance.reset_key, advance.env)
        return state.replace(  # type: ignore[reportAttributeAccessIssue]
            key=advance.key,
            env=env,
            last_obs=jnp.asarray(obs),
            eps_idx=jnp.zeros_like(state.eps_idx),
            eps_rew=jnp.zeros_like(state.eps_rew),
            eps_rew_queue=(
                jnp.roll(state.eps_rew_queue, shift=1)
                .at[0]
                .set(state.eps_rew + transition.rew)
            ),
            eps_len_queue=(
                jnp.roll(state.eps_len_queue, shift=1).at[0].set(state.eps_idx + 1)
            ),
        )

    @staticmethod
    def _continue_episode(
        advance: Mc._Advance[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Continue an unfinished episode from its true next observation."""
        state = advance.state
        transition = advance.transition
        return state.replace(  # type: ignore[reportAttributeAccessIssue]
            key=advance.key,
            env=advance.env,
            last_obs=transition.nobs,
            eps_idx=state.eps_idx + 1,
            eps_rew=state.eps_rew + transition.rew,
        )

    @staticmethod
    def _clear_metric_queues(
        state: Mc.State[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Clear completed-episode queues after their metrics are consumed."""
        return state.replace(  # type: ignore[reportAttributeAccessIssue]
            eps_rew_queue=jnp.full_like(state.eps_rew_queue, jnp.nan),
            eps_len_queue=jnp.full_like(state.eps_len_queue, jnp.nan),
        )


@dataclass
class VecMc[EnvStateT, EnvStepT: EnvStep]:
    """Vectorize one ``Mc`` over independent environment states.

    A normal batch step does not compute reset states. If any lane reaches a
    boundary, reset states are prepared once for the batch and selected only
    for boundary lanes.

    Attributes:
        mc: Scalar Markov-chain sampler shared by all lanes.

    Public methods:
        init: Vectorize ``Mc.init`` over random keys.
        observe: Read batched current observations.
        sample: Advance all lanes and handle mixed boundaries.
        metrics: Aggregate and clear per-lane episode metric queues.
    """

    mc: Mc[EnvStateT, EnvStepT]

    @dataclass
    class _Boundary[EnvDataT]:
        """Batched data required for conditional reset selection."""

        done: jax.Array
        advance: Mc._Advance[EnvDataT]
        continued: Mc.State[EnvDataT]

    def init(
        self,
        keys: jax.Array,
        env: EnvStateT,
    ) -> Mc.State[EnvStateT]:
        """Initialize one Markov-chain state per random key."""
        return jax.vmap(self.mc.init, in_axes=(0, None))(keys, env)

    def observe(self, state: Mc.State[EnvStateT]) -> jax.Array:
        """Read the batched current observations."""
        return jax.vmap(self.mc.observe)(state)

    def sample(
        self,
        act: chex.Array,
        state: Mc.State[EnvStateT],
    ) -> tuple[Mc.Transition, Mc.State[EnvStateT]]:
        """Advance every lane and reset only boundary-containing batches."""
        chex.assert_equal_shape_prefix([act, state.key], 1)
        transition, advance = jax.vmap(self.mc._advance)(act, state)
        done = jnp.logical_or(transition.term, transition.trun)
        boundary = self._Boundary(
            done=done,
            advance=advance,
            continued=jax.vmap(self.mc._continue_episode)(advance),
        )
        state = jax.lax.cond(
            jnp.any(done),
            self._reset_boundaries,
            self._continue_boundaries,
            boundary,
        )
        return transition, state

    def metrics(
        self,
        state: Mc.State[EnvStateT],
    ) -> tuple[Mc.Metrics, Mc.State[EnvStateT]]:
        """Aggregate episode metrics across lanes and clear their queues."""
        per_env, state = jax.vmap(self.mc.metrics)(state)
        return (
            Mc.Metrics(
                avg_eps_rew=jnp.nanmean(per_env.avg_eps_rew),
                avg_eps_len=jnp.nanmean(per_env.avg_eps_len),
            ),
            state,
        )

    def _reset_boundaries(
        self,
        boundary: VecMc._Boundary[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Prepare reset states and select them only for completed lanes."""
        reset = jax.vmap(self.mc._reset_episode)(boundary.advance)
        return self._select_boundary(boundary.done, reset, boundary.continued)

    @staticmethod
    def _continue_boundaries(
        boundary: VecMc._Boundary[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Return already-computed continuation states for a normal batch."""
        return boundary.continued

    @staticmethod
    def _select_boundary(
        boundary: jax.Array,
        reset: Mc.State[EnvStateT],
        continued: Mc.State[EnvStateT],
    ) -> Mc.State[EnvStateT]:
        """Select reset-state leaves for boundary lanes."""

        def select(reset_leaf: jax.Array, continued_leaf: jax.Array) -> jax.Array:
            extra_dims = continued_leaf.ndim - boundary.ndim
            if extra_dims < 0:
                raise ValueError("state leaves must include the environment batch axis")
            mask = jnp.reshape(boundary, (*boundary.shape, *(1,) * extra_dims))
            return jnp.where(mask, reset_leaf, continued_leaf)

        return cast(
            Mc.State[EnvStateT],
            jax.tree.map(select, reset, continued),
        )
