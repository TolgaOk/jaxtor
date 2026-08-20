"""Open Markov-chain sampling with episode lifecycle handling.

``Mc`` turns an environment into a transition sampler::

    mc = Mc(max_eps_len=500, env=env)
    state = mc.init(key, env_state)
    transition, state = mc.sample(act, state)

``VecMc`` applies the same interface to a batch of independent environment
states.
"""

from __future__ import annotations

from dataclasses import replace
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


class Env[S, Step: EnvStep](Protocol):
    """Environment capability required by ``Mc``."""

    def reset(self, key: jax.Array, state: S) -> tuple[chex.Numeric, S]: ...
    def step(self, key: jax.Array, act: jax.Array, state: S) -> tuple[Step, S]: ...


@dataclass
class Mc[EnvS, Step: EnvStep]:
    """Sample an open Markov chain with reset and truncation handling.

    Its state keeps the current observation and episode index so
    :meth:`observe` does not inspect backend-specific environment state.

    Required protocols::

        env.reset(key, env_state) -> (observation, env_state)
        env.step(key, action, env_state) -> (transition, env_state)
        transition: nobs, rew, term, trun

    Attributes:
        max_eps_len: Maximum number of transitions in one episode.
        env: Environment implementing ``reset`` and non-autoreset ``step``.

    Public dataclasses:
        State: Environment, RNG, current observation, and episode index.
        Transition: One aligned ``obs, act, rew, term, trun, nobs`` sample.

    Public methods:
        init: Initialize state from a backend-specific environment state.
        observe: Read the current observation without advancing state.
        sample: Advance once and autoreset at an episode boundary.
    """

    max_eps_len: int
    env: Env[EnvS, Step]

    @dataclass
    class State[EnvData]:
        """Dynamic state threaded through ``Mc``.

        Attributes:
            key: Random key used by environment steps and resets.
            env: Backend-specific environment state.
            last_obs: Observation from which the next action is taken.
            eps_idx: Number of transitions completed in the current episode.
        """

        key: jax.Array
        env: EnvData
        last_obs: jax.Array
        eps_idx: jax.Array

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
    class _Advance[EnvData]:
        """Data required to finish one environment advance."""

        state: Mc.State[EnvData]
        env: EnvData
        transition: Mc.Transition
        key: jax.Array
        reset_key: jax.Array

    def __post_init__(self) -> None:
        """Validate the static episode limit."""
        if self.max_eps_len < 1:
            raise ValueError("max_eps_len must be positive")

    def init(self, key: jax.Array, env: EnvS) -> Mc.State[EnvS]:
        """Initialize the Markov chain from an environment state."""
        key, reset_key = jrd.split(key)
        obs, env = self.env.reset(reset_key, env)
        return self.State(
            key=key,
            env=env,
            last_obs=jnp.asarray(obs),
            eps_idx=jnp.array(0, dtype=jnp.int32),
        )

    def observe(self, state: Mc.State[EnvS]) -> jax.Array:
        """Read the current observation without advancing state."""
        return state.last_obs

    def sample(
        self,
        act: chex.Array,
        state: Mc.State[EnvS],
    ) -> tuple[Mc.Transition, Mc.State[EnvS]]:
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

    def _advance(
        self,
        act: chex.Array,
        state: Mc.State[EnvS],
    ) -> tuple[Mc.Transition, Mc._Advance[EnvS]]:
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

        reached_limit = state.eps_idx + 1 >= self.max_eps_len
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

    def _reset_episode(self, advance: Mc._Advance[EnvS]) -> Mc.State[EnvS]:
        """Reset after a terminal or truncated transition."""
        state = advance.state
        obs, env = self.env.reset(advance.reset_key, advance.env)
        return replace(
            state,
            key=advance.key,
            env=env,
            last_obs=jnp.asarray(obs),
            eps_idx=jnp.zeros_like(state.eps_idx),
        )

    @staticmethod
    def _continue_episode(advance: Mc._Advance[EnvS]) -> Mc.State[EnvS]:
        """Continue an unfinished episode from its true next observation."""
        state = advance.state
        transition = advance.transition
        return replace(
            state,
            key=advance.key,
            env=advance.env,
            last_obs=transition.nobs,
            eps_idx=state.eps_idx + 1,
        )


@dataclass
class VecMc[EnvS, Step: EnvStep]:
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
    """

    mc: Mc[EnvS, Step]

    @dataclass
    class _Boundary[EnvData]:
        """Batched data required for conditional reset selection."""

        done: jax.Array
        advance: Mc._Advance[EnvData]
        continued: Mc.State[EnvData]

    def init(self, keys: jax.Array, env: EnvS) -> Mc.State[EnvS]:
        """Initialize from matching batches of keys and environment states."""
        return jax.vmap(self.mc.init)(keys, env)

    def observe(self, state: Mc.State[EnvS]) -> jax.Array:
        """Read the batched current observations."""
        return jax.vmap(self.mc.observe)(state)

    def sample(
        self,
        act: chex.Array,
        state: Mc.State[EnvS],
    ) -> tuple[Mc.Transition, Mc.State[EnvS]]:
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

    def _reset_boundaries(self, boundary: VecMc._Boundary[EnvS]) -> Mc.State[EnvS]:
        """Prepare reset states and select them only for completed lanes."""
        reset = jax.vmap(self.mc._reset_episode)(boundary.advance)
        return self._select_boundary(boundary.done, reset, boundary.continued)

    @staticmethod
    def _continue_boundaries(boundary: VecMc._Boundary[EnvS]) -> Mc.State[EnvS]:
        """Return already-computed continuation states for a normal batch."""
        return boundary.continued

    @staticmethod
    def _select_boundary(
        boundary: jax.Array,
        reset: Mc.State[EnvS],
        continued: Mc.State[EnvS],
    ) -> Mc.State[EnvS]:
        """Select reset-state leaves for boundary lanes."""

        def select(reset_leaf: jax.Array, continued_leaf: jax.Array) -> jax.Array:
            extra_dims = continued_leaf.ndim - boundary.ndim
            if extra_dims < 0:
                raise ValueError("state leaves must include the environment batch axis")
            mask = jnp.reshape(boundary, (*boundary.shape, *(1,) * extra_dims))
            return jnp.where(mask, reset_leaf, continued_leaf)

        return cast(
            Mc.State[EnvS],
            jax.tree.map(select, reset, continued),
        )
