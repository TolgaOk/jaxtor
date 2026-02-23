"""Gymnasium environment adapter.

Wraps Gymnasium environments to conform to jaxtor's per-env Env protocol
using io_callback and custom_vmap. Each GymEnv presents a scalar (per-env)
interface identical to TabularEnv/GymnaxEnv. When VecMc vmaps over Mc.sample,
the custom_vmap rule intercepts and issues a single batched io_callback to
Gymnasium's VectorEnv.

Auto-reset (SAME_STEP) keeps Gymnasium's internal state valid. Mc's
lax.select handles the JAX-side reset logic; env.reset is a no-op returning
cached observations.

Example:
    >>> import jax
    >>> import jax.random as jrd
    >>> from jaxtor.env import gymnasium
    >>> from jaxtor.sampler.mc import Mc, VecMc
    >>> env = gymnasium.make("CartPole-v1", num_envs=4)
    >>> mc = Mc(max_episode_len=500, queue_size=10, env=env)
    >>> vec_mc = VecMc(mc=mc)
    >>> key = jax.random.PRNGKey(0)
    >>> env_state = env.init(key)
    >>> keys = jrd.split(key, 4)
    >>> mc_state = vec_mc.init(keys, env_state)
    >>> actions = jax.random.randint(key, (4,), 0, 2)
    >>> transition, mc_state = vec_mc.sample(actions, mc_state)
"""

from __future__ import annotations

from typing import Callable

import gymnasium as _gymnasium
from gymnasium.vector import AutoresetMode
import jax
import jax.numpy as jnp
import numpy as np
import chex
from chex import dataclass


@dataclass
class GymEnv:
    """Adapter for Gymnasium vectorized environments.

    Presents a per-env interface (scalar rewards, terminations, truncations)
    identical to TabularEnv and GymnaxEnv. Under vmap, custom_vmap rules on
    ``_io_step`` and ``_io_reset`` route to single batched operations.

    Attributes:
        _vec_env: Gymnasium VectorEnv instance.
        _io_step: Custom-vmap-decorated step closure.
        _io_reset: Custom-vmap-decorated reset closure.
        _init_obs: Mutable cache ``[jnp.array | None]`` for per-env initial obs.
        _num_envs: Number of parallel environments.
        _obs_shape: Per-env observation shape.
        _obs_dtype: Observation dtype.
        _act_shape: Per-env action shape.
        _act_dtype: Action dtype.
    """

    _vec_env: _gymnasium.vector.VectorEnv
    _io_step: Callable
    _io_reset: Callable
    _init_obs: list
    _num_envs: int
    _obs_shape: tuple
    _obs_dtype: jnp.dtype
    _act_shape: tuple
    _act_dtype: np.dtype

    @dataclass
    class State:
        """Per-env environment state.

        Attributes:
            obs: Current observation, shape ``(*obs_shape)``.
            reset_obs: Cached auto-reset observation, shape ``(*obs_shape)``.
        """

        obs: chex.Array
        reset_obs: chex.Array

    @dataclass
    class Step:
        """Per-env single-step transition result.

        Attributes:
            nobs: Next observation, shape ``(*obs_shape)``.
            rew: Reward, scalar.
            term: Natural termination flag, scalar.
            trun: Truncation flag, scalar.
        """

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def init(self, key: chex.PRNGKey) -> GymEnv.State:
        """Initialize the environment.

        Resets all Gymnasium envs and returns a single-env template state.
        For vectorized usage, pass this to ``VecMc.init(keys, env_state)``
        which vmaps ``mc.init`` with ``in_axes=(0, None)``.

        Args:
            key: Random key used to seed the Gymnasium environments.

        Returns:
            Single-env template state with shape ``(*obs_shape)``.
        """
        seed = int(jax.random.bits(key))
        obs, _ = self._vec_env.reset(seed=seed)
        obs = jnp.array(np.array(obs, dtype=self._obs_dtype))
        self._init_obs[0] = obs
        return self.State(obs=obs[0], reset_obs=obs[0])

    def step(
        self, key: chex.PRNGKey, act: chex.Array, state: State
    ) -> tuple[Step, State]:
        """Step one environment with auto-reset handling.

        Gymnasium auto-resets done envs (SAME_STEP mode). The true terminal
        observation is extracted from ``info["final_obs"]``; the returned obs
        is the auto-reset observation cached in state.

        Args:
            key: Random key (unused).
            act: Action, shape ``(*act_shape)``.
            state: Current per-env state.

        Returns:
            Step result and updated state.
        """
        nobs, rew, term, trun, reset_obs = self._io_step(act)
        done = jnp.logical_or(term, trun)
        new_reset_obs = jnp.where(done, reset_obs, state.reset_obs)
        return (
            self.Step(nobs=nobs, rew=rew, term=term, trun=trun),
            state.replace(obs=nobs, reset_obs=new_reset_obs),
        )

    def reset(self, key: chex.PRNGKey, state: State) -> tuple[chex.Array, State]:
        """Return cached reset observation (no-op).

        The actual reset happens inside Gymnasium's auto-reset. This returns
        cached ``reset_obs``, safe for Mc's ``lax.select``. Under vmap during
        ``VecMc.init``, the custom_vmap rule returns per-env initial obs from
        the cache populated by ``init()``.

        Args:
            key: Random key (unused).
            state: Current per-env state.

        Returns:
            Cached reset observation and updated state.
        """
        obs = self._io_reset(key, state.reset_obs)
        return obs, state.replace(obs=obs)

    def obs(self, state: State) -> chex.Array:
        """Get observation from state.

        Args:
            state: Current per-env state.

        Returns:
            Current observation, shape ``(*obs_shape)``.
        """
        return state.obs


def make(
    name: str,
    num_envs: int | None = None,
    async_envs: bool = False,
    **kwargs,
) -> GymEnv:
    """Create a Gymnasium environment adapter.

    Constructs a ``GymEnv`` with ``custom_vmap``-decorated closures for
    ``step`` and ``reset`` that enable efficient batched execution under
    ``jax.vmap``.

    Args:
        name: Gymnasium environment name (e.g. ``"CartPole-v1"``).
        num_envs: Number of parallel environments. ``None`` for a single env.
        async_envs: Use AsyncVectorEnv (multiprocessing) if True.
        **kwargs: Overrides passed to ``gymnasium.make``.

    Returns:
        GymEnv instance.
    """
    if num_envs is None:
        num_envs = 1
    vec_env = _gymnasium.make_vec(
        name,
        num_envs=num_envs,
        vectorization_mode="async" if async_envs else "sync",
        vector_kwargs={"autoreset_mode": AutoresetMode.SAME_STEP},
        **kwargs,
    )
    obs_shape = vec_env.single_observation_space.shape
    act_shape = vec_env.single_action_space.shape
    obs_dtype = jnp.float32
    act_dtype = vec_env.single_action_space.dtype

    _init_obs = [None]

    # -- step closure with custom_vmap --

    @jax.custom_batching.custom_vmap
    def _io_step(act):
        """Per-env step via io_callback (scalar path)."""
        result_shapes = (
            jax.ShapeDtypeStruct(obs_shape, obs_dtype),
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct((), jnp.bool_),
            jax.ShapeDtypeStruct(obs_shape, obs_dtype),
        )

        def _single_step(a):
            acts = np.array(a, dtype=act_dtype).reshape(1, *act_shape)
            obs, rew, term, trun, info = vec_env.step(acts)
            done = np.logical_or(term, trun)
            reset_obs = np.array(obs[0], dtype=obs_dtype)
            true_obs = np.array(obs[0], dtype=obs_dtype).copy()
            if done[0] and info["final_obs"][0] is not None:
                true_obs = np.array(info["final_obs"][0], dtype=obs_dtype)
            return (
                true_obs,
                np.array(rew[0], dtype=np.float32),
                np.array(term[0], dtype=np.bool_),
                np.array(trun[0], dtype=np.bool_),
                reset_obs,
            )

        return jax.experimental.io_callback(_single_step, result_shapes, act)

    @_io_step.def_vmap
    def _io_step_vmap(axis_size, in_batched, acts):
        """Batched step — one io_callback for all envs."""
        (act_batched,) = in_batched
        if not act_batched:
            raise ValueError("act must be batched under vmap")
        if axis_size != num_envs:
            raise ValueError(f"vmap axis_size ({axis_size}) != num_envs ({num_envs})")

        result_shapes = (
            jax.ShapeDtypeStruct((num_envs, *obs_shape), obs_dtype),
            jax.ShapeDtypeStruct((num_envs,), jnp.float32),
            jax.ShapeDtypeStruct((num_envs,), jnp.bool_),
            jax.ShapeDtypeStruct((num_envs,), jnp.bool_),
            jax.ShapeDtypeStruct((num_envs, *obs_shape), obs_dtype),
        )

        def _batched_step(acts):
            obs, rew, term, trun, info = vec_env.step(np.array(acts, dtype=act_dtype))
            done = np.logical_or(term, trun)
            reset_obs = np.array(obs, dtype=obs_dtype)
            true_obs = np.array(obs, dtype=obs_dtype).copy()
            for i in range(num_envs):
                if done[i] and info["final_obs"][i] is not None:
                    true_obs[i] = np.array(info["final_obs"][i], dtype=obs_dtype)
            return (
                true_obs,
                np.array(rew, dtype=np.float32),
                np.array(term, dtype=np.bool_),
                np.array(trun, dtype=np.bool_),
                reset_obs,
            )

        results = jax.experimental.io_callback(_batched_step, result_shapes, acts)
        return results, (True, True, True, True, True)

    # -- reset closure with custom_vmap --

    @jax.custom_batching.custom_vmap
    def _io_reset(key, reset_obs):
        """Scalar reset — return cached reset_obs as-is."""
        return reset_obs

    @_io_reset.def_vmap
    def _io_reset_vmap(axis_size, in_batched, keys, reset_obss):
        """Batched reset — dispatch based on whether state is batched."""
        _, reset_obs_batched = in_batched
        if reset_obs_batched:
            return reset_obss, True
        if _init_obs[0] is None:
            raise RuntimeError("env.init() must be called before VecMc.init()")
        if axis_size != num_envs:
            raise ValueError(f"vmap axis_size ({axis_size}) != num_envs ({num_envs})")
        return _init_obs[0], True

    return GymEnv(
        _vec_env=vec_env,
        _io_step=_io_step,
        _io_reset=_io_reset,
        _init_obs=_init_obs,
        _num_envs=num_envs,
        _obs_shape=obs_shape,
        _obs_dtype=obs_dtype,
        _act_shape=act_shape,
        _act_dtype=act_dtype,
    )
