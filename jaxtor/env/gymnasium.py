"""Gymnasium adapter for JAX samplers.

``GymEnv`` keeps static configuration in the component and mutable environments
in a process-local runtime store. Its array-only state carries an opaque handle,
allowing ``jit``, ``scan``, and ``vmap`` to use the external environment through
host callbacks.

Copied handles share the same runtime. Close it with ``env.close(state)`` when
it is no longer needed.
"""

from __future__ import annotations

import atexit
import itertools
import threading
from dataclasses import dataclass as py_dataclass
from typing import Any, Callable, Generic, ParamSpec, Protocol, TypeVar, cast

import chex
from chex import dataclass  # pyright: ignore[reportUnknownVariableType]
import gymnasium as gym
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

__all__ = ["GymEnv", "make"]


@py_dataclass(frozen=True)
class Schema:
    """Static array schema needed while tracing callbacks."""

    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype
    act_shape: tuple[int, ...]
    act_dtype: np.dtype


class ArraySpace(Protocol):
    """Array metadata consumed from a Gymnasium-compatible space."""

    @property
    def shape(self) -> tuple[int, ...] | None: ...

    @property
    def dtype(self) -> np.dtype | None: ...


class ScalarRuntime(Protocol):
    """Minimal mutable scalar-environment surface used by the adapter."""

    observation_space: ArraySpace
    action_space: ArraySpace

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[npt.ArrayLike, dict[str, Any]]: ...

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        dict[str, Any],
    ]: ...

    def close(self) -> None: ...


class VectorRuntime(Protocol):
    """Minimal mutable vector-environment surface used by the adapter."""

    single_observation_space: ArraySpace
    single_action_space: ArraySpace

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[npt.ArrayLike, dict[str, Any]]: ...

    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        npt.ArrayLike,
        dict[str, Any],
    ]: ...

    def close(self) -> None: ...


type BackendRuntime = ScalarRuntime | VectorRuntime


class Runtime:
    """One live, mutable environment outside the JAX pytree."""

    def __init__(self, env: BackendRuntime, initial_obs: np.ndarray):
        self.env = env
        self.initial_obs = initial_obs
        self.token = 0
        self.closed = False
        self.lock = threading.Lock()

    def close(self) -> None:
        """Close exactly once, serialized against an in-flight callback."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.env.close()


RuntimeFactory = Callable[[int | None], tuple[BackendRuntime, np.ndarray]]
runtime_store: dict[int, Runtime] = {}
runtime_store_lock = threading.Lock()
next_runtime_id = itertools.count(1)


def make_scalar_runtime(name: str, **kwargs: Any) -> ScalarRuntime:
    """Create a scalar probe while narrowing Gymnasium's unbound generics."""
    return cast(
        ScalarRuntime,
        gym.make(  # pyright: ignore[reportUnknownMemberType]
            name, **kwargs
        ),
    )


def make_vector_runtime(name: str, **kwargs: Any) -> VectorRuntime:
    """Create a vector runtime while narrowing Gymnasium's unbound generics."""
    return cast(
        VectorRuntime,
        gym.make_vec(  # pyright: ignore[reportUnknownMemberType]
            name, **kwargs
        ),
    )


def register_runtime(runtime: Runtime) -> int:
    """Register a runtime and return its array-compatible opaque id."""
    runtime_id = next(next_runtime_id)
    if runtime_id > np.iinfo(np.int32).max:
        raise OverflowError("Gymnasium runtime id space exhausted")
    with runtime_store_lock:
        runtime_store[runtime_id] = runtime
    return runtime_id


def lookup_runtime(runtime_ids: npt.ArrayLike) -> Runtime:
    """Resolve one repeated runtime id received by a host callback."""
    ids = np.asarray(runtime_ids, dtype=np.int32).reshape(-1)
    if len(ids) == 0 or np.any(ids != ids[0]):
        raise RuntimeError(f"one callback received mixed runtime ids: {ids}")
    runtime_id = int(ids[0])
    with runtime_store_lock:
        runtime = runtime_store.get(runtime_id)
    if runtime is None:
        raise RuntimeError(f"Gymnasium runtime {runtime_id} is closed or unknown")
    return runtime


def close_runtime_ids(runtime_ids: npt.ArrayLike) -> None:
    """Close selected runtimes before surfacing the first close error.

    Already-closed handles are ignored.
    """
    ids = np.unique(np.asarray(runtime_ids, dtype=np.int32))
    with runtime_store_lock:
        runtimes = [runtime_store.pop(int(runtime_id), None) for runtime_id in ids]

    error = None
    for runtime in runtimes:
        if runtime is None:
            continue
        try:
            runtime.close()
        except Exception as exc:
            if error is None:
                error = exc
    if error is not None:
        raise error


def close_all_runtimes() -> None:
    """Best-effort process cleanup, also used to isolate backend tests."""
    with runtime_store_lock:
        runtimes = list(runtime_store.values())
        runtime_store.clear()
    for runtime in runtimes:
        try:
            runtime.close()
        except Exception:
            pass


atexit.register(close_all_runtimes)


P = ParamSpec("P")
R = TypeVar("R")
Leaf = TypeVar("Leaf")
type StepBuffers[Buffer] = tuple[Buffer, Buffer, Buffer]


def attach_vmap_rule(
    function: Callable[P, R],
    rule: Callable[..., tuple[R, Any]],
) -> Callable[P, R]:
    """Attach a batching rule while preserving the callable's public type."""
    operation = jax.custom_batching.custom_vmap(function)
    operation.def_vmap(rule)
    return cast(Callable[P, R], operation)


class CallbackIO:
    """Bridge JAX transformations to registered mutable runtimes."""

    TOKEN_MAX = int(np.iinfo(np.int32).max)

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class Step(Generic[Leaf]):
        """Semantic step leaves or their matching custom-vmap batch flags.

        Attributes:
            nobs: Observation associated with the transition.
            rew: Scalar reward.
            term: Whether the environment terminated naturally.
            trun: Whether the environment was truncated.
            reset_obs: Post-autoreset observation cached for reset selection.
            runtime: Updated ``[runtime_id, token]`` handle.
        """

        nobs: Leaf
        rew: Leaf
        term: Leaf
        trun: Leaf
        reset_obs: Leaf
        runtime: Leaf

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class Reset(Generic[Leaf]):
        """Semantic reset leaves or their matching custom-vmap batch flags.

        Attributes:
            obs: Selected reset observation.
            runtime: Unchanged ``[runtime_id, token]`` handle.
            reset_obs: Cached reset observation retained in State.
        """

        obs: Leaf
        runtime: Leaf
        reset_obs: Leaf

    def __init__(
        self, schema: Schema, num_envs: int | None, allow_subset: bool
    ):
        self.schema = schema
        self.num_envs = num_envs
        self.allow_subset = allow_subset
        self.step_operation = attach_vmap_rule(self.jax_step, self.jax_step_vmap)
        self.reset_operation = attach_vmap_rule(self.jax_reset, self.jax_reset_vmap)

    def step(self, runtime: jax.Array, action: jax.Array) -> Step[jax.Array]:
        """Step one logical environment through the registered runtime."""
        return self.step_operation(runtime, action)

    def reset(
        self, key: jax.Array, runtime: jax.Array, reset_obs: jax.Array
    ) -> Reset[jax.Array]:
        """Select one logical environment's cached reset observation."""
        return self.reset_operation(key, runtime, reset_obs)

    def validate_axis(self, axis_size: int) -> None:
        """Validate one mapped axis against the configured vector pool."""
        if self.num_envs is None:
            raise ValueError(
                "cannot vmap a scalar Gymnasium runtime; pass num_envs to "
                "create a vector runtime"
            )
        if axis_size > self.num_envs:
            raise ValueError(
                f"vmap axis_size ({axis_size}) > num_envs ({self.num_envs})"
            )
        if not self.allow_subset and axis_size != self.num_envs:
            raise ValueError(
                f"vmap axis_size ({axis_size}) != num_envs ({self.num_envs}); "
                "this backend steps its whole pool"
            )

    @staticmethod
    def validate_runtime(runtime: Runtime, token: int, operation: str) -> None:
        """Validate lifecycle and linear-state invariants for a host operation."""
        match runtime.closed, token == runtime.token:
            case True, _:
                raise RuntimeError(f"cannot {operation} a closed Gymnasium runtime")
            case False, False:
                raise RuntimeError(
                    f"cannot {operation} an out-of-order or forked State: "
                    f"state token={token}, runtime token={runtime.token}"
                )
            case False, True:
                return

    def host_step(
        self, handle: npt.ArrayLike, actions: npt.ArrayLike
    ) -> StepBuffers[np.ndarray]:
        """Validate and execute one complete vector-runtime transaction.

        Returns:
            A tuple of transition observations, post-autoreset observations,
            and packed ``int32`` metadata. Metadata columns contain reward
            bits, termination, truncation, runtime id, and runtime token.
        """
        runtime_id, token = np.asarray(handle, dtype=np.int32)
        runtime_id, token = int(runtime_id), int(token)
        action_batch = np.asarray(actions, dtype=self.schema.act_dtype)
        self.validate_axis(len(action_batch))
        runtime = lookup_runtime(runtime_id)

        with runtime.lock:
            self.validate_runtime(runtime, token, "step")
            if runtime.token == self.TOKEN_MAX:
                raise OverflowError("Gymnasium runtime token exhausted")
            vector_env = cast(VectorRuntime, runtime.env)
            obs, rew, term, trun, info = vector_env.step(action_batch)
            reset_obs = np.asarray(obs, dtype=self.schema.obs_dtype)
            nobs = reset_obs.copy()
            done = np.logical_or(term, trun)
            if np.any(done):
                final_obs = info["final_obs"]
                for slot in np.flatnonzero(done):
                    nobs[slot] = np.asarray(
                        final_obs[slot], dtype=self.schema.obs_dtype
                    )

            runtime.token += 1
            metadata = np.empty((len(action_batch), 5), dtype=np.int32)
            metadata[:, 0] = np.asarray(rew, dtype=np.float32).view(np.int32)
            metadata[:, 1] = np.asarray(term, dtype=np.int32)
            metadata[:, 2] = np.asarray(trun, dtype=np.int32)
            metadata[:, 3] = runtime_id
            metadata[:, 4] = runtime.token
            return nobs, reset_obs, metadata

    def host_step_scalar(
        self, handle: npt.ArrayLike, action: npt.ArrayLike
    ) -> StepBuffers[np.ndarray]:
        """Step and autoreset one scalar runtime transaction."""
        runtime_id, token = np.asarray(handle, dtype=np.int32)
        runtime_id, token = int(runtime_id), int(token)
        action_array = np.asarray(action, dtype=self.schema.act_dtype).reshape(
            self.schema.act_shape
        )
        scalar_action = (
            action_array.item() if self.schema.act_shape == () else action_array
        )
        runtime = lookup_runtime(runtime_id)

        with runtime.lock:
            self.validate_runtime(runtime, token, "step")
            if runtime.token == self.TOKEN_MAX:
                raise OverflowError("Gymnasium runtime token exhausted")
            scalar_env = cast(ScalarRuntime, runtime.env)
            obs, rew, term, trun, _ = scalar_env.step(scalar_action)
            nobs = np.asarray(obs, dtype=self.schema.obs_dtype)
            reset_obs = nobs
            if bool(np.logical_or(term, trun)):
                reset_obs, _ = scalar_env.reset()
                reset_obs = np.asarray(reset_obs, dtype=self.schema.obs_dtype)

            runtime.token += 1
            metadata = np.empty(5, dtype=np.int32)
            metadata[0] = np.asarray(rew, dtype=np.float32).view(np.int32)
            metadata[1] = np.asarray(term, dtype=np.int32)
            metadata[2] = np.asarray(trun, dtype=np.int32)
            metadata[3] = runtime_id
            metadata[4] = runtime.token
            return nobs, reset_obs, metadata

    def host_step_one(
        self, handle: npt.ArrayLike, action: npt.ArrayLike
    ) -> StepBuffers[np.ndarray]:
        """Step a scalar runtime or one lane of a size-one vector runtime."""
        if self.num_envs is None:
            return self.host_step_scalar(handle, action)

        actions = np.asarray(action, dtype=self.schema.act_dtype).reshape(
            1, *self.schema.act_shape
        )
        nobs, reset_obs, metadata = self.host_step(handle, actions)
        return nobs[0], reset_obs[0], metadata[0]

    def host_initial_obs(
        self, handle: npt.ArrayLike, axis_size: npt.ArrayLike
    ) -> np.ndarray:
        """Read the unused initial observations for one runtime."""
        runtime_id, token = np.asarray(handle, dtype=np.int32)
        runtime_id, token = int(runtime_id), int(token)
        size = int(np.asarray(axis_size))
        runtime = lookup_runtime(runtime_id)
        with runtime.lock:
            self.validate_runtime(runtime, token, "initialize from")
            return runtime.initial_obs[:size].copy()

    def step_spec(self, axis_size: int | None) -> StepBuffers[jax.ShapeDtypeStruct]:
        """Describe scalar or vector callback outputs to JAX.

        Returns:
            Shape descriptors ordered as transition observations,
            post-autoreset observations, and packed metadata.
        """
        match axis_size:
            case None:
                obs_shape = self.schema.obs_shape
                metadata_shape = (5,)
            case int(size):
                obs_shape = (size, *self.schema.obs_shape)
                metadata_shape = (size, 5)
        return (
            jax.ShapeDtypeStruct(obs_shape, self.schema.obs_dtype),
            jax.ShapeDtypeStruct(obs_shape, self.schema.obs_dtype),
            jax.ShapeDtypeStruct(metadata_shape, jnp.int32),
        )

    def decode_step(
        self,
        nobs: jax.Array,
        reset_obs: jax.Array,
        metadata: jax.Array,
    ) -> Step[jax.Array]:
        """Decode physical callback buffers into a semantic step result."""
        return self.Step(
            nobs=nobs,
            rew=jax.lax.bitcast_convert_type(metadata[..., 0], jnp.float32),
            term=metadata[..., 1].astype(jnp.bool_),
            trun=metadata[..., 2].astype(jnp.bool_),
            reset_obs=reset_obs,
            runtime=metadata[..., 3:5],
        )

    def jax_step(self, runtime: jax.Array, action: jax.Array) -> Step[jax.Array]:
        """Step one scalar State through a one-element host callback."""
        nobs, reset_obs, metadata = cast(
            StepBuffers[jax.Array],
            io_callback(
                self.host_step_one,
                self.step_spec(None),
                runtime,
                action,
                ordered=False,
            ),
        )
        return self.decode_step(nobs, reset_obs, metadata)

    def jax_step_vmap(
        self,
        axis_size: int,
        in_batched: list[bool],
        runtimes: jax.Array,
        actions: jax.Array,
    ) -> tuple[Step[jax.Array], Step[bool]]:
        """Batch one mapped environment axis into one vector callback."""
        match in_batched:
            case [True, True]:
                pass
            case _:
                raise ValueError("runtime and action must share the vmap axis")
        self.validate_axis(axis_size)
        nobs, reset_obs, metadata = cast(
            StepBuffers[jax.Array],
            io_callback(
                self.host_step,
                self.step_spec(axis_size),
                runtimes[0],
                actions,
                ordered=False,
            ),
        )
        return self.decode_step(nobs, reset_obs, metadata), self.Step(
            nobs=True,
            rew=True,
            term=True,
            trun=True,
            reset_obs=True,
            runtime=True,
        )

    def jax_reset(
        self, key: jax.Array, runtime: jax.Array, reset_obs: jax.Array
    ) -> Reset[jax.Array]:
        """Select the reset observation already cached by SAME_STEP."""
        del key
        return self.Reset(obs=reset_obs, runtime=runtime, reset_obs=reset_obs)

    def initial_batch(
        self, axis_size: int, runtimes: jax.Array
    ) -> tuple[Reset[jax.Array], Reset[bool]]:
        """Bind an unbatched template State to a vector-pool prefix."""
        self.validate_axis(axis_size)
        initial_obs = cast(
            jax.Array,
            io_callback(
                self.host_initial_obs,
                jax.ShapeDtypeStruct(
                    (axis_size, *self.schema.obs_shape), self.schema.obs_dtype
                ),
                runtimes,
                np.int32(axis_size),
                ordered=False,
            ),
        )
        batched_runtimes = jnp.broadcast_to(runtimes, (axis_size, 2))
        return self.Reset(
            obs=initial_obs,
            runtime=batched_runtimes,
            reset_obs=initial_obs,
        ), self.Reset(obs=True, runtime=True, reset_obs=True)

    def jax_reset_vmap(
        self,
        axis_size: int,
        in_batched: list[bool],
        keys: jax.Array,
        runtimes: jax.Array,
        reset_obss: jax.Array,
    ) -> tuple[Reset[jax.Array], Reset[bool]]:
        """Batch cached resets or bind a scalar template to pool slots."""
        del keys
        match in_batched:
            case [_, True, True]:
                return self.Reset(
                    obs=reset_obss,
                    runtime=runtimes,
                    reset_obs=reset_obss,
                ), self.Reset(obs=True, runtime=True, reset_obs=True)
            case [_, False, False]:
                return self.initial_batch(axis_size, runtimes)
            case _:
                raise ValueError("environment State leaves must share one vmap axis")


@dataclass  # pyright: ignore[reportUntypedClassDecorator]
class GymEnv:
    """Configured adapter for a mutable Gymnasium-like backend.

    The component contains a runtime factory, static array schema, and callback
    rules. Live environments are stored outside it and are addressed solely by
    the opaque fields threaded through :class:`State`.
    """

    factory: RuntimeFactory
    io: CallbackIO
    num_envs: int | None
    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype
    act_shape: tuple[int, ...]
    act_dtype: np.dtype

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class State:
        """Dynamic per-environment state.

        Attributes:
            runtime: Opaque ``[runtime_id, token]`` capability for the live pool.
            obs: Current observation, shape ``(*obs_shape)``.
            reset_obs: Most recent SAME_STEP reset observation.
        """

        runtime: jax.Array
        obs: chex.Array
        reset_obs: chex.Array

        @property
        def runtime_id(self) -> jax.Array:
            """Process-local identity encoded in the opaque runtime handle."""
            return self.runtime[..., 0]

        @property
        def token(self) -> jax.Array:
            """Sequencing value encoded in the opaque runtime handle."""
            return self.runtime[..., 1]

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class Step:
        """Per-environment transition returned by :meth:`step`."""

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def init(self, key: chex.PRNGKey) -> GymEnv.State:
        """Create a runtime and return its scalar state or vector template.

        ``VecMc.init`` vmaps its scalar initializer over this template; the
        custom reset rule binds those states to pool slots ``0..n-1``.
        """
        seed = int(
            np.asarray(
                jax.device_get(
                    jax.random.bits(  # pyright: ignore[reportUnknownMemberType]
                        key
                    )
                )
            )
        )
        runtime_env, initial_obs = self.factory(seed)
        initial_obs = np.asarray(initial_obs, dtype=self.obs_dtype)
        expected_shape = (
            self.obs_shape
            if self.num_envs is None
            else (self.num_envs, *self.obs_shape)
        )
        if initial_obs.shape != expected_shape:
            runtime_env.close()
            raise ValueError(
                f"runtime reset returned shape {initial_obs.shape}, "
                f"expected {expected_shape}"
            )

        runtime_id = register_runtime(Runtime(runtime_env, initial_obs))
        template_obs = initial_obs if self.num_envs is None else initial_obs[0]
        obs = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            template_obs
        )
        return self.State(
            runtime=jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
                (runtime_id, 0), dtype=jnp.int32
            ),
            obs=obs,
            reset_obs=obs,
        )

    def step(
        self, key: chex.PRNGKey, act: chex.Array, state: State
    ) -> tuple[Step, State]:
        """Step one logical environment through the external runtime."""
        del key
        result = self.io.step(state.runtime, cast(jax.Array, act))
        done = jnp.logical_or(result.term, result.trun)
        reset_obs = jnp.where(done, result.reset_obs, state.reset_obs)
        return (
            self.Step(
                nobs=result.nobs,
                rew=result.rew,
                term=result.term,
                trun=result.trun,
            ),
            state.replace(  # type: ignore[reportAttributeAccessIssue]
                runtime=result.runtime,
                obs=result.nobs,
                reset_obs=reset_obs,
            ),
        )

    def reset(self, key: chex.PRNGKey, state: State) -> tuple[chex.Array, State]:
        """Select the cached SAME_STEP reset observation."""
        result = self.io.reset(
            key,
            state.runtime,
            cast(jax.Array, state.reset_obs),
        )
        return result.obs, self.State(
            runtime=result.runtime,
            obs=result.obs,
            reset_obs=result.reset_obs,
        )

    def obs(self, state: State) -> chex.Array:
        """Return the current observation."""
        return state.obs

    def close(self, state: State) -> None:
        """Close the runtime referenced by a scalar or batched state.

        This is a host-side lifecycle operation and must not be called inside a
        JAX transformation. The supplied token is awaited first, so passing the
        latest state also waits for its in-flight callback chain.
        """
        jax.device_get(state.runtime)
        runtime_ids = np.asarray(jax.device_get(state.runtime_id), dtype=np.int32)
        close_runtime_ids(runtime_ids)


def make(
    name: str,
    num_envs: int | None = None,
    async_envs: bool = False,
    **kwargs: Any,
) -> GymEnv:
    """Configure a Gymnasium environment adapter.

    ``num_envs=None`` creates a scalar ``gym.Env`` for plain ``Mc``. Any
    positive integer, including one, creates a ``VectorEnv`` for ``VecMc``.
    The live runtime is created later by :meth:`GymEnv.init`.
    """
    if num_envs is not None and num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if num_envs is None and async_envs:
        raise ValueError("async_envs requires an integer num_envs")

    probe = make_scalar_runtime(name, **kwargs)
    try:
        observation_space = probe.observation_space
        action_space = probe.action_space
    finally:
        probe.close()

    def factory(seed: int | None) -> tuple[BackendRuntime, np.ndarray]:
        """Create and reset one live scalar or vector runtime."""
        runtime_env = (
            make_scalar_runtime(name, **kwargs)
            if num_envs is None
            else make_vector_runtime(
                name,
                num_envs=num_envs,
                vectorization_mode="async" if async_envs else "sync",
                vector_kwargs={"autoreset_mode": "SameStep"},
                **kwargs,
            )
        )
        try:
            obs, _ = runtime_env.reset(seed=seed)
        except Exception:
            runtime_env.close()
            raise
        return runtime_env, np.asarray(obs)

    return from_factory(
        factory,
        num_envs,
        observation_space=observation_space,
        action_space=action_space,
    )


def from_factory(
    factory: RuntimeFactory,
    num_envs: int | None,
    allow_subset: bool = False,
    *,
    observation_space: ArraySpace | None = None,
    action_space: ArraySpace | None = None,
) -> GymEnv:
    """Build an adapter from a seeded scalar- or vector-runtime factory.

    ``factory(seed)`` must return ``(runtime, initial_obs)``. Integer-sized
    vector runtimes must use SAME_STEP autoreset. When spaces are omitted, a
    transient unseeded runtime is created and closed to discover them.
    """
    if num_envs is None and allow_subset:
        raise ValueError("allow_subset applies only to vector runtimes")
    if (observation_space is None) != (action_space is None):
        raise ValueError("observation_space and action_space must be given together")

    if observation_space is None:
        probe, _ = factory(None)
        try:
            if num_envs is None:
                scalar_probe = cast(ScalarRuntime, probe)
                observation_space = scalar_probe.observation_space
                action_space = scalar_probe.action_space
            else:
                vector_probe = cast(VectorRuntime, probe)
                observation_space = vector_probe.single_observation_space
                action_space = vector_probe.single_action_space
        finally:
            probe.close()

    assert observation_space is not None and action_space is not None

    obs_shape = observation_space.shape
    act_shape = action_space.shape
    act_dtype = action_space.dtype
    if obs_shape is None:
        raise TypeError("observation space must define an array shape")
    if act_shape is None or act_dtype is None:
        raise TypeError("action space must define an array shape and dtype")

    schema = Schema(
        obs_shape=obs_shape,
        obs_dtype=np.dtype(np.float32),
        act_shape=act_shape,
        act_dtype=np.dtype(act_dtype),
    )
    return GymEnv(
        factory=factory,
        io=CallbackIO(schema, num_envs, allow_subset),
        num_envs=num_envs,
        obs_shape=schema.obs_shape,
        obs_dtype=schema.obs_dtype,
        act_shape=schema.act_shape,
        act_dtype=schema.act_dtype,
    )
