"""Gymnasium adapter for JAX samplers.

``make`` stores backend configuration. Mapping ``init`` adds runtime axes without
creating a backend. Executing the transformed initializer creates one runtime
whose shape matches the complete key batch.

    env = make("CartPole-v1")
    state = jax.vmap(env.init)(keys)
    step, state = jax.vmap(env.step)(keys, act, state)
    env.close(state)
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Hashable
from dataclasses import dataclass as pydataclass, replace
import itertools
import threading
from typing import Any, Protocol, cast
import warnings

import chex
from chex import dataclass
import gymnasium as gym
import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import numpy.typing as npt

__all__ = ["GymEnv", "RuntimeRequest", "from_factory", "make"]


@pydataclass(frozen=True)
class Schema:
    """Static array schema required while tracing callbacks."""

    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype
    act_shape: tuple[int, ...]
    act_dtype: np.dtype


@pydataclass(frozen=True)
class RuntimeSpec:
    """Configuration that determines whether adapters may share a runtime."""

    key: Hashable
    schema: Schema
    always_vectorized: bool


@pydataclass(frozen=True)
class RuntimeRequest:
    """Complete physical runtime request produced by one executed ``init``.

    Attributes:
        capacity: Total number of physical environment lanes.
        seeds: One flattened seed per lane.
        vectorized: Whether the factory must return a vector runtime.
    """

    capacity: int
    seeds: tuple[int, ...]
    vectorized: bool


class ArraySpace(Protocol):
    """Array metadata consumed from a Gymnasium-compatible space."""

    @property
    def shape(self) -> tuple[int, ...] | None: ...
    @property
    def dtype(self) -> np.dtype | None: ...


type ResetResult = tuple[npt.ArrayLike, dict[str, Any]]
type StepResult = tuple[
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    npt.ArrayLike,
    dict[str, Any],
]


class ScalarRuntime(Protocol):
    """Minimal mutable scalar-environment surface used by the adapter."""

    observation_space: ArraySpace
    action_space: ArraySpace

    def reset(self, *, seed: int | None = None) -> ResetResult: ...
    def step(self, action: npt.ArrayLike) -> StepResult: ...
    def close(self) -> None: ...


class VectorRuntime(Protocol):
    """Minimal mutable vector-environment surface used by the adapter."""

    single_observation_space: ArraySpace
    single_action_space: ArraySpace

    def reset(self, *, seed: int | list[int] | None = None) -> ResetResult: ...
    def step(self, actions: npt.ArrayLike) -> StepResult: ...
    def close(self) -> None: ...


type BackendRuntime = ScalarRuntime | VectorRuntime
type RuntimeFactory = Callable[[RuntimeRequest], tuple[BackendRuntime, np.ndarray]]


class Runtime:
    """One live mutable environment stored outside the JAX pytree."""

    def __init__(
        self,
        env: BackendRuntime,
        spec: RuntimeSpec,
        batch_shape: tuple[int, ...],
        vectorized: bool,
    ):
        self.env = env
        self.spec = spec
        self.batch_shape = batch_shape
        self.capacity = int(np.prod(batch_shape)) if batch_shape else 1
        self.vectorized = vectorized
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


_runtime_store: dict[int, Runtime] = {}
_runtime_store_lock = threading.Lock()
_next_runtime_id = itertools.count(1)


def config_key(value: Any) -> Hashable:
    """Convert ordinary constructor values into a structural compatibility key."""
    match value:
        case dict():
            items = ((config_key(key), config_key(item)) for key, item in value.items())
            return ("dict", tuple(sorted(items, key=repr)))
        case list():
            return ("list", tuple(config_key(item) for item in value))
        case tuple():
            return ("tuple", tuple(config_key(item) for item in value))
        case set() | frozenset():
            return ("set", tuple(sorted(map(config_key, value), key=repr)))
        case np.ndarray():
            return ("array", value.dtype.str, value.shape, value.tobytes())
        case np.generic():
            return ("scalar", value.dtype.str, config_key(value.item()))

    try:
        hash(value)
    except TypeError:
        return (type(value).__module__, type(value).__qualname__, id(value))
    return (type(value).__module__, type(value).__qualname__, value)


def make_scalar_runtime(name: str, **kwargs: Any) -> ScalarRuntime:
    """Create a scalar probe while narrowing Gymnasium's unbound generics."""
    return cast(ScalarRuntime, gym.make(name, **kwargs))


def make_vector_runtime(name: str, **kwargs: Any) -> VectorRuntime:
    """Create a vector runtime while narrowing Gymnasium's unbound generics."""
    return cast(VectorRuntime, gym.make_vec(name, **kwargs))


def _register_runtime(runtime: Runtime) -> int:
    """Register a runtime and return its array-compatible identity."""
    runtime_id = next(_next_runtime_id)
    if runtime_id > np.iinfo(np.int32).max:
        raise OverflowError("Gymnasium runtime id space exhausted")
    with _runtime_store_lock:
        _runtime_store[runtime_id] = runtime
    return runtime_id


def _lookup_runtime(
    runtime_ids: npt.ArrayLike,
    spec: RuntimeSpec | None = None,
) -> Runtime:
    """Resolve one repeated runtime identity and validate its configuration."""
    ids = np.asarray(runtime_ids, dtype=np.int32).reshape(-1)
    if len(ids) == 0 or np.any(ids != ids[0]):
        raise RuntimeError(f"one callback received mixed runtime ids: {ids}")
    runtime_id = int(ids[0])
    with _runtime_store_lock:
        runtime = _runtime_store.get(runtime_id)
    if runtime is None:
        raise RuntimeError(f"Gymnasium runtime {runtime_id} is closed or unknown")
    if spec is not None and runtime.spec != spec:
        raise RuntimeError(
            f"Gymnasium runtime {runtime_id} belongs to an incompatible environment"
        )
    return runtime


def _close_runtime_ids(runtime_ids: npt.ArrayLike, spec: RuntimeSpec) -> None:
    """Validate, remove, and close every distinct referenced runtime."""
    ids = np.unique(np.asarray(runtime_ids, dtype=np.int32))
    with _runtime_store_lock:
        runtimes = [_runtime_store.get(int(runtime_id)) for runtime_id in ids]
        for runtime_id, runtime in zip(ids, runtimes, strict=True):
            if runtime is not None and runtime.spec != spec:
                raise RuntimeError(
                    f"Gymnasium runtime {int(runtime_id)} belongs to an "
                    "incompatible environment"
                )
        for runtime_id in ids:
            _runtime_store.pop(int(runtime_id), None)

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


def _close_all_runtimes() -> None:
    """Close remaining runtimes during process shutdown."""
    with _runtime_store_lock:
        runtimes = list(_runtime_store.values())
        _runtime_store.clear()

    errors: list[Exception] = []
    for runtime in runtimes:
        try:
            runtime.close()
        except Exception as error:
            errors.append(error)
    if errors:
        warnings.warn(
            f"failed to close {len(errors)} Gymnasium runtime(s): {errors[0]!r}",
            ResourceWarning,
            stacklevel=1,
        )


atexit.register(_close_all_runtimes)


type InitBuffers[Buffer] = tuple[Buffer, Buffer]
type StepBuffers[Buffer] = tuple[Buffer, Buffer, Buffer, Buffer, Buffer, Buffer]


def attach_vmap_rule[**Args, Out](
    function: Callable[Args, Out],
    rule: Callable[..., tuple[Out, Any]],
) -> Callable[Args, Out]:
    """Attach a batching rule while preserving the callable's public type."""
    operation = jax.custom_batching.custom_vmap(function)
    operation.def_vmap(rule)
    return cast(Callable[Args, Out], operation)


class CallbackIO:
    """Bridge transformed array operations to one registered mutable runtime."""

    TOKEN_MAX = int(np.iinfo(np.int32).max)

    @dataclass
    class Init[Leaf]:
        """Initialized state leaves or their custom-vmap batch flags."""

        runtime: Leaf
        token: Leaf
        obs: Leaf
        reset_obs: Leaf

    @dataclass
    class Step[Leaf]:
        """Step leaves or their custom-vmap batch flags."""

        nobs: Leaf
        rew: Leaf
        term: Leaf
        trun: Leaf
        reset_obs: Leaf
        token: Leaf

    def __init__(self, factory: RuntimeFactory, spec: RuntimeSpec):
        self.factory = factory
        self.spec = spec
        self.schema = spec.schema
        self.init = attach_vmap_rule(self.jax_init, self.jax_init_vmap)
        self.step = attach_vmap_rule(self.jax_step, self.jax_step_vmap)

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

    @staticmethod
    def validate_batch(runtime: Runtime, batch_shape: tuple[int, ...]) -> None:
        """Require state to retain the complete shape that created its runtime."""
        if batch_shape != runtime.batch_shape:
            raise RuntimeError(
                f"state batch shape {batch_shape} does not match runtime shape "
                f"{runtime.batch_shape}"
            )

    @staticmethod
    def seeds_from_keys(key_data: np.ndarray) -> tuple[int, ...]:
        """Derive one deterministic backend seed per mapped key."""
        words = np.asarray(key_data, dtype=np.uint32)
        keys = words.reshape(-1, words.shape[-1])
        return tuple(
            int(
                np.random.SeedSequence(key.tolist()).generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )
            for key in keys
        )

    def host_init(self, key_data: npt.ArrayLike) -> InitBuffers[np.ndarray]:
        """Create and register one runtime for the complete mapped key array."""
        keys = np.asarray(key_data, dtype=np.uint32)
        batch_shape = tuple(keys.shape[:-1])
        capacity = int(np.prod(batch_shape)) if batch_shape else 1
        vectorized = self.spec.always_vectorized or bool(batch_shape)
        runtime_env, initial_obs = self.factory(
            RuntimeRequest(
                capacity=capacity,
                seeds=self.seeds_from_keys(keys),
                vectorized=vectorized,
            )
        )
        initial_obs = np.asarray(initial_obs, dtype=self.schema.obs_dtype)
        expected_shape = (
            (capacity, *self.schema.obs_shape) if vectorized else self.schema.obs_shape
        )
        if initial_obs.shape != expected_shape:
            runtime_env.close()
            raise ValueError(
                f"runtime reset returned shape {initial_obs.shape}, "
                f"expected {expected_shape}"
            )

        runtime = Runtime(
            env=runtime_env,
            spec=self.spec,
            batch_shape=batch_shape,
            vectorized=vectorized,
        )
        try:
            runtime_id = _register_runtime(runtime)
        except Exception:
            runtime.close()
            raise

        obs = initial_obs.reshape((*batch_shape, *self.schema.obs_shape))
        metadata = np.stack(
            (
                np.full(capacity, runtime_id, dtype=np.int32),
                np.zeros(capacity, dtype=np.int32),
            ),
            axis=-1,
        ).reshape((*batch_shape, 2))
        return obs, metadata

    def init_spec(
        self,
        batch_shape: tuple[int, ...],
    ) -> InitBuffers[jax.ShapeDtypeStruct]:
        """Describe one initialized state per mapped key."""
        return (
            jax.ShapeDtypeStruct(
                (*batch_shape, *self.schema.obs_shape),
                self.schema.obs_dtype,
            ),
            jax.ShapeDtypeStruct((*batch_shape, 2), jnp.int32),
        )

    def jax_init(self, key: jax.Array) -> Init[jax.Array]:
        """Create a runtime only when the transformed initializer executes."""
        keys = jrd.key_data(key)
        obs, metadata = cast(
            InitBuffers[jax.Array],
            io_callback(
                self.host_init,
                self.init_spec(keys.shape[:-1]),
                keys,
                ordered=False,
            ),
        )
        return self.Init(
            runtime=metadata[..., 0],
            token=metadata[..., 1],
            obs=obs,
            reset_obs=obs,
        )

    def jax_init_vmap(
        self,
        axis_size: int,
        in_batched: list[bool],
        keys: jax.Array,
    ) -> tuple[Init[jax.Array], Init[bool]]:
        """Accumulate every enclosing mapped axis before initialization."""
        if axis_size < 1:
            raise ValueError("environment vmap axes must be nonempty")
        if in_batched != [True]:
            raise ValueError("the initialization key must carry every vmap axis")
        return self.init(keys), self.Init(
            runtime=True,
            token=True,
            obs=True,
            reset_obs=True,
        )

    def transition_obs(
        self,
        result: StepResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separate true transition observations from SAME_STEP reset data."""
        obs, rew, term, trun, info = result
        reset_obs = np.asarray(obs, dtype=self.schema.obs_dtype)
        nobs = reset_obs.copy()
        rew = np.asarray(rew, dtype=np.float32)
        term = np.asarray(term, dtype=np.bool_)
        trun = np.asarray(trun, dtype=np.bool_)
        done = np.logical_or(term, trun)
        if np.any(done):
            final_obs = info.get("final_obs")
            if final_obs is None:
                raise RuntimeError("SAME_STEP runtime omitted final_obs")
            for index in np.flatnonzero(done):
                nobs[index] = np.asarray(
                    final_obs[index],
                    dtype=self.schema.obs_dtype,
                )
        return nobs, reset_obs, rew, term, trun

    def step_scalar(
        self,
        runtime: Runtime,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step one scalar runtime and apply SAME_STEP reset semantics."""
        scalar_env = cast(ScalarRuntime, runtime.env)
        scalar_action = action.item() if self.schema.act_shape == () else action
        obs, rew, term, trun, _ = scalar_env.step(scalar_action)
        nobs = np.asarray(obs, dtype=self.schema.obs_dtype)
        reset_obs = nobs
        if bool(np.logical_or(term, trun)):
            reset_obs, _ = scalar_env.reset()
            reset_obs = np.asarray(reset_obs, dtype=self.schema.obs_dtype)
        return (
            nobs.reshape(1, *self.schema.obs_shape),
            reset_obs.reshape(1, *self.schema.obs_shape),
            np.asarray(rew, dtype=np.float32).reshape(1),
            np.asarray(term, dtype=np.bool_).reshape(1),
            np.asarray(trun, dtype=np.bool_).reshape(1),
        )

    def step_vector(
        self,
        runtime: Runtime,
        actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step every lane of one exact-capacity vector runtime."""
        return self.transition_obs(cast(VectorRuntime, runtime.env).step(actions))

    def host_step(
        self,
        runtime_ids: npt.ArrayLike,
        tokens: npt.ArrayLike,
        actions: npt.ArrayLike,
    ) -> StepBuffers[np.ndarray]:
        """Validate and execute one complete runtime transaction."""
        ids = np.asarray(runtime_ids, dtype=np.int32)
        batch_shape = tuple(ids.shape)
        token_values = np.asarray(tokens, dtype=np.int32).reshape(-1)
        action_values = np.asarray(actions, dtype=self.schema.act_dtype).reshape(
            len(token_values),
            *self.schema.act_shape,
        )
        runtime = _lookup_runtime(ids, self.spec)
        if len(token_values) == 0 or np.any(token_values != token_values[0]):
            raise RuntimeError(f"one callback received mixed runtime tokens: {tokens}")
        self.validate_batch(runtime, batch_shape)

        with runtime.lock:
            self.validate_runtime(runtime, int(token_values[0]), "step")
            if runtime.token == self.TOKEN_MAX:
                raise OverflowError("Gymnasium runtime token exhausted")
            if runtime.vectorized:
                nobs, reset_obs, rew, term, trun = self.step_vector(
                    runtime,
                    action_values,
                )
            else:
                nobs, reset_obs, rew, term, trun = self.step_scalar(
                    runtime,
                    action_values[0],
                )
            runtime.token += 1
            next_tokens = np.full(len(token_values), runtime.token, dtype=np.int32)

        obs_shape = (*batch_shape, *self.schema.obs_shape)
        return (
            nobs.reshape(obs_shape),
            reset_obs.reshape(obs_shape),
            rew.reshape(batch_shape),
            term.reshape(batch_shape),
            trun.reshape(batch_shape),
            next_tokens.reshape(batch_shape),
        )

    def step_spec(
        self,
        batch_shape: tuple[int, ...],
    ) -> StepBuffers[jax.ShapeDtypeStruct]:
        """Describe callback outputs for a complete mapped state shape."""
        obs_shape = (*batch_shape, *self.schema.obs_shape)
        return (
            jax.ShapeDtypeStruct(obs_shape, self.schema.obs_dtype),
            jax.ShapeDtypeStruct(obs_shape, self.schema.obs_dtype),
            jax.ShapeDtypeStruct(batch_shape, jnp.float32),
            jax.ShapeDtypeStruct(batch_shape, jnp.bool_),
            jax.ShapeDtypeStruct(batch_shape, jnp.bool_),
            jax.ShapeDtypeStruct(batch_shape, jnp.int32),
        )

    def jax_step(
        self,
        runtime: jax.Array,
        token: jax.Array,
        action: jax.Array,
    ) -> Step[jax.Array]:
        """Step one scalar state or a completely accumulated mapped state."""
        act_rank = len(self.schema.act_shape)
        batch_shape = action.shape[:-act_rank] if act_rank else action.shape
        if runtime.shape != batch_shape or token.shape != batch_shape:
            raise ValueError("runtime State and action must have matching batch shapes")
        nobs, reset_obs, rew, term, trun, next_token = cast(
            StepBuffers[jax.Array],
            io_callback(
                self.host_step,
                self.step_spec(batch_shape),
                runtime,
                token,
                action,
                ordered=False,
            ),
        )
        return self.Step(
            nobs=nobs,
            rew=rew,
            term=term,
            trun=trun,
            reset_obs=reset_obs,
            token=next_token,
        )

    def jax_step_vmap(
        self,
        axis_size: int,
        in_batched: list[bool],
        runtimes: jax.Array,
        tokens: jax.Array,
        actions: jax.Array,
    ) -> tuple[Step[jax.Array], Step[bool]]:
        """Accumulate every enclosing mapped axis into one backend step."""
        del axis_size
        if in_batched != [True, True, True]:
            raise ValueError("runtime State and action must share every vmap axis")
        return self.step(runtimes, tokens, actions), self.Step(
            nobs=True,
            rew=True,
            term=True,
            trun=True,
            reset_obs=True,
            token=True,
        )


@dataclass
class GymEnv:
    """Configured adapter for a mutable Gymnasium-like backend.

    Public dataclasses:
        State: Runtime reference and current observation data.
        Step: One environment transition.

    Public methods:
        init: Create one runtime for the complete transformed key shape.
        reset: Select the current episode's initial observation.
        step: Advance one logical environment.
        obs: Read the current observation.
        close: Close the runtime referenced by state.
    """

    _spec: RuntimeSpec
    _io: CallbackIO
    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype
    act_shape: tuple[int, ...]
    act_dtype: np.dtype

    @dataclass
    class State:
        """Dynamic per-environment state.

        Attributes:
            runtime: Process-local runtime identity, scalar before mapping.
            token: Sequencing value with the same leading shape as ``runtime``.
            obs: Current observation, followed by ``obs_shape``.
            reset_obs: Most recent SAME_STEP reset observation.
        """

        runtime: jax.Array
        token: jax.Array
        obs: chex.Array
        reset_obs: chex.Array

    @dataclass
    class Step:
        """Per-environment transition returned by :meth:`step`."""

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def init(self, key: chex.PRNGKey) -> GymEnv.State:
        """Create one runtime for this call's complete transformed key shape."""
        result = self._io.init(key)
        return self.State(
            runtime=result.runtime,
            token=result.token,
            obs=result.obs,
            reset_obs=result.reset_obs,
        )

    def step(
        self,
        key: chex.PRNGKey,
        act: chex.Array,
        state: State,
    ) -> tuple[Step, State]:
        """Step one logical environment through the external runtime."""
        del key
        result = self._io.step(
            state.runtime,
            state.token,
            cast(jax.Array, act),
        )
        done = jnp.logical_or(result.term, result.trun)
        return (
            self.Step(
                nobs=result.nobs,
                rew=result.rew,
                term=result.term,
                trun=result.trun,
            ),
            replace(
                state,
                token=result.token,
                obs=result.nobs,
                reset_obs=jnp.where(done, result.reset_obs, state.reset_obs),
            ),
        )

    def reset(self, key: chex.PRNGKey, state: State) -> tuple[chex.Array, State]:
        """Select the cached SAME_STEP reset observation."""
        del key
        return state.reset_obs, replace(state, obs=state.reset_obs)

    def obs(self, state: State) -> chex.Array:
        """Return the current observation."""
        return state.obs

    def close(self, state: State) -> None:
        """Close every runtime referenced by state outside JAX transformations.

        A stale token remains valid for cleanup. The adapter configuration must
        still match the runtime that produced the state.
        """
        runtime = jax.device_get(state.runtime)
        _close_runtime_ids(np.asarray(runtime, dtype=np.int32), self._spec)


def make(
    name: str,
    *,
    async_envs: bool = False,
    **kwargs: Any,
) -> GymEnv:
    """Configure a Gymnasium environment whose runtime axes come from ``vmap``."""
    if "num_envs" in kwargs:
        raise TypeError("num_envs is defined by mapping env.init")

    probe = make_scalar_runtime(name, **kwargs)
    try:
        observation_space = probe.observation_space
        action_space = probe.action_space
    finally:
        probe.close()

    def factory(request: RuntimeRequest) -> tuple[BackendRuntime, np.ndarray]:
        """Create and reset one exact-capacity scalar or vector runtime."""
        if request.vectorized:
            runtime_env: BackendRuntime = make_vector_runtime(
                name,
                num_envs=request.capacity,
                vectorization_mode="async" if async_envs else "sync",
                vector_kwargs={"autoreset_mode": "SameStep"},
                **kwargs,
            )
            try:
                obs, _ = runtime_env.reset(seed=list(request.seeds))
            except Exception:
                runtime_env.close()
                raise
        else:
            runtime_env = make_scalar_runtime(name, **kwargs)
            try:
                obs, _ = runtime_env.reset(seed=request.seeds[0])
            except Exception:
                runtime_env.close()
                raise
        return runtime_env, np.asarray(obs)

    return from_factory(
        factory,
        compatibility_key=("gymnasium", name, async_envs, config_key(kwargs)),
        always_vectorized=async_envs,
        observation_space=observation_space,
        action_space=action_space,
    )


def from_factory(
    factory: RuntimeFactory,
    *,
    compatibility_key: Hashable,
    observation_space: ArraySpace,
    action_space: ArraySpace,
    always_vectorized: bool = False,
) -> GymEnv:
    """Build an adapter from an exact-capacity runtime factory.

    The factory receives the complete mapped capacity, one seed per environment,
    and scalar/vector mode. Vector runtimes must use SAME_STEP autoreset.
    ``compatibility_key`` identifies adapters whose states may share a runtime.
    """
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
    spec = RuntimeSpec(
        key=compatibility_key,
        schema=schema,
        always_vectorized=always_vectorized,
    )
    return GymEnv(
        _spec=spec,
        _io=CallbackIO(factory, spec),
        obs_shape=schema.obs_shape,
        obs_dtype=schema.obs_dtype,
        act_shape=schema.act_shape,
        act_dtype=schema.act_dtype,
    )
