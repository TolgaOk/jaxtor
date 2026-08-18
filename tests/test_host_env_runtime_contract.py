"""Black-box contract tests for host-backed environment runtimes."""

import math
from contextlib import ExitStack
from unittest.mock import patch

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from jaxtor.env import gymnasium
from jaxtor.sampler import Mc, VecMc


def _keys(seed: int, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.split(jax.random.key(seed), math.prod(shape)).reshape(shape)


def _ready(value):
    return jax.block_until_ready(value)


@pytest.mark.backend
def test_nested_vmap_is_lazy_and_compiled_calls_route_independent_runtimes():
    env = gymnasium.make("CartPole-v1")
    equivalent_env = gymnasium.make("CartPole-v1")
    keys_2d = _keys(0, (2, 3))

    with patch.object(gym, "make_vec", wraps=gym.make_vec) as make_vec:
        mapped_init = jax.vmap(jax.vmap(env.init))
        compiled_init = jax.jit(mapped_init).lower(keys_2d).compile()

        # Transforming and compiling the initializer must not allocate a runtime.
        assert make_vec.call_count == 0

        first = _ready(compiled_init(keys_2d))
        second = _ready(compiled_init(_keys(1, (2, 3))))

        assert [call.kwargs["num_envs"] for call in make_vec.call_args_list] == [6, 6]
        assert env.obs(first).shape == (2, 3, 4)
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(first))

    actions_2d = jnp.zeros((2, 3), dtype=jnp.int32)
    step_keys_2d = _keys(2, (2, 3))
    compiled_step = jax.jit(jax.vmap(jax.vmap(equivalent_env.step)))

    with ExitStack() as cleanup:
        cleanup.callback(env.close, first)
        cleanup.callback(equivalent_env.close, second)
        first_step, first_next = _ready(
            compiled_step(step_keys_2d, actions_2d, first)
        )
        second_step, second_next = _ready(
            compiled_step(_keys(3, (2, 3)), actions_2d, second)
        )

        assert first_step.nobs.shape == (2, 3, 4)
        assert second_step.rew.shape == (2, 3)

    # A different logical shape gets its own runtime and correspondingly mapped
    # step, without changing the component configuration.
    keys_1d = _keys(4, (4,))
    state_1d = _ready(jax.jit(jax.vmap(env.init))(keys_1d))
    try:
        _, next_1d = _ready(
            jax.jit(jax.vmap(equivalent_env.step))(
                _keys(5, (4,)), jnp.zeros((4,), dtype=jnp.int32), state_1d
            )
        )
        assert env.obs(next_1d).shape == (4, 4)
    finally:
        env.close(state_1d)


@pytest.mark.backend
def test_state_shape_configuration_and_lifecycle_are_enforced():
    env = gymnasium.make("CartPole-v1")
    incompatible_env = gymnasium.make("CartPole-v1", max_episode_steps=17)

    nested_keys = _keys(10, (2, 3))
    nested_state = _ready(jax.vmap(jax.vmap(env.init))(nested_keys))
    sliced_state = jax.tree.map(lambda leaf: leaf[0], nested_state)

    try:
        with pytest.raises(jax.errors.JaxRuntimeError, match="batch shape"):
            _ready(
                jax.jit(jax.vmap(env.step))(
                    _keys(11, (3,)),
                    jnp.zeros((3,), dtype=jnp.int32),
                    sliced_state,
                )
            )

        with pytest.raises(
            jax.errors.JaxRuntimeError,
            match="incompatible environment",
        ):
            _ready(
                jax.jit(jax.vmap(jax.vmap(incompatible_env.step)))(
                    _keys(12, (2, 3)),
                    jnp.zeros((2, 3), dtype=jnp.int32),
                    nested_state,
                )
            )
    finally:
        env.close(nested_state)

    original = _ready(env.init(jax.random.key(20)))
    compiled_step = jax.jit(env.step)
    _, current = _ready(
        compiled_step(jax.random.key(21), jnp.asarray(0, dtype=jnp.int32), original)
    )

    closed = False
    try:
        with pytest.raises(
            (ValueError, jax.errors.JaxRuntimeError),
            match="out-of-order|forked",
        ):
            _ready(
                compiled_step(
                    jax.random.key(22),
                    jnp.asarray(0, dtype=jnp.int32),
                    original,
                )
            )

        # Close is deliberately allowed to use an older token.
        env.close(original)
        closed = True

        with pytest.raises(
            (ValueError, jax.errors.JaxRuntimeError),
            match="closed or unknown",
        ):
            _ready(
                compiled_step(
                    jax.random.key(23),
                    jnp.asarray(0, dtype=jnp.int32),
                    current,
                )
            )
    finally:
        if not closed:
            env.close(original)


@pytest.mark.backend
@pytest.mark.integration
def test_vec_mc_threads_each_lane_of_a_mapped_environment_state():
    env = gymnasium.make("CartPole-v1")
    env_state = _ready(jax.jit(jax.vmap(env.init))(_keys(30, (3,))))
    mc = Mc(max_eps_len=8, env=env)
    vec_mc = VecMc(mc=mc)

    try:
        sampler_state = _ready(
            jax.jit(vec_mc.init)(_keys(31, (3,)), env_state)
        )

        # A lane-wise composition retains one mapped axis; broadcasting the
        # complete environment state would introduce a second axis here.
        assert vec_mc.observe(sampler_state).shape == (3, 4)
        assert env.obs(sampler_state.env).shape == (3, 4)

        transition, sampler_next = _ready(
            jax.jit(vec_mc.sample)(
                jnp.zeros((3,), dtype=jnp.int32), sampler_state
            )
        )

        assert transition.obs.shape == (3, 4)
        assert transition.nobs.shape == (3, 4)
        assert transition.rew.shape == (3,)
        assert env.obs(sampler_next.env).shape == (3, 4)
    finally:
        env.close(env_state)


@pytest.mark.backend
def test_envpool_uses_mapped_init_shape_as_vector_capacity():
    envpool = pytest.importorskip("jaxtor.env.envpool", exc_type=ImportError)
    env = envpool.make("CartPole-v1")
    state = _ready(jax.jit(jax.vmap(env.init))(_keys(40, (4,))))

    try:
        step, next_state = _ready(
            jax.jit(jax.vmap(env.step))(
                _keys(41, (4,)), jnp.zeros((4,), dtype=jnp.int32), state
            )
        )
        assert env.obs(state).shape == (4, 4)
        assert step.nobs.shape == (4, 4)
        assert env.obs(next_state).shape == (4, 4)
    finally:
        env.close(state)
