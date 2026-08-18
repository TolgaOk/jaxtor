"""Reset and sampler contracts for the Gymnax adapter."""

import chex
import jax
import jax.numpy as jnp
import pytest

from jaxtor.sampler import Mc, VecMc

gymnax = pytest.importorskip("jaxtor.env.gymnax")

pytestmark = pytest.mark.backend


def test_reset_replaces_dirty_state_and_matches_observation():
    """An explicit reset clears elapsed time and returns the stored observation."""
    env = gymnax.make("CartPole-v1")
    init_key, step_key, reset_key = jax.random.split(jax.random.key(0), 3)
    state = env.init(init_key)
    _, dirty = env.step(step_key, jnp.array(0), state)

    obs, reset = jax.jit(env.reset)(reset_key, dirty)

    assert dirty.env.time == 1
    assert reset.env.time == 0
    assert jnp.array_equal(obs, env.obs(reset))
    assert env.env.observation_space(env.params).contains(obs)
    assert env.env.state_space(env.params).contains(reset.env)


def test_step_matches_the_wrapped_gymnax_transition():
    """The adapter changes boundary semantics, not Gymnax transition data."""
    env = gymnax.make("CartPole-v1")
    init_key, step_key = jax.random.split(jax.random.key(4))
    state = env.init(init_key)
    action = jnp.array(1)
    obs, raw_state, rew, done, _ = env.env.step_env(
        step_key,
        state.env,
        action,
        env.params,
    )
    expected_trun = raw_state.time >= env.params.max_steps_in_episode
    expected_term = done & ~expected_trun

    transition, next_state = jax.jit(env.step)(step_key, action, state)

    assert jnp.allclose(transition.nobs, obs)
    assert jnp.allclose(transition.rew, rew)
    assert transition.term == expected_term
    assert transition.trun == expected_trun
    chex.assert_trees_all_close(next_state.env, raw_state)


def test_mc_boundary_retains_successor_and_carries_reset_observation():
    """A truncated transition and the following reset observation stay distinct."""
    env = gymnax.make("CartPole-v1", max_steps_in_episode=1)
    mc = Mc(max_eps_len=1, env=env)
    key = jax.random.key(1)
    state = mc.init(key, env.init(key))

    transition, state = jax.jit(mc.sample)(jnp.array(0), state)
    reset_obs = state.last_obs
    following, state = jax.jit(mc.sample)(jnp.array(0), state)

    assert transition.trun
    assert state.eps_idx == 0
    assert jnp.array_equal(reset_obs, following.obs)
    assert jnp.array_equal(state.last_obs, env.obs(state.env))


def test_vecmc_resets_every_boundary_lane_under_jit():
    """Vectorized Gymnax states reset independently without losing batch axes."""
    n_envs = 3
    env = gymnax.make("CartPole-v1", max_steps_in_episode=1)
    mc = VecMc(mc=Mc(max_eps_len=1, env=env))
    key = jax.random.key(2)
    keys = jax.random.split(key, n_envs)
    state = mc.init(keys, jax.vmap(env.init)(keys))

    transition, state = jax.jit(mc.sample)(
        jnp.zeros(n_envs, dtype=jnp.int32),
        state,
    )

    assert jnp.all(transition.trun)
    assert jnp.all(state.eps_idx == 0)
    assert jnp.array_equal(state.last_obs, jax.vmap(env.obs)(state.env))
