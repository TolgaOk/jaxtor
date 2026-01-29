"""Tests for Mc sampler."""

import jax
import jax.numpy as jnp
import pytest
from jaxtor.sampler.mc import Mc
from jaxtor.env import tabular


# =============================================================================
# Mc Tests
# =============================================================================


def test_mc_init():
    """Test Mc initialization."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_episode_len=50, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    assert state.last_obs.shape == ()
    assert state.eps_rew_queue.shape == (5,)
    assert state.eps_len_queue.shape == (5,)
    assert jnp.all(jnp.isnan(state.eps_rew_queue))


def test_mc_single_sample():
    """Test single sample from the sampler."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_episode_len=50, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    action = jnp.array(1)
    transition, next_state = sampler.sample(action, state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()
    assert transition.rew.shape == ()
    assert transition.nobs.shape == ()


def test_mc_episode_statistics():
    """Test episode statistics are tracked correctly."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    sampler = Mc(max_episode_len=10, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    # Take action right (index 1) to reach goal
    action = jnp.array(1)
    transition, state = sampler.sample(action, state)

    # Episode should have completed
    assert transition.term or transition.trun
    assert not jnp.isnan(state.eps_rew_queue[0])


def test_mc_metrics():
    """Test metrics computation."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    sampler = Mc(max_episode_len=10, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    # Run a few episodes
    action = jnp.array(1)
    for _ in range(5):
        _, state = sampler.sample(action, state)

    metrics, refreshed_state = sampler.metrics(state)

    assert hasattr(metrics, "avg_eps_rew")
    assert hasattr(metrics, "avg_eps_len")
    assert jnp.all(jnp.isnan(refreshed_state.eps_rew_queue))


def test_mc_jit_compilation():
    """Verify sample() can be JIT compiled."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_episode_len=50, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    jit_sample = jax.jit(sampler.sample)

    action = jnp.array(1)
    transition, next_state = jit_sample(action, state)

    assert transition.obs.shape == ()


def test_mc_consecutive_observations():
    """Test consecutive observations match."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_episode_len=50, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    action = jnp.array(1)
    t1, state = sampler.sample(action, state)
    t2, state = sampler.sample(action, state)

    done = jnp.logical_or(t1.term, t1.trun)
    if not done:
        assert jnp.allclose(t1.nobs, t2.obs)


