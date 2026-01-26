"""Tests for MarkovChain and VecMC samplers."""

import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jaxtor.sampler import mc
from jaxtor.env import tabular


# =============================================================================
# MarkovChain Tests
# =============================================================================


def test_mc_init():
    """Test MarkovChain initialization."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
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

    sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
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

    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)
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

    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)
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

    sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
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

    sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    action = jnp.array(1)
    t1, state = sampler.sample(action, state)
    t2, state = sampler.sample(action, state)

    done = jnp.logical_or(t1.term, t1.trun)
    if not done:
        assert jnp.allclose(t1.nobs, t2.obs)


# =============================================================================
# VecMC Tests
# =============================================================================


def test_vecmc_init():
    """Test VecMC initialization produces batched states."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    env_state = env.init(key)
    state = vec_mc.init(key, env_state)

    assert state.last_obs.shape == (num_envs,)
    assert state.eps_rew_queue.shape == (num_envs, 5)
    assert state.eps_len_queue.shape == (num_envs, 5)


def test_vecmc_sample():
    """Test VecMC sampling produces batched transitions."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    env_state = env.init(key)
    state = vec_mc.init(key, env_state)

    batched_action = jnp.array([1, 2, 0, 3])
    transition, next_state = vec_mc.sample(batched_action, state)

    assert transition.obs.shape == (num_envs,)
    assert transition.act.shape == (num_envs,)
    assert transition.rew.shape == (num_envs,)
    assert transition.nobs.shape == (num_envs,)


def test_vecmc_metrics_aggregation():
    """Test VecMC.metrics returns aggregated scalar metrics."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    env_state = env.init(key)
    state = vec_mc.init(key, env_state)

    # Run multiple samples to complete episodes
    batched_action = jnp.array([1, 1, 1, 1])
    for _ in range(10):
        _, state = vec_mc.sample(batched_action, state)

    metrics, refreshed_state = vec_mc.metrics(state)

    # Metrics should be scalars (aggregated)
    assert metrics.avg_eps_rew.shape == ()
    assert metrics.avg_eps_len.shape == ()
    assert jnp.all(jnp.isnan(refreshed_state.eps_rew_queue))


def test_vecmc_jit_compilation():
    """Verify VecMC can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    env_state = env.init(key)
    state = vec_mc.init(key, env_state)

    jit_sample = jax.jit(vec_mc.sample)
    jit_metrics = jax.jit(vec_mc.metrics)

    batched_action = jnp.array([1, 2, 0, 3])
    transition, state = jit_sample(batched_action, state)
    metrics, state = jit_metrics(state)

    assert transition.obs.shape == (num_envs,)
    assert metrics.avg_eps_rew.shape == ()


def test_vecmc_different_initial_states():
    """Test VecMC environments start with different states."""
    key = jax.random.PRNGKey(0)
    num_envs = 8

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    env_state = env.init(key)
    state = vec_mc.init(key, env_state)

    # Check that not all environments start in the same state
    unique_states = len(jnp.unique(state.last_obs))
    assert unique_states > 1
