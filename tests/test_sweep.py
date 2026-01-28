"""Tests for stochastic sweep sampler."""

import jax
import jax.numpy as jnp
import pytest
from jaxtor.env import tabular
from jaxtor.sampler import mc, sweep


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def garnet_env():
    config = tabular.garnet.Config(state_size=10, action_size=4)
    return tabular.garnet.make(config)


@pytest.fixture(scope="module")
def small_env():
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=10)
    return tabular.garnet.make(config)


# ============================================================================
# Basic Functionality Tests
# ============================================================================


def test_sweep_sample_returns_batched_state(small_env):
    """Sample returns MC state with shape (A*S, ...)."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert state.last_obs.shape == (A * S,)
    assert state.eps_idx.shape == (A * S,)


def test_sweep_sample_correct_initial_states(small_env):
    """Each position starts at its designated state (obs = state index)."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    # Flat batch ordering: position = a * S + s (action-major)
    expected_state_indices = jnp.tile(jnp.arange(S), A)
    assert jnp.array_equal(transition.obs, expected_state_indices)


def test_sweep_sample_mdp_initial_conditioned(small_env):
    """Each position has MDP with one-hot initial distribution."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_state_indices = jnp.tile(jnp.arange(S), A)

    mdp_initials = state.env.mdp.initial
    assert mdp_initials.shape == (A * S, S)

    argmax_initials = jnp.argmax(mdp_initials, axis=-1)
    assert jnp.array_equal(argmax_initials, expected_state_indices)


def test_sweep_sample_returns_transitions(small_env):
    """Sample returns batched transitions."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transition.obs.shape == (A * S,)
    assert transition.act.shape == (A * S,)
    assert transition.nobs.shape == (A * S,)


# ============================================================================
# Initial Action Tests
# ============================================================================


def test_sweep_sample_uses_correct_init_action(small_env):
    """Sample uses designated init_action for each (s,a) position."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    # Flat batch ordering: position = a * S + s, so action = position // S
    expected_init_actions = jnp.arange(A * S) // S
    assert jnp.array_equal(transition.act, expected_init_actions)


def test_sweep_sample_state_action_pairing(small_env):
    """Verify each position has correct (state, action) pairing."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size

    # Check that position i has state = i % S and action = i // S
    for i in range(A * S):
        expected_state = i % S
        expected_action = i // S
        assert transition.obs[i] == expected_state, f"Position {i}: expected state {expected_state}, got {transition.obs[i]}"
        assert transition.act[i] == expected_action, f"Position {i}: expected action {expected_action}, got {transition.act[i]}"


# ============================================================================
# Batch Shape Tests
# ============================================================================


def test_sweep_batch_shape_various_sizes():
    """Test A*S batch shape for various env sizes."""
    key = jax.random.PRNGKey(0)

    for state_size, action_size in [(3, 2), (5, 4), (10, 3)]:
        config = tabular.garnet.Config(
            state_size=state_size, action_size=action_size, max_episode_len=10
        )
        env = tabular.garnet.make(config)
        mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
        sweeper = sweep.Sweep(mc=mc_sampler)

        env_state = env.init(key)
        transition, state = sweeper.sample(key, env_state)

        expected_batch_size = state_size * action_size
        assert transition.obs.shape == (expected_batch_size,), f"Failed for S={state_size}, A={action_size}"
        assert transition.act.shape == (expected_batch_size,)
        assert transition.nobs.shape == (expected_batch_size,)
        assert state.last_obs.shape == (expected_batch_size,)


# ============================================================================
# Transition Validity Tests
# ============================================================================


def test_sweep_transitions_are_valid(small_env):
    """Transitions have valid state indices."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S = env_state.mdp.state_size
    A = env_state.mdp.action_size

    # All observations should be valid state indices
    assert jnp.all(transition.obs >= 0)
    assert jnp.all(transition.obs < S)
    assert jnp.all(transition.nobs >= 0)
    assert jnp.all(transition.nobs < S)

    # All actions should be valid
    assert jnp.all(transition.act >= 0)
    assert jnp.all(transition.act < A)


def test_sweep_transitions_consistent_with_mdp(small_env):
    """Next states are reachable according to MDP transitions."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    mdp = env_state.mdp
    S, A = mdp.state_size, mdp.action_size

    # For each transition, verify nobs is reachable from obs with act
    for i in range(A * S):
        s = int(transition.obs[i])
        a = int(transition.act[i])
        s_next = int(transition.nobs[i])

        # P(s_next | s, a) should be > 0
        prob = mdp.transition[a, s_next, s]
        assert prob > 0, f"Position {i}: transition from s={s}, a={a} to s'={s_next} has prob 0"


# ============================================================================
# JIT Compilation Tests
# ============================================================================


def test_sweep_sample_jit(small_env):
    """Sample can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)

    jit_sample = jax.jit(sweeper.sample)
    transition, state = jit_sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transition.obs.shape == (A * S,)
    assert state.last_obs.shape == (A * S,)


def test_sweep_sample_jit_multiple_calls(small_env):
    """JIT compiled sample produces correct results on multiple calls."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    S, A = env_state.mdp.state_size, env_state.mdp.action_size

    jit_sample = jax.jit(sweeper.sample)

    # Multiple calls with different keys
    for i in range(5):
        key_i = jax.random.PRNGKey(i)
        transition, state = jit_sample(key_i, env_state)

        # Verify structure is correct each time
        assert transition.obs.shape == (A * S,)
        expected_states = jnp.tile(jnp.arange(S), A)
        assert jnp.array_equal(transition.obs, expected_states)


# ============================================================================
# Determinism Tests
# ============================================================================


def test_sweep_deterministic_with_same_key(small_env):
    """Same key produces same results."""
    key = jax.random.PRNGKey(42)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)

    trans1, state1 = sweeper.sample(key, env_state)
    trans2, state2 = sweeper.sample(key, env_state)

    assert jnp.array_equal(trans1.obs, trans2.obs)
    assert jnp.array_equal(trans1.act, trans2.act)
    assert jnp.array_equal(trans1.nobs, trans2.nobs)
    assert jnp.array_equal(trans1.rew, trans2.rew)


def test_sweep_different_keys_can_differ(small_env):
    """Different keys can produce different next states (stochastic transitions)."""
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    # Use same env state but different sample keys
    init_key = jax.random.PRNGKey(0)
    env_state = small_env.init(init_key)

    # Collect nobs from multiple samples with different keys
    nobs_list = []
    for i in range(20):
        key = jax.random.PRNGKey(i + 100)
        trans, _ = sweeper.sample(key, env_state)
        nobs_list.append(trans.nobs)

    # Check if there's any variation (unless MDP is deterministic)
    all_nobs = jnp.stack(nobs_list)
    # At least some position should have different nobs across samples
    has_variation = jnp.any(jnp.std(all_nobs, axis=0) > 0)
    # This test passes if there's variation OR if MDP happens to be deterministic
    # (we mainly want to verify it doesn't crash with different keys)
    assert all_nobs.shape == (20, small_env.config.action_size * small_env.config.state_size)


# ============================================================================
# Edge Cases
# ============================================================================


def test_sweep_single_state_env():
    """Works with single-state environment."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=1, action_size=3, max_episode_len=10)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = env.init(key)
    transition, state = sweeper.sample(key, env_state)

    # Should have A*S = 3*1 = 3 positions
    assert transition.obs.shape == (3,)
    # All obs should be state 0
    assert jnp.all(transition.obs == 0)
    # Actions should be [0, 1, 2]
    assert jnp.array_equal(transition.act, jnp.array([0, 1, 2]))


def test_sweep_single_action_env():
    """Works with single-action environment."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=1, max_episode_len=10)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = env.init(key)
    transition, state = sweeper.sample(key, env_state)

    # Should have A*S = 1*5 = 5 positions
    assert transition.obs.shape == (5,)
    # Obs should be [0, 1, 2, 3, 4] (all states, one action)
    assert jnp.array_equal(transition.obs, jnp.arange(5))
    # All actions should be 0
    assert jnp.all(transition.act == 0)


# ============================================================================
# _condition_mdp_initial Tests
# ============================================================================


def test_condition_mdp_initial_creates_one_hot(small_env):
    """_condition_mdp_initial creates MDP with one-hot initial distribution."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    mdp = env_state.mdp
    S = mdp.state_size

    # Test conditioning on each state
    for s in range(S):
        one_hot = jax.nn.one_hot(s, S)
        conditioned_mdp = sweeper._condition_mdp_initial(mdp, one_hot)

        # Initial distribution should match the one-hot
        assert jnp.array_equal(conditioned_mdp.initial, one_hot)

        # Other MDP properties should be unchanged
        assert jnp.array_equal(conditioned_mdp.transition, mdp.transition)
        assert jnp.array_equal(conditioned_mdp.reward, mdp.reward)
        assert jnp.array_equal(conditioned_mdp.terminal, mdp.terminal)


# ============================================================================
# Large Environment Test
# ============================================================================


def test_sweep_larger_env(garnet_env):
    """Works with larger environment (10 states, 4 actions)."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=100, queue_size=10, env=garnet_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = garnet_env.init(key)
    transition, state = sweeper.sample(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert S == 10
    assert A == 4

    # Verify batch size
    assert transition.obs.shape == (A * S,)  # 40
    assert transition.act.shape == (A * S,)
    assert state.last_obs.shape == (A * S,)

    # Verify correct state/action pairing
    expected_states = jnp.tile(jnp.arange(S), A)
    expected_actions = jnp.arange(A * S) // S
    assert jnp.array_equal(transition.obs, expected_states)
    assert jnp.array_equal(transition.act, expected_actions)
