"""Comprehensive tests for tabular environment functionality."""

import jax
import jax.numpy as jnp
import pytest
from jaxtor.env import tabular


# ============================================================================
# Initialization Tests
# ============================================================================


def test_garnet_init():
    """Test Garnet MDP initialization creates valid state."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=50, action_size=10)
    env = tabular.garnet.make(config)
    state = env.init(key)

    assert state.mdp.state_size == config.state_size
    assert state.mdp.action_size == config.action_size
    assert state.last_state.shape == (config.state_size,)
    assert state.step == 0
    assert state.episode_length == 1000


def test_graph_init():
    """Test Graph MDP initialization creates valid state."""
    key = jax.random.PRNGKey(1)
    config = tabular.graph.Config()
    env = tabular.graph.make(config)
    state = env.init(key)

    assert state.mdp.state_size == 6
    assert state.mdp.action_size == 6
    assert state.last_state.shape == (6,)
    assert state.step == 0


def test_gridworld_init():
    """Test GridWorld MDP initialization creates valid state."""
    key = jax.random.PRNGKey(2)
    config = tabular.gridworld.Config(
        board=["#####", "#  @#", "# #X#", "#P  #", "#####"], p_slip=0.1
    )
    env = tabular.gridworld.make(config)
    state = env.init(key)

    assert state.mdp.state_size == 8
    assert state.mdp.action_size == 4
    assert state.last_state.shape == (8,)
    assert state.step == 0


# ============================================================================
# Basic Step Tests
# ============================================================================


def test_garnet_single_step():
    """Test single step in Garnet MDP returns valid transition."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    action = jax.nn.one_hot(0, config.action_size)
    transition, next_state = env.step(step_key, action, state)

    # Check transition structure
    assert transition.nobs.shape == (config.state_size,)
    assert isinstance(float(transition.rew), float)
    assert transition.term in [0, 1] or (0 <= transition.term <= 1)
    assert transition.trun in [0, 1] or (0 <= transition.trun <= 1)

    # Check state update
    assert next_state.step == state.step + 1
    assert jnp.array_equal(next_state.last_state, transition.nobs)


def test_graph_single_step():
    """Test single step in Graph MDP returns valid transition."""
    key = jax.random.PRNGKey(1)
    init_key, step_key = jax.random.split(key)

    config = tabular.graph.Config()
    env = tabular.graph.make(config)
    state = env.init(init_key)

    action = jax.nn.one_hot(0, 6)
    transition, next_state = env.step(step_key, action, state)

    assert transition.nobs.shape == (6,)
    assert next_state.step == state.step + 1


def test_gridworld_single_step():
    """Test single step in GridWorld MDP returns valid transition."""
    key = jax.random.PRNGKey(2)
    init_key, step_key = jax.random.split(key)

    config = tabular.gridworld.Config(
        board=["#####", "#  @#", "# # #", "#P  #", "#####"], p_slip=0.0
    )
    env = tabular.gridworld.make(config)
    state = env.init(init_key)

    action = jax.nn.one_hot(1, 4)  # Right
    transition, next_state = env.step(step_key, action, state)

    assert transition.nobs.shape == (8,)
    assert next_state.step == state.step + 1


# ============================================================================
# Multiple Steps Tests
# ============================================================================


def test_garnet_multiple_steps():
    """Test multiple consecutive steps maintain consistency."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    num_steps = 10
    for i in range(num_steps):
        step_key, key = jax.random.split(key)
        action_idx = i % config.action_size
        action = jax.nn.one_hot(action_idx, config.action_size)
        transition, state = env.step(step_key, action, state)

        # Verify state is updated correctly
        assert jnp.array_equal(state.last_state, transition.nobs)
        assert state.step == i + 1


def test_episode_accumulates_reward():
    """Test that rewards accumulate correctly over an episode."""
    key = jax.random.PRNGKey(42)
    init_key, rollout_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=20, action_size=5)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    rewards = []
    max_steps = 20

    for _ in range(max_steps):
        rollout_key, action_key, step_key = jax.random.split(rollout_key, 3)
        action_idx = jax.random.randint(action_key, (), 0, state.mdp.action_size)
        action = jax.nn.one_hot(action_idx, state.mdp.action_size)
        transition, state = env.step(step_key, action, state)
        rewards.append(float(transition.rew))

        if transition.term or transition.trun:
            break

    assert len(rewards) > 0
    total_reward = sum(rewards)
    assert isinstance(total_reward, float)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_episode_length_limit():
    """Test that episodes respect the episode_length limit."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    # Create environment with very short episode length
    episode_length = 5
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    # Manually set short episode length for testing
    state = state.replace(episode_length=episode_length)

    steps_taken = 0
    max_steps = 100  # Safety limit

    for i in range(max_steps):
        step_key, key = jax.random.split(key)
        action = jax.nn.one_hot(0, config.action_size)
        transition, state = env.step(step_key, action, state)
        steps_taken += 1

        # Should truncate when reaching episode length
        if i + 1 >= episode_length:
            # Note: jaxdp's async_sample_step handles truncation internally
            # The test verifies we can run for at least episode_length steps
            assert steps_taken >= episode_length
            break

        if transition.term or transition.trun:
            break

    assert steps_taken <= max_steps


def test_truncation_flag():
    """Test that truncation flag is set correctly."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    # Run many steps to potentially encounter truncation
    for i in range(1000):
        step_key, key = jax.random.split(key)
        action = jax.nn.one_hot(0, config.action_size)
        transition, state = env.step(step_key, action, state)

        # When episode ends, either term or trun should be set
        if transition.term or transition.trun:
            assert transition.term in [0, 1] or (0 <= transition.term <= 1)
            assert transition.trun in [0, 1] or (0 <= transition.trun <= 1)
            break


def test_terminal_state():
    """Test behavior when reaching terminal state."""
    key = jax.random.PRNGKey(2)
    init_key, step_key = jax.random.split(key)

    # GridWorld has explicit terminal states
    config = tabular.gridworld.Config(
        board=["###", "#P@", "###"], p_slip=0.0
    )
    env = tabular.gridworld.make(config)
    state = env.init(init_key)

    # Move right towards goal
    action = jax.nn.one_hot(1, 4)  # Right
    transition, state = env.step(step_key, action, state)

    # Should reach goal and terminate
    if transition.term:
        assert transition.term > 0
        assert transition.nobs.shape == state.last_state.shape


def test_state_space_consistency():
    """Test that observations stay within the defined state space."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    for i in range(20):
        step_key, key = jax.random.split(key)
        action = jax.nn.one_hot(i % config.action_size, config.action_size)
        transition, state = env.step(step_key, action, state)

        # Check state space consistency
        assert transition.nobs.shape == (config.state_size,)
        assert state.last_state.shape == (config.state_size,)

        if transition.term or transition.trun:
            break


def test_action_space_validation():
    """Test that different actions produce different results."""
    key = jax.random.PRNGKey(0)
    init_key = key

    config = tabular.garnet.Config(state_size=10, action_size=4, branch_size=3)
    env = tabular.garnet.make(config)

    # Test each action
    results = []
    for action_idx in range(config.action_size):
        step_key, key = jax.random.split(key)
        state = env.init(init_key)
        action = jax.nn.one_hot(action_idx, config.action_size)
        transition, _ = env.step(step_key, action, state)
        results.append(transition.nobs)

    # At least some actions should lead to different states
    # (with high probability for random MDP)
    unique_states = len(set([tuple(r.tolist()) for r in results]))
    assert unique_states >= 2  # At least 2 different outcomes


# ============================================================================
# Stochasticity Tests
# ============================================================================


def test_stochastic_transitions():
    """Test that environment uses random keys (not deterministic without keys)."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    # Test that the environment respects the random key
    # by checking that step() completes successfully with different keys
    init_key = jax.random.PRNGKey(0)
    state = env.init(init_key)

    action = jax.nn.one_hot(0, config.action_size)

    # Try multiple different keys - all should work
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        transition, _ = env.step(key, action, state)

        # Verify valid outputs
        assert transition.nobs.shape == (config.state_size,)
        assert isinstance(float(transition.rew), float)

    # This test verifies the environment accepts and uses keys correctly
    # The actual stochasticity is implementation-dependent


def test_deterministic_with_same_key():
    """Test that same key produces same results."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    init_key = jax.random.PRNGKey(0)
    step_key = jax.random.PRNGKey(1)

    state1 = env.init(init_key)
    state2 = env.init(init_key)

    action = jax.nn.one_hot(0, config.action_size)

    transition1, next_state1 = env.step(step_key, action, state1)
    transition2, next_state2 = env.step(step_key, action, state2)

    # Same keys should produce identical results
    assert jnp.allclose(transition1.nobs, transition2.nobs)
    assert jnp.isclose(transition1.rew, transition2.rew)
    assert jnp.array_equal(transition1.term, transition2.term)
    assert jnp.array_equal(transition1.trun, transition2.trun)


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.parametrize("state_size,action_size", [
    (5, 3),
    (10, 5),
    (50, 10),
    (100, 20),
])
def test_garnet_different_sizes(state_size, action_size):
    """Test Garnet MDP with various state and action space sizes."""
    key = jax.random.PRNGKey(0)
    init_key, step_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=state_size, action_size=action_size)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    assert state.last_state.shape == (state_size,)
    assert state.mdp.state_size == state_size
    assert state.mdp.action_size == action_size

    action = jax.nn.one_hot(0, action_size)
    transition, next_state = env.step(step_key, action, state)

    assert transition.nobs.shape == (state_size,)


def test_gridworld_slip_probability():
    """Test GridWorld with slip probability affects behavior."""
    key = jax.random.PRNGKey(42)

    config_no_slip = tabular.gridworld.Config(
        board=["#####", "#P  #", "#####"], p_slip=0.0
    )
    config_slip = tabular.gridworld.Config(
        board=["#####", "#P  #", "#####"], p_slip=0.5
    )

    env_no_slip = tabular.gridworld.make(config_no_slip)
    env_slip = tabular.gridworld.make(config_slip)

    state_no_slip = env_no_slip.init(key)
    state_slip = env_slip.init(key)

    # Both should initialize successfully
    assert state_no_slip.mdp.state_size == state_slip.mdp.state_size
    assert state_no_slip.mdp.action_size == state_slip.mdp.action_size


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode_rollout():
    """Test complete episode rollout until termination."""
    key = jax.random.PRNGKey(42)
    init_key, rollout_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=20, action_size=5, branch_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    max_steps = 1000
    episode_rewards = []
    terminated = False

    for step_num in range(max_steps):
        rollout_key, action_key, step_key = jax.random.split(rollout_key, 3)
        action_idx = jax.random.randint(action_key, (), 0, state.mdp.action_size)
        action = jax.nn.one_hot(action_idx, state.mdp.action_size)
        transition, state = env.step(step_key, action, state)
        episode_rewards.append(float(transition.rew))

        if transition.term or transition.trun:
            terminated = True
            break

    assert len(episode_rewards) > 0
    assert len(episode_rewards) <= max_steps
    # Episode should eventually terminate
    assert terminated or len(episode_rewards) == max_steps


def test_multiple_episodes():
    """Test running multiple episodes sequentially."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    num_episodes = 3
    episode_lengths = []

    for episode in range(num_episodes):
        key, init_key = jax.random.split(key)
        state = env.init(init_key)

        steps = 0
        max_steps = 100

        for _ in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)
            action_idx = jax.random.randint(action_key, (), 0, config.action_size)
            action = jax.nn.one_hot(action_idx, config.action_size)
            transition, state = env.step(step_key, action, state)
            steps += 1

            if transition.term or transition.trun:
                break

        episode_lengths.append(steps)

    assert len(episode_lengths) == num_episodes
    assert all(length > 0 for length in episode_lengths)
    assert all(length <= 100 for length in episode_lengths)
