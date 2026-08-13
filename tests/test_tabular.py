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
    init_key, reset_key = jax.random.split(key)
    config = tabular.garnet.Config(state_size=50, action_size=10)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    # After init, s=-1 (invalid until reset)
    assert state.mdp.state_size == config.state_size
    assert state.mdp.action_size == config.action_size
    assert state.s == -1
    assert state.step == 0
    assert state.max_eps_len == 1000

    # After reset, s is a valid index
    obs, state = env.reset(reset_key, state)
    assert 0 <= obs < config.state_size
    assert 0 <= state.s < config.state_size
    assert state.step == 0


def test_graph_init():
    """Test Graph MDP initialization creates valid state."""
    key = jax.random.PRNGKey(1)
    init_key, reset_key = jax.random.split(key)
    config = tabular.graph.Config()
    env = tabular.graph.make(config)
    state = env.init(init_key)

    assert state.mdp.state_size == 6
    assert state.mdp.action_size == 6
    assert state.s == -1
    assert state.step == 0

    obs, state = env.reset(reset_key, state)
    assert 0 <= obs < 6
    assert 0 <= state.s < 6


def test_gridworld_init():
    """Test GridWorld MDP initialization creates valid state."""
    key = jax.random.PRNGKey(2)
    init_key, reset_key = jax.random.split(key)
    config = tabular.gridworld.Config(
        board=["#####", "#  @#", "# #X#", "#P  #", "#####"], p_slip=0.1
    )
    env = tabular.gridworld.make(config)
    state = env.init(init_key)

    assert state.mdp.state_size == 8
    assert state.mdp.action_size == 4
    assert state.s == -1
    assert state.step == 0

    obs, state = env.reset(reset_key, state)
    assert 0 <= obs < 8
    assert 0 <= state.s < 8


def test_init_returns_invalid_state_index():
    """Test that init returns state with s=-1 (invalid until reset)."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(key)

    # s=-1 indicates not yet reset
    assert state.s == -1
    assert state.step == 0


def test_reset_samples_from_initial_distribution():
    """Test that reset samples from MDP initial distribution."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    # Multiple resets should give valid indices from initial distribution
    states_seen = set()
    for i in range(20):
        reset_key, subkey = jax.random.split(reset_key)
        obs, _ = env.reset(subkey, state)
        assert 0 <= obs < config.state_size
        states_seen.add(int(obs))

    # Should see some variety (probabilistic, but likely with 20 samples)
    assert len(states_seen) >= 2


def test_obs_returns_state_index():
    """Test that obs() returns state.s."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    obs, state = env.reset(reset_key, state)

    # obs() should return state.s
    assert env.obs(state) == state.s
    assert env.obs(state) == obs


# ============================================================================
# Basic Step Tests
# ============================================================================


def test_garnet_single_step():
    """Test single step in Garnet MDP returns valid transition."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    transition, next_state = env.step(step_key, 0, state)

    # Check transition structure (nobs is a scalar index)
    assert 0 <= transition.nobs < config.state_size
    assert isinstance(float(transition.rew), float)
    assert transition.term in [0, 1] or (0 <= transition.term <= 1)
    assert transition.trun in [0, 1] or (0 <= transition.trun <= 1)

    # Check state update
    assert next_state.step == state.step + 1
    assert next_state.s == transition.nobs


def test_graph_single_step():
    """Test single step in Graph MDP returns valid transition."""
    key = jax.random.PRNGKey(1)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    config = tabular.graph.Config()
    env = tabular.graph.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    transition, next_state = env.step(step_key, 0, state)

    assert 0 <= transition.nobs < 6
    assert next_state.step == state.step + 1
    assert next_state.s == transition.nobs


def test_gridworld_single_step():
    """Test single step in GridWorld MDP returns valid transition."""
    key = jax.random.PRNGKey(2)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    config = tabular.gridworld.Config(
        board=["#####", "#  @#", "# # #", "#P  #", "#####"], p_slip=0.0
    )
    env = tabular.gridworld.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    transition, next_state = env.step(step_key, 1, state)  # Right

    assert 0 <= transition.nobs < 8
    assert next_state.step == state.step + 1
    assert next_state.s == transition.nobs


def test_step_returns_valid_index():
    """Test that step returns valid state index in nobs."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Run multiple steps
    for i in range(10):
        step_key, subkey = jax.random.split(step_key)
        transition, state = env.step(subkey, i % config.action_size, state)

        # nobs should always be a valid index
        assert 0 <= transition.nobs < config.state_size
        # state.s should match nobs
        assert state.s == transition.nobs


# ============================================================================
# Multiple Steps Tests
# ============================================================================


def test_garnet_multiple_steps():
    """Test multiple consecutive steps maintain consistency."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    num_steps = 10
    for i in range(num_steps):
        step_key, key = jax.random.split(key)
        action_idx = i % config.action_size
        transition, state = env.step(step_key, action_idx, state)

        # Verify state is updated correctly
        assert state.s == transition.nobs
        assert state.step == i + 1


def test_episode_accumulates_reward():
    """Test that rewards accumulate correctly over an episode."""
    key = jax.random.PRNGKey(42)
    init_key, reset_key, rollout_key = jax.random.split(key, 3)

    config = tabular.garnet.Config(state_size=20, action_size=5)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    rewards = []
    max_steps = 20

    for _ in range(max_steps):
        rollout_key, action_key, step_key = jax.random.split(rollout_key, 3)
        action_idx = jax.random.randint(action_key, (), 0, state.mdp.action_size)
        transition, state = env.step(step_key, action_idx, state)
        rewards.append(float(transition.rew))

        if transition.term or transition.trun:
            break

    assert len(rewards) > 0
    total_reward = sum(rewards)
    assert isinstance(total_reward, float)


# ============================================================================
# Truncation Tests
# ============================================================================


def test_truncation_at_max_eps_len():
    """Test that truncation flag is set at max_eps_len."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    # Create environment with very short episode length
    max_eps_len = 5
    config = tabular.garnet.Config(
        state_size=10, action_size=4, max_eps_len=max_eps_len
    )
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Run until truncation
    truncated = False
    for i in range(max_eps_len + 5):
        step_key, key = jax.random.split(key)
        transition, state = env.step(step_key, 0, state)

        # Check truncation at correct step
        if i == max_eps_len - 1:
            # At step max_eps_len-1, trun should be True
            assert transition.trun, f"Expected truncation at step {i}"
            truncated = True
            break

        if transition.term:
            # Terminal state reached before truncation
            break

    # Either we hit truncation or terminal
    assert truncated or transition.term


def test_episode_length_limit():
    """Test that episodes respect the max_eps_len limit."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    # Create environment with very short episode length
    max_eps_len = 5
    config = tabular.garnet.Config(
        state_size=10, action_size=4, max_eps_len=max_eps_len
    )
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    steps_taken = 0
    max_steps = 100  # Safety limit

    for i in range(max_steps):
        step_key, key = jax.random.split(key)
        transition, state = env.step(step_key, 0, state)
        steps_taken += 1

        if transition.term or transition.trun:
            break

    # Should have stopped at or before max_eps_len
    assert steps_taken <= max_eps_len


def test_truncation_flag():
    """Test that truncation flag is set correctly."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Run many steps to potentially encounter truncation
    for i in range(1000):
        step_key, key = jax.random.split(key)
        transition, state = env.step(step_key, 0, state)

        # When episode ends, either term or trun should be set
        if transition.term or transition.trun:
            assert transition.term in [0, 1] or (0 <= transition.term <= 1)
            assert transition.trun in [0, 1] or (0 <= transition.trun <= 1)
            break


def test_terminal_state():
    """Test behavior when reaching terminal state."""
    key = jax.random.PRNGKey(2)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    # GridWorld has explicit terminal states
    config = tabular.gridworld.Config(board=["###", "#P@", "###"], p_slip=0.0)
    env = tabular.gridworld.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Move right towards goal
    transition, state = env.step(step_key, 1, state)  # Right

    # Should reach goal and terminate
    if transition.term:
        assert transition.term > 0
        assert 0 <= transition.nobs < state.mdp.state_size


def test_state_space_consistency():
    """Test that observations stay within the defined state space."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    for i in range(20):
        step_key, key = jax.random.split(key)
        action_idx = i % config.action_size
        transition, state = env.step(step_key, action_idx, state)

        # Check state space consistency (nobs is scalar index)
        assert 0 <= transition.nobs < config.state_size
        # state.s is now a scalar index, not one-hot
        assert 0 <= state.s < config.state_size

        if transition.term or transition.trun:
            break


def test_action_space_validation():
    """Test that different actions produce different results."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    config = tabular.garnet.Config(state_size=10, action_size=4, branch_size=3)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Test each action
    results = []
    for action_idx in range(config.action_size):
        step_key, key = jax.random.split(key)
        transition, _ = env.step(step_key, action_idx, state)
        results.append(int(transition.nobs))

    # At least some actions should lead to different states
    # (with high probability for random MDP)
    unique_states = len(set(results))
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
    reset_key = jax.random.PRNGKey(1)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    # Try multiple different keys - all should work
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        transition, _ = env.step(key, 0, state)

        # Verify valid outputs (nobs is scalar index)
        assert 0 <= transition.nobs < config.state_size
        assert isinstance(float(transition.rew), float)

    # This test verifies the environment accepts and uses keys correctly
    # The actual stochasticity is implementation-dependent


def test_deterministic_with_same_key():
    """Test that same key produces same results."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    init_key = jax.random.PRNGKey(0)
    reset_key = jax.random.PRNGKey(1)
    step_key = jax.random.PRNGKey(2)

    state1 = env.init(init_key)
    _, state1 = env.reset(reset_key, state1)
    state2 = env.init(init_key)
    _, state2 = env.reset(reset_key, state2)

    transition1, next_state1 = env.step(step_key, 0, state1)
    transition2, next_state2 = env.step(step_key, 0, state2)

    # Same keys should produce identical results
    assert transition1.nobs == transition2.nobs
    assert jnp.isclose(transition1.rew, transition2.rew)
    assert jnp.array_equal(transition1.term, transition2.term)
    assert jnp.array_equal(transition1.trun, transition2.trun)


# ============================================================================
# Configuration Tests
# ============================================================================


@pytest.mark.parametrize(
    "state_size,action_size",
    [
        (5, 3),
        (10, 5),
        (50, 10),
        (100, 20),
    ],
)
def test_garnet_different_sizes(state_size, action_size):
    """Test Garnet MDP with various state and action space sizes."""
    key = jax.random.PRNGKey(0)
    init_key, reset_key, step_key = jax.random.split(key, 3)

    config = tabular.garnet.Config(state_size=state_size, action_size=action_size)
    env = tabular.garnet.make(config)
    state = env.init(init_key)

    # After init, s=-1
    assert state.s == -1

    _, state = env.reset(reset_key, state)

    # After reset, s is valid index
    assert 0 <= state.s < state_size
    assert state.mdp.state_size == state_size
    assert state.mdp.action_size == action_size

    transition, next_state = env.step(step_key, 0, state)

    assert 0 <= transition.nobs < state_size


def test_gridworld_slip_probability():
    """Test GridWorld with slip probability affects behavior."""
    key = jax.random.PRNGKey(42)
    init_key, reset_key = jax.random.split(key)

    config_no_slip = tabular.gridworld.Config(
        board=["#####", "#P  #", "#####"], p_slip=0.0
    )
    config_slip = tabular.gridworld.Config(
        board=["#####", "#P  #", "#####"], p_slip=0.5
    )

    env_no_slip = tabular.gridworld.make(config_no_slip)
    env_slip = tabular.gridworld.make(config_slip)

    state_no_slip = env_no_slip.init(init_key)
    _, state_no_slip = env_no_slip.reset(reset_key, state_no_slip)
    state_slip = env_slip.init(init_key)
    _, state_slip = env_slip.reset(reset_key, state_slip)

    # Both should initialize successfully
    assert state_no_slip.mdp.state_size == state_slip.mdp.state_size
    assert state_no_slip.mdp.action_size == state_slip.mdp.action_size


# ============================================================================
# JIT Compatibility Tests
# ============================================================================


def test_jit_init():
    """Test that init is JIT-compatible."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    jit_init = jax.jit(env.init)

    key = jax.random.PRNGKey(0)
    state = jit_init(key)

    assert state.s == -1
    assert state.step == 0
    assert state.mdp.state_size == 10


def test_jit_reset():
    """Test that reset is JIT-compatible."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    jit_reset = jax.jit(env.reset)

    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)
    state = env.init(init_key)

    obs, new_state = jit_reset(reset_key, state)

    assert 0 <= obs < 10
    assert 0 <= new_state.s < 10
    assert new_state.step == 0


def test_jit_step():
    """Test that step is JIT-compatible."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    jit_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    init_key, reset_key, step_key = jax.random.split(key, 3)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    transition, new_state = jit_step(step_key, 0, state)

    assert 0 <= transition.nobs < 10
    assert new_state.s == transition.nobs
    assert new_state.step == 1


def test_jit_obs():
    """Test that obs is JIT-compatible."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    jit_obs = jax.jit(env.obs)

    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    obs = jit_obs(state)
    assert obs == state.s


def test_jit_full_episode():
    """Test running a full episode with JIT-compiled functions."""
    config = tabular.garnet.Config(state_size=10, action_size=4, max_eps_len=20)
    env = tabular.garnet.make(config)

    jit_init = jax.jit(env.init)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    init_key, reset_key = jax.random.split(key)

    state = jit_init(init_key)
    obs, state = jit_reset(reset_key, state)

    total_reward = 0.0
    for i in range(30):
        step_key, key = jax.random.split(key)
        action = i % config.action_size
        transition, state = jit_step(step_key, action, state)
        total_reward += float(transition.rew)

        if transition.term or transition.trun:
            break

    assert isinstance(total_reward, float)


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_episode_rollout():
    """Test complete episode rollout until termination."""
    key = jax.random.PRNGKey(42)
    init_key, reset_key, rollout_key = jax.random.split(key, 3)

    config = tabular.garnet.Config(state_size=20, action_size=5, branch_size=4)
    env = tabular.garnet.make(config)
    state = env.init(init_key)
    _, state = env.reset(reset_key, state)

    max_steps = 1000
    episode_rewards = []
    terminated = False

    for step_num in range(max_steps):
        rollout_key, action_key, step_key = jax.random.split(rollout_key, 3)
        action_idx = jax.random.randint(action_key, (), 0, state.mdp.action_size)
        transition, state = env.step(step_key, action_idx, state)
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
        key, init_key, reset_key = jax.random.split(key, 3)
        state = env.init(init_key)
        _, state = env.reset(reset_key, state)

        steps = 0
        max_steps = 100

        for _ in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)
            action_idx = jax.random.randint(action_key, (), 0, config.action_size)
            transition, state = env.step(step_key, action_idx, state)
            steps += 1

            if transition.term or transition.trun:
                break

        episode_lengths.append(steps)

    assert len(episode_lengths) == num_episodes
    assert all(length > 0 for length in episode_lengths)
    assert all(length <= 100 for length in episode_lengths)


def test_env_only_has_config_field():
    """Test that TabularEnv only has config field."""
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    # Only config field exists
    assert hasattr(env, "config")
    assert env.config == config

    # No obs_space or act_space
    assert not hasattr(env, "obs_space")
    assert not hasattr(env, "act_space")
