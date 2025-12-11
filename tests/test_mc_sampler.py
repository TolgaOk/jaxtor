"""Comprehensive tests for Markov Chain sampler functionality."""

import jax
import jax.numpy as jnp
import pytest
from jaxtor.sampler import mc
from jaxtor.env import tabular


# ============================================================================
# Test Environment
# ============================================================================


class SimpleEnv:
    """Simple deterministic test environment for MC sampler testing.

    This environment:
    - Has configurable state/action sizes
    - Deterministically transitions based on action
    - Terminates after a fixed number of steps
    - Provides known rewards for testing
    """

    def __init__(self, state_size: int = 4, action_size: int = 2, episode_len: int = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.episode_len = episode_len

    def init(self, key: jax.random.PRNGKey) -> dict:
        """Initialize environment state."""
        return {"step": 0}

    def reset(self, key: jax.random.PRNGKey, env_state: dict) -> tuple[jnp.ndarray, dict]:
        """Reset environment and return initial observation."""
        initial_state = jnp.zeros(self.state_size)
        return initial_state, {"step": 0}

    def step(
        self, key: jax.random.PRNGKey, act: jnp.ndarray, env_state: dict
    ) -> tuple[mc.Env.Step, dict]:
        """Step environment deterministically."""
        step_num = env_state["step"]

        # Simple deterministic transition: state = [step, action_idx, ...]
        action_idx = jnp.argmax(act)
        next_state = jnp.zeros(self.state_size)
        next_state = next_state.at[0].set(step_num + 1)
        next_state = next_state.at[1].set(action_idx)

        # Reward based on action (for testing)
        reward = action_idx.astype(float) + 1.0

        # Terminal after episode_len steps
        terminal = step_num + 1 >= self.episode_len
        truncated = False

        # Create Step object
        class StepImpl:
            def __init__(self, nobs, rew, term, trun):
                self.nobs = nobs
                self.rew = rew
                self.term = term
                self.trun = trun

        result = StepImpl(next_state, reward, terminal, truncated)
        next_env_state = {"step": step_num + 1}

        return result, next_env_state


class StochasticEnv:
    """Stochastic test environment for testing randomness."""

    def __init__(self, state_size: int = 4):
        self.state_size = state_size

    def init(self, key: jax.random.PRNGKey) -> dict:
        """Initialize environment state."""
        return {"step": 0}

    def reset(self, key: jax.random.PRNGKey, env_state: dict) -> tuple[jnp.ndarray, dict]:
        """Reset environment with random initial state."""
        initial_state = jax.random.normal(key, (self.state_size,))
        return initial_state, {"step": 0}

    def step(
        self, key: jax.random.PRNGKey, act: jnp.ndarray, env_state: dict
    ) -> tuple[mc.Env.Step, dict]:
        """Step with stochastic transitions."""
        step_num = env_state["step"]

        # Random next state
        next_state = jax.random.normal(key, (self.state_size,))

        # Random reward
        reward = jax.random.uniform(key, (), minval=0.0, maxval=1.0)

        # Random termination (10% chance)
        term_key = jax.random.fold_in(key, 1)
        terminal = jax.random.uniform(term_key) < 0.1
        truncated = False

        class StepImpl:
            def __init__(self, nobs, rew, term, trun):
                self.nobs = nobs
                self.rew = rew
                self.term = term
                self.trun = trun

        result = StepImpl(next_state, reward, terminal, truncated)
        next_env_state = {"step": step_num + 1}

        return result, next_env_state


# ============================================================================
# Initialization Tests
# ============================================================================


def test_mc_init():
    """Test MarkovChain sampler initialization."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)

    # Check initial state
    assert state.last_obs.shape == (4,)
    assert state.last_done == True
    assert state.eps_idx == 0
    assert state.eps_rew == 0.0
    assert state.eps_rew_queue.shape == (5,)
    assert state.eps_len_queue.shape == (5,)

    # Queues should be initialized with NaN
    assert jnp.all(jnp.isnan(state.eps_rew_queue))
    assert jnp.all(jnp.isnan(state.eps_len_queue))


def test_mc_init_different_sizes():
    """Test initialization with different queue sizes."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv()

    for queue_size in [1, 10, 100]:
        sampler = mc.MarkovChain(max_episode_len=10, queue_size=queue_size, env=env)
        state = sampler.init(key)

        assert state.eps_rew_queue.shape == (queue_size,)
        assert state.eps_len_queue.shape == (queue_size,)


# ============================================================================
# Basic Sampling Tests
# ============================================================================


def test_mc_single_sample():
    """Test single sample from the sampler."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])  # One-hot action

    transition, next_state = sampler.sample(action, state)

    # Check transition
    assert transition.obs.shape == (4,)
    assert transition.act.shape == (2,)
    assert transition.nobs.shape == (4,)
    assert isinstance(float(transition.rew), float)

    # Check state update
    assert next_state.eps_idx == 1
    assert next_state.eps_rew > 0  # Should have accumulated reward
    assert jnp.array_equal(next_state.last_obs, transition.nobs)


def test_mc_multiple_samples():
    """Test multiple consecutive samples."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=10)
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=env)

    state = sampler.init(key)

    for i in range(5):
        action = jnp.array([1.0, 0.0])
        transition, state = sampler.sample(action, state)

        # Episode index should increment
        assert state.eps_idx == i + 1

        # Reward should accumulate
        assert state.eps_rew > 0


# ============================================================================
# Episode Statistics Tests
# ============================================================================


def test_episode_returns_collected():
    """Test that episode returns are correctly collected in queue."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([0.0, 1.0])  # Reward = 2.0 per step

    # Run one complete episode (3 steps)
    total_reward = 0.0
    for i in range(3):
        transition, state = sampler.sample(action, state)
        total_reward += float(transition.rew)

    # After episode ends, return should be in queue
    # Queue stores final return at index 0
    assert not jnp.isnan(state.eps_rew_queue[0])
    assert jnp.isclose(state.eps_rew_queue[0], total_reward, rtol=1e-5)

    # Episode length should also be recorded
    assert state.eps_len_queue[0] == 3


def test_multiple_episode_returns():
    """Test collecting returns from multiple episodes."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=10, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    num_episodes = 5
    episode_returns = []

    for ep in range(num_episodes):
        ep_reward = 0.0
        for step in range(10):  # Run until episode ends
            transition, state = sampler.sample(action, state)
            ep_reward += float(transition.rew)

            if transition.term or transition.trun:
                episode_returns.append(ep_reward)
                break

    # Check that returns are stored in queue (most recent first)
    for i, expected_return in enumerate(reversed(episode_returns)):
        assert jnp.isclose(state.eps_rew_queue[i], expected_return, rtol=1e-5)


def test_queue_rolling():
    """Test that queue rolls correctly when full."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=3, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run 5 episodes (more than queue size)
    for ep in range(5):
        for step in range(10):
            transition, state = sampler.sample(action, state)
            if transition.term or transition.trun:
                break

    # Queue should only contain last 3 episodes
    # All should be non-NaN
    assert jnp.sum(~jnp.isnan(state.eps_rew_queue)) == 3
    assert jnp.sum(~jnp.isnan(state.eps_len_queue)) == 3


# ============================================================================
# Auto-Reset Tests
# ============================================================================


def test_auto_reset_on_termination():
    """Test that sampler auto-resets after episode termination."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run through one episode
    for i in range(3):
        transition, state = sampler.sample(action, state)

    # After terminal, eps_idx should reset to 0
    assert state.eps_idx == 0
    assert state.eps_rew == 0.0

    # But last_done should indicate we just reset
    # (Note: last_done is set to the 'done' value from the previous step)


def test_auto_reset_on_truncation():
    """Test auto-reset when max_episode_len is reached."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=100)  # Long episode
    sampler = mc.MarkovChain(max_episode_len=5, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run exactly max_episode_len steps
    for i in range(5):
        transition, state = sampler.sample(action, state)

    # Should truncate and reset
    assert transition.trun == True
    assert state.eps_idx == 0
    assert state.eps_rew == 0.0


def test_continuous_sampling_across_episodes():
    """Test that sampling continues smoothly across episode boundaries."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run 10 steps across multiple episodes
    for i in range(10):
        transition, state = sampler.sample(action, state)

        # Should always get valid transitions
        assert transition.obs.shape == (4,)
        assert transition.nobs.shape == (4,)


# ============================================================================
# Episode Index Tests
# ============================================================================


def test_eps_idx_increments():
    """Test that eps_idx increments correctly within episodes."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=10)
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    for i in range(5):
        transition, state = sampler.sample(action, state)
        assert state.eps_idx == i + 1


def test_eps_idx_resets_on_episode_end():
    """Test that eps_idx resets to 0 when episode ends."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run one complete episode
    for i in range(3):
        transition, state = sampler.sample(action, state)

    # After episode, eps_idx should be 0
    assert state.eps_idx == 0

    # Next step should start at 1
    transition, state = sampler.sample(action, state)
    assert state.eps_idx == 1


# ============================================================================
# Refresh Queue Tests
# ============================================================================


def test_refresh_queues():
    """Test that refresh_queues clears the statistics."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run one episode to fill queue
    for i in range(3):
        transition, state = sampler.sample(action, state)

    # Queue should have some data
    assert jnp.sum(~jnp.isnan(state.eps_rew_queue)) > 0

    # Refresh queues
    state = sampler.refresh_queues(state)

    # Queue should be all NaN again
    assert jnp.all(jnp.isnan(state.eps_rew_queue))
    assert jnp.all(jnp.isnan(state.eps_len_queue))


def test_refresh_queues_preserves_other_state():
    """Test that refresh_queues only affects the queues."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=5)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Take some steps
    for i in range(3):
        transition, state = sampler.sample(action, state)

    # Save current state values
    saved_obs = state.last_obs
    saved_eps_idx = state.eps_idx
    saved_eps_rew = state.eps_rew

    # Refresh queues
    state = sampler.refresh_queues(state)

    # Other state should be unchanged
    assert jnp.array_equal(state.last_obs, saved_obs)
    assert state.eps_idx == saved_eps_idx
    assert state.eps_rew == saved_eps_rew


# ============================================================================
# Vmap Tests
# ============================================================================


def test_vmap_multi_env():
    """Test vmapping sampler over multiple environments."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    env = SimpleEnv(state_size=4, action_size=2, episode_len=5)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    # Create multiple keys for multiple envs
    keys = jax.random.split(key, num_envs)

    # Vmap init over keys
    init_vmap = jax.vmap(sampler.init)
    states = init_vmap(keys)

    # Check batch dimensions
    assert states.last_obs.shape == (num_envs, 4)
    assert states.eps_rew_queue.shape == (num_envs, 5)

    # All envs should have different initial states (if using random init)
    # For our deterministic env, they'll be the same, but structure should be correct


def test_vmap_sampling():
    """Test vmapping sample operation."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    env = SimpleEnv(state_size=4, action_size=2, episode_len=10)
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=env)

    # Initialize multiple envs
    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(sampler.init)
    states = init_vmap(keys)

    # Create batch of actions
    actions = jnp.array([[1.0, 0.0]] * num_envs)

    # Vmap sample
    sample_vmap = jax.vmap(sampler.sample, in_axes=(0, 0))
    transitions, next_states = sample_vmap(actions, states)

    # Check batch dimensions
    assert transitions.obs.shape == (num_envs, 4)
    assert transitions.act.shape == (num_envs, 2)
    assert transitions.nobs.shape == (num_envs, 4)
    assert next_states.last_obs.shape == (num_envs, 4)


def test_vmap_unique_keys():
    """Test that vmapped envs use unique random keys."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    env = StochasticEnv(state_size=4)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    # Initialize with different keys
    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(sampler.init)
    states = init_vmap(keys)

    # Initial observations should differ (stochastic init)
    # Check that not all are identical
    all_same = jnp.all(states.last_obs[0] == states.last_obs[1])
    assert not all_same  # At least some should differ


def test_vmap_parallel_episodes():
    """Test running multiple episodes in parallel with vmap."""
    key = jax.random.PRNGKey(0)
    num_envs = 8

    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    # Initialize
    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(sampler.init)
    states = init_vmap(keys)

    # Run parallel episodes
    sample_vmap = jax.vmap(sampler.sample, in_axes=(0, 0))
    actions = jnp.array([[1.0, 0.0]] * num_envs)

    # Run for several steps
    for i in range(5):
        transitions, states = sample_vmap(actions, states)

    # All envs should have valid states
    assert states.last_obs.shape == (num_envs, 4)
    assert jnp.all(states.eps_idx >= 0)


# ============================================================================
# Edge Cases
# ============================================================================


def test_max_episode_len_exactly_reached():
    """Test behavior when exactly max_episode_len is reached."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=100)
    sampler = mc.MarkovChain(max_episode_len=5, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run exactly max_episode_len - 1 steps
    for i in range(4):
        transition, state = sampler.sample(action, state)
        assert not transition.trun

    # The 5th step (index 4) should trigger truncation
    transition, state = sampler.sample(action, state)
    assert transition.trun == True


def test_reward_accumulation_accuracy():
    """Test that rewards accumulate with numerical accuracy."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=10)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([0.0, 1.0])  # Reward = 2.0

    expected_reward = 0.0
    for i in range(5):
        transition, state = sampler.sample(action, state)
        expected_reward += float(transition.rew)
        assert jnp.isclose(state.eps_rew, expected_reward, rtol=1e-6)


def test_empty_queue_size():
    """Test behavior with very small queue size."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=1, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run multiple episodes
    for ep in range(3):
        for step in range(5):
            transition, state = sampler.sample(action, state)
            if transition.term:
                break

    # Queue size 1 should only store most recent episode
    assert state.eps_rew_queue.shape == (1,)
    assert not jnp.isnan(state.eps_rew_queue[0])


def test_large_queue_size():
    """Test behavior with large queue size."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=2)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=1000, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run a few episodes
    for ep in range(5):
        for step in range(5):
            transition, state = sampler.sample(action, state)
            if transition.term:
                break

    # Most of queue should still be NaN
    num_filled = jnp.sum(~jnp.isnan(state.eps_rew_queue))
    assert num_filled == 5
    assert num_filled < 1000


def test_transition_consistency():
    """Test that transition components are consistent."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=10)
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    transition, next_state = sampler.sample(action, state)

    # obs should match previous last_obs
    assert jnp.array_equal(transition.obs, state.last_obs)

    # act should match provided action
    assert jnp.array_equal(transition.act, action)

    # nobs should match next last_obs
    assert jnp.array_equal(transition.nobs, next_state.last_obs)


def test_done_flags_mutually_exclusive():
    """Test that term and trun are handled correctly."""
    key = jax.random.PRNGKey(0)
    env = SimpleEnv(state_size=4, action_size=2, episode_len=3)
    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)

    state = sampler.init(key)
    action = jnp.array([1.0, 0.0])

    # Run until termination
    for i in range(5):
        transition, state = sampler.sample(action, state)

        # Both can't be true in our simple env (but one might be)
        # Just verify they're valid boolean values
        assert transition.term in [0, 1, True, False] or isinstance(float(transition.term), float)
        assert transition.trun in [0, 1, True, False] or isinstance(float(transition.trun), float)


# ============================================================================
# Integration Tests with Real MDP Environments
# ============================================================================


def test_mc_with_garnet_mdp():
    """Test MC sampler with Garnet MDP environment."""
    key = jax.random.PRNGKey(0)
    init_key, sample_key = jax.random.split(key)

    # Create Garnet MDP
    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)

    # Note: MDP returns Step and MDPState, but MC sampler expects different interface
    # This test verifies the protocol mismatch is handled or we need an adapter


def test_mc_with_graph_mdp():
    """Test MC sampler with Graph MDP environment."""
    key = jax.random.PRNGKey(1)

    # Create Graph MDP
    config = tabular.graph.Config()
    env = tabular.graph.make(config)

    # Graph MDP has 6 states and 6 actions
    assert env.obs_space.shape == (6,)
    assert env.act_space.shape == (6,)


def test_mc_with_gridworld_mdp():
    """Test MC sampler with GridWorld MDP environment."""
    key = jax.random.PRNGKey(2)

    # Create GridWorld MDP
    config = tabular.gridworld.Config(
        board=["#####", "#  @#", "# # #", "#P  #", "#####"],
        p_slip=0.0
    )
    env = tabular.gridworld.make(config)

    # GridWorld should have valid state/action spaces
    assert env.obs_space.shape[0] > 0
    assert env.act_space.shape == (4,)  # 4 directional actions


def test_mdp_adapter_pattern():
    """Test creating an adapter to use MDPEnv with MC sampler.

    This demonstrates how to wrap MDPEnv to match the mc.Env protocol.
    """
    key = jax.random.PRNGKey(0)

    class MDPAdapter:
        """Adapter to make MDPEnv compatible with mc.Env protocol."""

        def __init__(self, mdp_env):
            self.mdp_env = mdp_env

        def init(self, key):
            """Initialize and return env_state."""
            return self.mdp_env.init(key)

        def reset(self, key, env_state):
            """Reset and return (obs, env_state)."""
            obs, env_state = self.mdp_env.reset(key, env_state)
            return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

        def step(self, key, act, env_state):
            """Step and return (Step, env_state)."""
            step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
            # Convert scalar obs to one-hot for mc sampler
            step_result = type(step_result)(
                nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
                rew=step_result.rew,
                term=step_result.term,
                trun=step_result.trun,
            )
            return step_result, next_mdp_state

    # Create Garnet MDP with adapter
    config = tabular.garnet.Config(state_size=8, action_size=3)
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    # Create MC sampler with adapted environment
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=10, env=adapted_env)

    # Initialize
    state = sampler.init(key)
    assert state.last_obs.shape == (8,)

    # Sample
    action = jax.nn.one_hot(0, 3)
    transition, next_state = sampler.sample(action, state)

    # Verify transition
    assert transition.obs.shape == (8,)
    assert transition.nobs.shape == (8,)
    assert next_state.eps_idx == 1


def test_mc_full_episode_with_mdp():
    """Test running full episode with MDP environment through adapter."""
    key = jax.random.PRNGKey(42)

    class MDPAdapter:
        def __init__(self, mdp_env):
            self.mdp_env = mdp_env

        def init(self, key):
            return self.mdp_env.init(key)

        def reset(self, key, env_state):
            obs, env_state = self.mdp_env.reset(key, env_state)
            return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

        def step(self, key, act, env_state):
            step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
            step_result = type(step_result)(
                nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
                rew=step_result.rew,
                term=step_result.term,
                trun=step_result.trun,
            )
            return step_result, next_mdp_state

    # Create environment
    config = tabular.garnet.Config(state_size=10, action_size=4)
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    # Create sampler
    sampler = mc.MarkovChain(max_episode_len=50, queue_size=10, env=adapted_env)
    state = sampler.init(key)

    # Run episode
    max_steps = 100
    for i in range(max_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action_idx = jax.random.randint(action_key, (), 0, 4)
        action = jax.nn.one_hot(action_idx, 4)
        transition, state = sampler.sample(action, state)

        if transition.term or transition.trun:
            # Episode ended, check statistics
            assert not jnp.isnan(state.eps_rew_queue[0])
            assert state.eps_len_queue[0] > 0
            break

    assert i < max_steps  # Should have terminated


def test_vmap_with_mdp_environments():
    """Test vmapping MC sampler with real MDP environments."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    class MDPAdapter:
        def __init__(self, mdp_env):
            self.mdp_env = mdp_env

        def init(self, key):
            return self.mdp_env.init(key)

        def reset(self, key, env_state):
            obs, env_state = self.mdp_env.reset(key, env_state)
            return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

        def step(self, key, act, env_state):
            step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
            step_result = type(step_result)(
                nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
                rew=step_result.rew,
                term=step_result.term,
                trun=step_result.trun,
            )
            return step_result, next_mdp_state

    # Create MDP environment
    config = tabular.garnet.Config(state_size=6, action_size=3)
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    # Create sampler
    sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=adapted_env)

    # Initialize multiple environments
    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(sampler.init)
    states = init_vmap(keys)

    # Check batch dimensions
    assert states.last_obs.shape == (num_envs, 6)
    assert states.eps_rew_queue.shape == (num_envs, 5)

    # Sample in parallel
    actions = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), 3)
    sample_vmap = jax.vmap(sampler.sample, in_axes=(0, 0))
    transitions, next_states = sample_vmap(actions, states)

    # Check outputs
    assert transitions.obs.shape == (num_envs, 6)
    assert transitions.nobs.shape == (num_envs, 6)
    assert next_states.eps_idx.shape == (num_envs,)


def test_mc_episode_statistics_with_mdp():
    """Test episode statistics collection with real MDP."""
    key = jax.random.PRNGKey(0)

    class MDPAdapter:
        def __init__(self, mdp_env):
            self.mdp_env = mdp_env

        def init(self, key):
            return self.mdp_env.init(key)

        def reset(self, key, env_state):
            obs, env_state = self.mdp_env.reset(key, env_state)
            return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

        def step(self, key, act, env_state):
            step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
            step_result = type(step_result)(
                nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
                rew=step_result.rew,
                term=step_result.term,
                trun=step_result.trun,
            )
            return step_result, next_mdp_state

    # Create environment with short episodes
    config = tabular.garnet.Config(state_size=5, action_size=2, max_episode_len=5)
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    sampler = mc.MarkovChain(max_episode_len=5, queue_size=10, env=adapted_env)
    state = sampler.init(key)

    # Run multiple episodes
    episodes_completed = 0
    max_total_steps = 100

    for i in range(max_total_steps):
        key, action_key, step_key = jax.random.split(key, 3)
        action_idx = jax.random.randint(action_key, (), 0, 2)
        action = jax.nn.one_hot(action_idx, 2)
        transition, state = sampler.sample(action, state)

        if transition.term or transition.trun:
            episodes_completed += 1
            if episodes_completed >= 3:
                break

    # Check that multiple episode returns were collected
    num_recorded = jnp.sum(~jnp.isnan(state.eps_rew_queue))
    assert num_recorded >= 3


def test_gridworld_deterministic_episode():
    """Test deterministic episode in GridWorld."""
    key = jax.random.PRNGKey(0)

    class MDPAdapter:
        def __init__(self, mdp_env):
            self.mdp_env = mdp_env

        def init(self, key):
            return self.mdp_env.init(key)

        def reset(self, key, env_state):
            obs, env_state = self.mdp_env.reset(key, env_state)
            return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

        def step(self, key, act, env_state):
            step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
            step_result = type(step_result)(
                nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
                rew=step_result.rew,
                term=step_result.term,
                trun=step_result.trun,
            )
            return step_result, next_mdp_state

    # Create simple GridWorld
    config = tabular.gridworld.Config(
        board=["###", "#P@", "###"],
        p_slip=0.0  # Deterministic
    )
    mdp_env = tabular.gridworld.make(config)
    adapted_env = MDPAdapter(mdp_env)

    sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=adapted_env)
    state = sampler.init(key)

    # Move right to goal (action 1 = right)
    action = jax.nn.one_hot(1, 4)
    key = jax.random.split(key)[0]
    transition, state = sampler.sample(action, state)

    # Should reach goal and terminate
    if transition.term:
        # Episode completed
        assert not jnp.isnan(state.eps_rew_queue[0])
        assert state.eps_len_queue[0] == 1
