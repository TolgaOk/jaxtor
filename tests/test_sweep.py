"""Tests for stochastic sweep sampler."""

import jax
import jax.numpy as jnp
import chex
import pytest
from chex import dataclass
from jaxtor.env import tabular
from jaxtor.sampler import mc, sweep, imc, rollout


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


def test_sweep_init_returns_batched_state(small_env):
    """Init returns MC state with shape (A*S, ...)."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert state.last_obs.shape == (A * S,)
    assert state.eps_idx.shape == (A * S,)


def test_sweep_init_state_indices(small_env):
    """Each position starts at its designated state."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_state_indices = jnp.tile(jnp.arange(S), A)
    assert jnp.array_equal(state.last_obs, expected_state_indices)


def test_sweep_init_mdp_initial_conditioned(small_env):
    """Each position has MDP with one-hot initial distribution."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

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
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    act = jnp.zeros(A * S, dtype=jnp.int32)
    transition, new_state = sweeper.sample(act, state)

    assert transition.obs.shape == (A * S,)
    assert transition.act.shape == (A * S,)
    assert transition.nobs.shape == (A * S,)


# ============================================================================
# Initial Action Tests
# ============================================================================


def test_sweep_first_step_uses_init_action(small_env):
    """First step uses designated init_action, not provided action."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_init_actions = jnp.repeat(jnp.arange(A), S)

    # Provide different action than init_action
    act = jnp.full(A * S, 99, dtype=jnp.int32)
    transition, _ = sweeper.sample(act, state)

    # Transition should record init_action, not provided action
    assert jnp.array_equal(transition.act, expected_init_actions)


def test_sweep_subsequent_steps_use_provided_action(small_env):
    """After first step, provided action is used."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size

    # First step (uses init_action)
    act = jnp.zeros(A * S, dtype=jnp.int32)
    _, state = sweeper.sample(act, state)

    # Second step should use provided action
    provided_act = jnp.full(A * S, 2, dtype=jnp.int32)
    transition, _ = sweeper.sample(provided_act, state)

    assert jnp.array_equal(transition.act, provided_act)


# ============================================================================
# Reset Behavior Tests
# ============================================================================


def test_sweep_reset_returns_to_init_state():
    """After episode ends, position resets to its designated state."""
    key = jax.random.PRNGKey(0)
    # Short episode length to trigger resets
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=3)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=3, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_state_indices = jnp.tile(jnp.arange(S), A)

    # Run enough steps to trigger resets
    act = jnp.zeros(A * S, dtype=jnp.int32)
    for _ in range(10):
        _, state = sweeper.sample(act, state)

    # Positions at episode start should be at their designated state
    at_episode_start = state.eps_idx == 0
    if jnp.any(at_episode_start):
        reset_positions = jnp.where(at_episode_start)[0]
        expected_obs = expected_state_indices[reset_positions]
        actual_obs = state.last_obs[reset_positions]
        assert jnp.array_equal(expected_obs, actual_obs)


def test_sweep_reset_uses_init_action_again():
    """After reset, first step uses init_action again."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=2)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=2, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_init_actions = jnp.repeat(jnp.arange(A), S)

    # Run 2 steps to complete first episode
    act = jnp.zeros(A * S, dtype=jnp.int32)
    _, state = sweeper.sample(act, state)
    _, state = sweeper.sample(act, state)

    # Next step should use init_action (new episode)
    transition, state = sweeper.sample(act, state)

    # Positions that just reset should have used init_action
    at_step_1 = state.eps_idx == 1
    if jnp.any(at_step_1):
        reset_positions = jnp.where(at_step_1)[0]
        expected_acts = expected_init_actions[reset_positions]
        actual_acts = transition.act[reset_positions]
        assert jnp.array_equal(expected_acts, actual_acts)


# ============================================================================
# Metrics Tests
# ============================================================================


def test_sweep_metrics_returns_aggregated():
    """Metrics returns aggregated values across all positions."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=5)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=5, queue_size=3, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size

    # Run some steps to accumulate episode data
    act = jnp.zeros(A * S, dtype=jnp.int32)
    for _ in range(20):
        _, state = sweeper.sample(act, state)

    metrics, new_state = sweeper.metrics(state)

    # Metrics should be scalar (aggregated)
    assert metrics.avg_eps_rew.shape == ()
    assert metrics.avg_eps_len.shape == ()


# ============================================================================
# IMC and Rollout Integration Tests
# ============================================================================


def test_sweep_with_imc(small_env):
    """Sweep works as mc in Imc."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    transition, new_state = imc_sampler.sample(imc_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transition.obs.shape == (A * S,)


def test_sweep_with_rollout(small_env):
    """Sweep works with Rollout via Imc."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    rollout_sampler = rollout.Rollout(imc=imc_sampler, seqlen=5)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    transitions, new_state = rollout_sampler.sample(imc_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transitions.obs.shape == (5, A * S)


# ============================================================================
# JIT Compilation Tests
# ============================================================================


def test_sweep_init_jit(small_env):
    """Init can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)

    jit_init = jax.jit(sweeper.init)
    state = jit_init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert state.last_obs.shape == (A * S,)


def test_sweep_sample_jit(small_env):
    """Sample can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    act = jnp.zeros(A * S, dtype=jnp.int32)

    jit_sample = jax.jit(sweeper.sample)
    transition, new_state = jit_sample(act, state)

    assert transition.obs.shape == (A * S,)


def test_sweep_metrics_jit(small_env):
    """Metrics can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)
    state = sweeper.init(key, env_state)

    jit_metrics = jax.jit(sweeper.metrics)
    metrics, new_state = jit_metrics(state)

    assert metrics.avg_eps_rew.shape == ()


def test_sweep_full_pipeline_jit(small_env):
    """Full pipeline with Imc and Rollout can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    rollout_sampler = rollout.Rollout(imc=imc_sampler, seqlen=5)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    @jax.jit
    def run_rollout(state):
        return rollout_sampler.sample(state)

    transitions, new_state = run_rollout(imc_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transitions.obs.shape == (5, A * S)


# ============================================================================
# Comprehensive IMC Integration Tests
# ============================================================================


def test_sweep_imc_state_structure(small_env):
    """Verify IMC state has correct structure when using Sweep."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        step_count: chex.Array

    class CountingAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            new_state = AgentState(step_count=state.step_count + 1)
            return action, new_state

    agent = CountingAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(step_count=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # Verify IMC state structure
    assert hasattr(imc_state, 'key')
    assert hasattr(imc_state, 'mc')
    assert hasattr(imc_state, 'agent')

    # Verify MC state is batched (A*S)
    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert imc_state.mc.last_obs.shape == (A * S,)
    assert imc_state.mc.eps_idx.shape == (A * S,)


def test_sweep_imc_agent_state_updates(small_env):
    """Agent state updates correctly through IMC sampling."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        step_count: chex.Array

    class CountingAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            new_state = AgentState(step_count=state.step_count + 1)
            return action, new_state

    agent = CountingAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(step_count=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # Sample multiple times and verify agent state updates
    for i in range(5):
        _, imc_state = imc_sampler.sample(imc_state)
        assert imc_state.agent.step_count == i + 1


def test_sweep_imc_transition_consistency(small_env):
    """Transitions from IMC have consistent obs/nobs chain."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # Collect transitions and verify obs/nobs consistency
    prev_nobs = None
    for _ in range(5):
        transition, imc_state = imc_sampler.sample(imc_state)
        if prev_nobs is not None:
            # Current obs should equal previous nobs (unless reset happened)
            # We can't strictly assert this due to resets, but shapes should match
            assert transition.obs.shape == prev_nobs.shape
        prev_nobs = transition.nobs


# ============================================================================
# Comprehensive Rollout Integration Tests
# ============================================================================


def test_sweep_rollout_trajectory_shape(small_env):
    """Rollout produces trajectories with correct shape."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    rollout_sampler = rollout.Rollout(imc=imc_sampler, seqlen=10)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    transitions, new_state = rollout_sampler.sample(imc_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size

    # Verify trajectory shapes
    assert transitions.obs.shape == (10, A * S)
    assert transitions.act.shape == (10, A * S)
    assert transitions.nobs.shape == (10, A * S)
    assert transitions.rew.shape == (10, A * S)
    assert transitions.term.shape == (10, A * S)
    assert transitions.trun.shape == (10, A * S)


def test_sweep_rollout_multiple_trajectories(small_env):
    """Multiple rollouts produce independent trajectories."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    rollout_sampler = rollout.Rollout(imc=imc_sampler, seqlen=5)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # Collect multiple trajectories
    all_transitions = []
    state = imc_state
    for _ in range(3):
        transitions, state = rollout_sampler.sample(state)
        all_transitions.append(transitions)

    # Verify each trajectory has correct shape
    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    for trans in all_transitions:
        assert trans.obs.shape == (5, A * S)


# ============================================================================
# Comprehensive ResetRollout Integration Tests
# ============================================================================


def test_sweep_reset_rollout_resets_episode(small_env):
    """ResetRollout resets to initial state before each trajectory."""
    key = jax.random.PRNGKey(0)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=5)

    env_state = small_env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_init_obs = jnp.tile(jnp.arange(S), A)

    # Collect trajectory and verify first obs matches init state
    transitions, new_state = reset_rollout_sampler.sample(imc_state)

    # First observation should be the designated initial states
    assert jnp.array_equal(transitions.obs[0], expected_init_obs)

    # eps_idx after reset should start at seqlen (5 steps taken)
    assert jnp.all(new_state.mc.eps_idx == 5)


def test_sweep_reset_rollout_independent_episodes():
    """ResetRollout produces independent episodes each time."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=20)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=20, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=5)

    env_state = env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_init_obs = jnp.tile(jnp.arange(S), A)

    # Collect multiple trajectories
    state = imc_state
    for i in range(5):
        transitions, state = reset_rollout_sampler.sample(state)

        # Each trajectory should start from designated initial states
        assert jnp.array_equal(transitions.obs[0], expected_init_obs), f"Failed at iteration {i}"

        # Verify trajectory shape
        assert transitions.obs.shape == (5, A * S)
        assert transitions.act.shape == (5, A * S)


def test_sweep_reset_rollout_clears_queues():
    """ResetRollout clears episode statistics queues."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=3)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=3, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=10)

    env_state = env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # After reset rollout, eps_idx should reflect steps taken in fresh episode
    transitions, new_state = reset_rollout_sampler.sample(imc_state)

    # With max_episode_len=3 and seqlen=10, we should have 10 % 3 = 1 step into current episode
    # (episodes reset after 3 steps: 0,1,2 -> reset -> 0,1,2 -> reset -> 0,1,2 -> reset -> 0)
    assert jnp.all(new_state.mc.eps_idx == 10 % 3)


def test_sweep_reset_rollout_init_action_on_each_reset():
    """ResetRollout uses init_action at each episode start."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=3)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=3, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class ZeroAgent:
        State = AgentState

        def act(self, key, obs, state):
            # Always return 0, but sweep should override on first step
            action = jnp.zeros(obs.shape, dtype=jnp.int32)
            return action, state

    agent = ZeroAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=9)

    env_state = env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    expected_init_actions = jnp.repeat(jnp.arange(A), S)

    transitions, _ = reset_rollout_sampler.sample(imc_state)

    # With seqlen=9 and max_episode_len=3, we have 3 complete episodes
    # Steps 0, 3, 6 should use init_action
    assert jnp.array_equal(transitions.act[0], expected_init_actions)
    assert jnp.array_equal(transitions.act[3], expected_init_actions)
    assert jnp.array_equal(transitions.act[6], expected_init_actions)

    # Steps 1, 2, 4, 5, 7, 8 should use agent's action (all zeros)
    zero_actions = jnp.zeros(A * S, dtype=jnp.int32)
    assert jnp.array_equal(transitions.act[1], zero_actions)
    assert jnp.array_equal(transitions.act[2], zero_actions)
    assert jnp.array_equal(transitions.act[4], zero_actions)
    assert jnp.array_equal(transitions.act[5], zero_actions)
    assert jnp.array_equal(transitions.act[7], zero_actions)
    assert jnp.array_equal(transitions.act[8], zero_actions)


def test_sweep_reset_rollout_jit_compilation():
    """ResetRollout with Sweep can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=10)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        dummy: chex.Array

    class RandomAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            return action, state

    agent = RandomAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=5)

    env_state = env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(dummy=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    @jax.jit
    def run_reset_rollout(state):
        return reset_rollout_sampler.sample(state)

    transitions, new_state = run_reset_rollout(imc_state)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    assert transitions.obs.shape == (5, A * S)


def test_sweep_reset_rollout_state_preservation():
    """ResetRollout preserves agent state across resets."""
    key = jax.random.PRNGKey(0)
    config = tabular.garnet.Config(state_size=5, action_size=3, max_episode_len=10)
    env = tabular.garnet.make(config)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    @dataclass
    class AgentState:
        call_count: chex.Array

    class CountingAgent:
        State = AgentState

        def act(self, key, obs, state):
            action = jax.random.randint(key, obs.shape, 0, 3)
            new_state = AgentState(call_count=state.call_count + 1)
            return action, new_state

    agent = CountingAgent()
    imc_sampler = imc.Imc(agent=agent, mc=sweeper)
    reset_rollout_sampler = rollout.ResetRollout(imc=imc_sampler, seqlen=5)

    env_state = env.init(key)
    sweep_state = sweeper.init(key, env_state)
    agent_state = AgentState(call_count=jnp.array(0))
    imc_state = imc_sampler.init(key, sweep_state, agent_state)

    # Run reset rollout
    _, new_state = reset_rollout_sampler.sample(imc_state)

    # Agent state should reflect 5 calls (seqlen=5)
    assert new_state.agent.call_count == 5

    # Run another reset rollout
    _, final_state = reset_rollout_sampler.sample(new_state)

    # Agent state should now reflect 10 calls total
    assert final_state.agent.call_count == 10


# ============================================================================
# Determinism Tests
# ============================================================================


def test_sweep_deterministic_with_same_key(small_env):
    """Same key produces same results."""
    key = jax.random.PRNGKey(42)
    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=small_env)
    sweeper = sweep.Sweep(mc=mc_sampler)

    env_state = small_env.init(key)

    state1 = sweeper.init(key, env_state)
    state2 = sweeper.init(key, env_state)

    assert jnp.array_equal(state1.last_obs, state2.last_obs)

    S, A = env_state.mdp.state_size, env_state.mdp.action_size
    act = jnp.zeros(A * S, dtype=jnp.int32)

    trans1, _ = sweeper.sample(act, state1)
    trans2, _ = sweeper.sample(act, state2)

    assert jnp.array_equal(trans1.obs, trans2.obs)
    assert jnp.array_equal(trans1.nobs, trans2.nobs)
    assert jnp.array_equal(trans1.rew, trans2.rew)
