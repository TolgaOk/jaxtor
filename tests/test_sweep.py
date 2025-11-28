"""Tests for tabular MDP sweeping utilities."""

import jax
import jax.numpy as jnp
import pytest
from jaxtor.sampler import sweep
from jaxtor.env import tabular


@pytest.fixture(scope="module")
def simple_gridworld():
    """Simple 3x3 gridworld (3x3 internal area without walls)"""
    key = jax.random.PRNGKey(0)
    config = tabular.gridworld.Config(
        board=[
            "#####",
            "#P  #",
            "#   #",
            "#  @#",
            "#####",
        ],
        p_slip=0.0,
    )
    env = tabular.gridworld.make(config)
    return env.init(key)


@pytest.fixture(scope="module")
def hallway_gridworld():
    """4-step hallway gridworld: P----@"""
    key = jax.random.PRNGKey(0)
    config = tabular.gridworld.Config(
        board=[
            "#######",
            "#P   @#",
            "#######",
        ],
        p_slip=0.0,
    )
    env = tabular.gridworld.make(config)
    return env.init(key)


def test_sweep_q_single_step(simple_gridworld):
    """Test single-step returns only initial values."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=1)

    q_arr = jnp.ones((state.mdp.action_size, state.mdp.state_size))

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    trajectory = sweeper.backward(q_arr, mdp, mu)

    assert trajectory.shape == (1, state.mdp.action_size, state.mdp.state_size)

    assert jnp.allclose(trajectory[0], q_arr)


def test_sweep_q_two_steps(simple_gridworld):
    """Test with n_step=2 returns initial and one propagated value."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=2)

    q_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    q_arr = q_arr.at[:, -1].set(1.0)  # High value at goal state
    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    result = sweeper.backward(q_arr, mdp, mu)

    assert result.shape == (2, state.mdp.action_size, state.mdp.state_size)

    assert jnp.allclose(result[0], q_arr)

    assert not jnp.allclose(result[-1], q_arr)


def test_sweep_q_deterministic_propagation(hallway_gridworld):
    """Test propagation with deterministic transitions and policy."""
    state = hallway_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=3)

    q_arr = jnp.linspace(1.0, 3.0, state.mdp.state_size)[None, :].repeat(
        state.mdp.action_size, axis=0
    )

    mu = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    mu = mu.at[1, :].set(1.0)

    result = sweeper.backward(q_arr, mdp, mu)

    assert result.shape == (3, state.mdp.action_size, state.mdp.state_size)

    assert jnp.allclose(result[0], q_arr)

    assert not jnp.allclose(result[0], result[1])
    assert not jnp.allclose(result[1], result[2])


def test_sweep_q_policy_influence(hallway_gridworld):
    """Test that different policies produce different propagations."""
    state = hallway_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=3)

    q_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    q_arr = q_arr.at[0, -1].set(2.0)  # Action 0 has value 2 at goal
    q_arr = q_arr.at[1, -1].set(1.0)  # Action 1 has value 1 at goal

    mu_1 = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    mu_1 = mu_1.at[0, :].set(1.0)

    mu_2 = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    mu_2 = mu_2.at[1, :].set(1.0)

    result_1 = sweeper.backward(q_arr, mdp, mu_1)
    result_2 = sweeper.backward(q_arr, mdp, mu_2)

    assert (
        result_1.shape
        == result_2.shape
        == (3, state.mdp.action_size, state.mdp.state_size)
    )
    assert jnp.allclose(result_1[0], result_2[0])

    assert not jnp.allclose(result_1[1], result_2[1])


def test_sweep_q_uniform_convergence(simple_gridworld):
    """Test that uniform Q-values stay uniform under uniform policy."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=5)

    q_arr = jnp.ones((state.mdp.action_size, state.mdp.state_size))

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    result = sweeper.backward(q_arr, mdp, mu)

    assert result.shape == (5, state.mdp.action_size, state.mdp.state_size)

    for i in range(5):
        mean_val = jnp.mean(result[i])
        assert jnp.allclose(result[i], jnp.ones_like(result[i]) * mean_val, rtol=1e-5)


def test_sweep_q_absorbing_state(hallway_gridworld):
    """Test propagation with absorbing state."""
    state = hallway_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=10)

    q_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    q_arr = q_arr.at[:, -1].set(1.0)  # High value at goal

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    result = sweeper.backward(q_arr, mdp, mu)

    assert result.shape == (10, state.mdp.action_size, state.mdp.state_size)

    assert jnp.max(result[-1]) >= jnp.max(result[0])


def test_sweep_q_value_bounds(simple_gridworld):
    """Test that propagated values stay within reasonable bounds."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=10)

    q_arr = jnp.linspace(0.0, 1.0, state.mdp.state_size)[None, :].repeat(
        state.mdp.action_size, axis=0
    )
    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    result = sweeper.backward(q_arr, mdp, mu)

    assert result.shape == (10, state.mdp.action_size, state.mdp.state_size)

    assert jnp.all(jnp.isfinite(result))

    min_val = jnp.min(q_arr)
    max_val = jnp.max(q_arr)
    assert jnp.all(result >= min_val - 1e-5)
    assert jnp.all(result <= max_val + 1e-5)


def test_sweep_q_jit_compilation(simple_gridworld):
    """Test that sweep.backward can be JIT compiled."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=5)

    q_arr = jnp.ones((state.mdp.action_size, state.mdp.state_size))
    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    @jax.jit
    def propagate(q, mu):
        return sweeper.backward(q, mdp, mu)

    result = propagate(q_arr, mu)

    assert result.shape == (5, state.mdp.action_size, state.mdp.state_size)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result[0], q_arr)


def test_sweep_q_vmap(simple_gridworld):
    """Test that sweep.backward works with vmap for batch processing."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=3)

    batch_size = 4
    q_batch = jnp.ones((batch_size, state.mdp.action_size, state.mdp.state_size))
    for i in range(batch_size):
        q_batch = q_batch.at[i, :, :].mul(i + 1)

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    propagate_batch = jax.vmap(lambda q: sweeper.backward(q, mdp, mu))
    results = propagate_batch(q_batch)

    assert results.shape == (batch_size, 3, state.mdp.action_size, state.mdp.state_size)
    assert jnp.all(jnp.isfinite(results))

    for i in range(batch_size):
        assert jnp.allclose(results[i, 0], q_batch[i])


def test_sweep_q_sum_vs_explicit_power_series(hallway_gridworld):
    """Test that sweep.backward sum equals explicit computation sum_{k=0}^{n-1} (P^mu)^k q_arr."""
    state = hallway_gridworld
    mdp = state.mdp

    q_arr = jnp.linspace(1.0, 3.0, state.mdp.state_size)[None, :].repeat(
        state.mdp.action_size, axis=0
    )

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    n_step = 5

    sweeper = sweep.Sweep(n_step=n_step)
    trajectory = sweeper.backward(q_arr, mdp, mu)
    sweep_sum = trajectory.sum(axis=0)

    def propagate_once(q):
        return jnp.einsum("axs,ux,ux->as", state.mdp.transition, mu, q)

    explicit_sum = jnp.zeros_like(q_arr)
    current_q = q_arr.copy()
    for k in range(n_step):
        explicit_sum += current_q
        current_q = propagate_once(current_q)

    assert jnp.allclose(sweep_sum, explicit_sum, rtol=1e-5)


def test_sweep_forward_single_step(simple_gridworld):
    """Test single-step forward propagation returns only initial values."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=1)

    pi_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    pi_arr = pi_arr.at[0, 0].set(1.0)

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    trajectory = sweeper.forward(pi_arr, mdp, mu)

    assert trajectory.shape == (1, state.mdp.action_size, state.mdp.state_size)
    assert jnp.allclose(trajectory[0], pi_arr)


def test_sweep_forward_conservation(simple_gridworld):
    """Test that forward propagation conserves probability mass."""
    state = simple_gridworld
    mdp = state.mdp
    sweeper = sweep.Sweep(n_step=5)

    pi_arr = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / (
        state.mdp.action_size * state.mdp.state_size
    )
    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    result = sweeper.forward(pi_arr, mdp, mu)

    for i in range(5):
        total_mass = jnp.sum(result[i])
        assert jnp.isclose(total_mass, 1.0, rtol=1e-5)


def test_sweep_forward_sum_vs_explicit(hallway_gridworld):
    """Test forward propagation sum matches explicit computation."""
    state = hallway_gridworld
    mdp = state.mdp

    pi_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    pi_arr = pi_arr.at[:, 0].set(1.0 / state.mdp.action_size)

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    n_step = 5

    sweeper = sweep.Sweep(n_step=n_step)
    trajectory = sweeper.forward(pi_arr, mdp, mu)
    sweep_sum = trajectory.sum(axis=0)

    def propagate_forward_once(pi):
        return jnp.einsum("as,axs,ux->ux", pi, state.mdp.transition, mu)

    explicit_sum = jnp.zeros_like(pi_arr)
    current_pi = pi_arr.copy()
    for k in range(n_step):
        explicit_sum += current_pi
        current_pi = propagate_forward_once(current_pi)

    assert jnp.allclose(sweep_sum, explicit_sum, rtol=1e-5)


def test_backward_reward_propagation_4_steps(hallway_gridworld):
    """Verify reward propagates backward from goal to initial state in exactly 4 steps.

    Hallway: P----@ (5 states, 4 steps from start to goal)
    Policy: Always go right (toward goal)
    Expectation: Reward at state 3 propagates to initial state at step 3
    """
    state = hallway_gridworld
    transition = state.mdp.transition
    reward_asx = state.mdp.reward

    reward = jnp.einsum("axs,asx->as", transition, reward_asx)

    mdp = state.mdp

    mu = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    mu = mu.at[1, :].set(1.0)

    n_step = 5
    sweeper = sweep.Sweep(n_step=n_step)

    reward_trajectory = sweeper.backward(reward, mdp, mu)

    assert reward_trajectory.shape == (
        n_step,
        state.mdp.action_size,
        state.mdp.state_size,
    )

    initial_state_idx = 0
    right_action_idx = 1

    initial_reward_at_start = reward_trajectory[0, right_action_idx, initial_state_idx]
    assert jnp.isclose(initial_reward_at_start, 0.0, atol=1e-5)

    reward_at_start_step3 = reward_trajectory[3, right_action_idx, initial_state_idx]
    assert jnp.isclose(reward_at_start_step3, 1.0, atol=1e-5)


def test_backward_reward_uniform_policy_3x3(simple_gridworld):
    """Verify reward propagates with uniform random policy in 3x3 grid.

    Grid: 0 1 2 / 3 4 5 / 6 7 8 (P at 0, @ at 8)
    Policy: μ(a|s) = 0.25 for all actions

    Manual calculation:
    Step 0: Q(down,5)=1.0, Q(right,7)=1.0
    Step 1: Q(right,4)=0.25, Q(down,4)=0.25, Q(down,2)=0.25, Q(right,6)=0.25
    Step 2: Q(right,1)=0.0625, Q(down,1)=0.125, Q(right,3)=0.125, Q(down,3)=0.0625
    Step 3: Q(right,0)=0.046875, Q(down,0)=0.046875
    Expected sum at state 0, step 3: 0.09375
    """
    state = simple_gridworld
    transition = state.mdp.transition
    reward_asx = state.mdp.reward

    reward = jnp.einsum("axs,asx->as", transition, reward_asx)

    mdp = state.mdp
    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    sweeper = sweep.Sweep(n_step=4)
    reward_trajectory = sweeper.backward(reward, mdp, mu)

    assert reward_trajectory.shape == (4, state.mdp.action_size, state.mdp.state_size)

    reward_sum_at_state0_step3 = jnp.sum(reward_trajectory[3, :, 0])
    assert reward_sum_at_state0_step3 == pytest.approx(0.09375, abs=1e-6)


def test_forward_mass_uniform_policy_3x3(simple_gridworld):
    """Verify mass propagates with uniform random policy in 3x3 grid.

    Grid: 0 1 2 / 3 4 5 / 6 7 8 (P at 0, @ at 8)
    Policy: μ(a|s) = 0.25 for all actions

    Manual calculation:
    Step 0: π(a,0) = 0.25 for each action, total=1.0 at state 0
    Step 1: From 0: up→0, right→1, down→3, left→0
            State 0: 4×0.125 = 0.5 (from up and left actions)
            State 1: 4×0.0625 = 0.25 (from right action)
            State 3: 4×0.0625 = 0.25 (from down action)
    """
    state = simple_gridworld
    transition = state.mdp.transition

    mdp = state.mdp

    pi_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    pi_arr = pi_arr.at[:, 0].set(0.25)

    mu = jnp.ones((state.mdp.action_size, state.mdp.state_size)) / state.mdp.action_size

    sweeper = sweep.Sweep(n_step=2)
    trajectory = sweeper.forward(pi_arr, mdp, mu)

    assert trajectory.shape == (2, state.mdp.action_size, state.mdp.state_size)

    mass_at_state0_step1 = jnp.sum(trajectory[1, :, 0])
    mass_at_state1_step1 = jnp.sum(trajectory[1, :, 1])
    mass_at_state3_step1 = jnp.sum(trajectory[1, :, 3])

    assert mass_at_state0_step1 == pytest.approx(0.5, abs=1e-6)
    assert mass_at_state1_step1 == pytest.approx(0.25, abs=1e-6)
    assert mass_at_state3_step1 == pytest.approx(0.25, abs=1e-6)


def test_forward_mass_propagation_4_steps(hallway_gridworld):
    """Verify state distribution moves from initial state to goal in 4 steps.

    Hallway: P----@ (5 states, 4 steps from start to goal)
    Policy: Always go right (toward goal)
    Initial: One-hot on initial state
    Expectation: Mass moves to goal state after forward propagation
    """
    state = hallway_gridworld
    transition = state.mdp.transition

    mdp = state.mdp

    pi_arr = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    pi_arr = pi_arr.at[1, 0].set(1.0)

    mu = jnp.zeros((state.mdp.action_size, state.mdp.state_size))
    mu = mu.at[1, :].set(1.0)

    n_step = 5
    sweeper = sweep.Sweep(n_step=n_step)

    trajectory = sweeper.forward(pi_arr, mdp, mu)

    assert trajectory.shape == (n_step, state.mdp.action_size, state.mdp.state_size)

    initial_state_idx = 0
    goal_state_idx = state.mdp.state_size - 1
    right_action_idx = 1

    initial_mass_at_start = trajectory[0, right_action_idx, initial_state_idx]
    assert jnp.isclose(initial_mass_at_start, 1.0, atol=1e-5)

    final_mass_at_goal = trajectory[-1, right_action_idx, goal_state_idx]
    assert final_mass_at_goal > 0.5
