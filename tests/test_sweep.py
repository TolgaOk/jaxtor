"""Tests for tabular MDP sweeping utilities."""

import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jaxtor.sampler import sweep


@dataclass
class DummyMDP:
    transition: jnp.ndarray
    reward: jnp.ndarray
    state_size: int
    action_size: int


@dataclass
class DummyState:
    mdp: DummyMDP


class DummyEnv:
    def __init__(self, transition: jnp.ndarray, reward: jnp.ndarray):
        self._state = DummyState(
            mdp=DummyMDP(
                transition=transition,
                reward=reward,
                state_size=transition.shape[1],
                action_size=transition.shape[0],
            )
        )

    def init(self, key: jax.random.PRNGKey) -> DummyState:
        return self._state


class NoMDPEnv:
    @dataclass
    class State:
        value: int

    def init(self, key: jax.random.PRNGKey) -> "NoMDPEnv.State":
        return self.State(value=0)


def test_sweep_returns_matrices():
    key = jax.random.PRNGKey(0)
    transition = jnp.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    reward = jnp.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    env = DummyEnv(transition=transition, reward=reward)
    result = sweep.sweep(env, key)

    assert jnp.array_equal(result.transition, transition)
    assert jnp.array_equal(result.reward, reward)


def test_sweep_validates_transition_probabilities():
    key = jax.random.PRNGKey(1)
    transition = jnp.ones((2, 2, 2))
    reward = jnp.zeros((2, 2))
    env = DummyEnv(transition=transition, reward=reward)

    with pytest.raises(ValueError):
        sweep.sweep(env, key)


def test_sweep_requires_mdp_attribute():
    key = jax.random.PRNGKey(2)
    env = NoMDPEnv()

    with pytest.raises(ValueError):
        sweep.sweep(env, key)


# ============================================================================
# Sweep.q() N-step Propagation Tests
# ============================================================================


def test_sweep_q_single_step():
    """Test single-step returns only initial values."""
    # Simple 2-state, 2-action MDP
    transition = jnp.array([
        [[0.0, 1.0], [1.0, 0.0]],  # Action 0
        [[1.0, 0.0], [0.0, 1.0]],  # Action 1
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=1)

    # Initial Q-values
    q_arr = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    mu = jnp.ones((2, 2)) / 2.0

    # Apply n_step=1 (should return just initial values)
    trajectory = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory with shape (1, 2, 2)
    assert trajectory.shape == (1, 2, 2)

    # First (and only) element should be initial q_arr
    assert jnp.allclose(trajectory[0], q_arr)


def test_sweep_q_shape_consistency():
    """Test that output shape is (n_step, action_size, state_size)."""
    state_size = 5
    action_size = 3
    n_step = 4

    transition = jnp.ones((action_size, state_size, state_size)) / state_size
    mdp = sweep.TabularMDP(transition=transition)

    sweeper = sweep.Sweep(n_step=n_step)
    q_arr = jnp.ones((action_size, state_size))
    mu = jnp.ones((action_size, state_size)) / action_size

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory with time dimension
    assert result.shape == (n_step, action_size, state_size)

    # First element should be initial values
    assert jnp.allclose(result[0], q_arr)


def test_sweep_q_two_steps():
    """Test with n_step=2 returns initial and one propagated value."""
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=2)

    q_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mu = jnp.ones((2, 2)) / 2.0

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 2
    assert result.shape == (2, 2, 2)

    # First element should be initial values
    assert jnp.allclose(result[0], q_arr)

    # Second element should be propagated (different from initial)
    assert not jnp.allclose(result[1], q_arr)


def test_sweep_q_deterministic_propagation():
    """Test propagation with deterministic transitions and policy."""
    # Chain MDP: 0 -> 1 -> 2 (deterministic)
    transition = jnp.array([
        [[0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [0.0, 0.0, 1.0]],  # Action 0 (always move forward)
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=3)

    # Q-values: higher at the end of chain
    q_arr = jnp.array([[1.0, 2.0, 3.0]])

    # Deterministic policy: always take action 0
    mu = jnp.array([[1.0, 1.0, 1.0]])

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 3
    assert result.shape == (3, 1, 3)

    # First element is initial
    assert jnp.allclose(result[0], q_arr)

    # Values should propagate through chain
    assert not jnp.allclose(result[1], result[0])
    assert not jnp.allclose(result[2], result[1])


def test_sweep_q_multi_step():
    """Test multi-step propagation returns increasing trajectory."""
    # Simple MDP
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)

    q_arr = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    mu = jnp.ones((2, 2)) / 2.0

    # Compare different n_step values
    sweeper_2 = sweep.Sweep(n_step=2)
    sweeper_5 = sweep.Sweep(n_step=5)

    result_2 = sweeper_2.q(q_arr, mdp, mu)
    result_5 = sweeper_5.q(q_arr, mdp, mu)

    # Different lengths
    assert result_2.shape == (2, 2, 2)
    assert result_5.shape == (5, 2, 2)

    # Both should start with same initial values
    assert jnp.allclose(result_2[0], result_5[0])
    assert jnp.allclose(result_2[0], q_arr)


def test_sweep_q_policy_influence():
    """Test that different policies produce different propagations."""
    transition = jnp.array([
        [[1.0, 0.0], [0.0, 1.0]],  # Action 0: stay in same state
        [[0.0, 1.0], [1.0, 0.0]],  # Action 1: swap states
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=3)

    q_arr = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    # Policy 1: Always take action 0
    mu_1 = jnp.array([[1.0, 1.0], [0.0, 0.0]])

    # Policy 2: Always take action 1
    mu_2 = jnp.array([[0.0, 0.0], [1.0, 1.0]])

    result_1 = sweeper.q(q_arr, mdp, mu_1)
    result_2 = sweeper.q(q_arr, mdp, mu_2)

    # Both should have same shape and initial values
    assert result_1.shape == result_2.shape == (3, 2, 2)
    assert jnp.allclose(result_1[0], result_2[0])

    # But propagated values should differ
    assert not jnp.allclose(result_1[1], result_2[1])


def test_sweep_q_uniform_convergence():
    """Test that uniform Q-values stay uniform under uniform policy."""
    state_size = 4
    action_size = 3

    # Uniform transition matrix
    transition = jnp.ones((action_size, state_size, state_size)) / state_size

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=5)

    # Uniform Q-values
    q_arr = jnp.ones((action_size, state_size))

    # Uniform policy
    mu = jnp.ones((action_size, state_size)) / action_size

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 5
    assert result.shape == (5, action_size, state_size)

    # All values in trajectory should remain uniform
    for i in range(5):
        mean_val = jnp.mean(result[i])
        assert jnp.allclose(result[i], jnp.ones_like(result[i]) * mean_val, rtol=1e-5)


def test_sweep_q_absorbing_state():
    """Test propagation with absorbing state."""
    # State 1 is absorbing
    transition = jnp.array([
        [[0.0, 1.0], [0.0, 1.0]],  # Action 0: go to state 1
        [[0.0, 1.0], [0.0, 1.0]],  # Action 1: go to state 1
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=10)

    # High value at absorbing state
    q_arr = jnp.array([[0.0, 1.0], [0.0, 1.0]])

    mu = jnp.ones((2, 2)) / 2.0

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 10
    assert result.shape == (10, 2, 2)

    # Values should converge - later steps should be similar
    assert jnp.allclose(result[-1], result[-2], rtol=1e-3)


def test_sweep_q_stochastic_transitions():
    """Test with stochastic transition matrix."""
    # 50-50 transitions
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=6)

    q_arr = jnp.array([[2.0, 0.0], [0.0, 2.0]])
    mu = jnp.array([[0.5, 0.5], [0.5, 0.5]])

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 6
    assert result.shape == (6, 2, 2)

    # With uniform mixing, values should average out over steps
    # Check that later values are more mixed than initial
    assert jnp.std(result[-1]) < jnp.std(result[0])


def test_sweep_q_different_sized_mdps():
    """Test with various MDP sizes."""
    n_step = 3
    for state_size in [2, 5, 10]:
        for action_size in [2, 4]:
            transition = jnp.ones((action_size, state_size, state_size)) / state_size

            mdp = sweep.TabularMDP(transition=transition)
            sweeper = sweep.Sweep(n_step=n_step)

            q_arr = jnp.ones((action_size, state_size))
            mu = jnp.ones((action_size, state_size)) / action_size

            result = sweeper.q(q_arr, mdp, mu)

            # Check trajectory shape
            assert result.shape == (n_step, action_size, state_size)
            assert not jnp.any(jnp.isnan(result))
            assert not jnp.any(jnp.isinf(result))

            # First element should match initial
            assert jnp.allclose(result[0], q_arr)


def test_sweep_q_value_bounds():
    """Test that propagated values stay within reasonable bounds."""
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=10)

    # Bounded initial values
    q_arr = jnp.array([[0.0, 1.0], [0.5, 0.5]])
    mu = jnp.ones((2, 2)) / 2.0

    result = sweeper.q(q_arr, mdp, mu)

    # Should return trajectory of length 10
    assert result.shape == (10, 2, 2)

    # All values should be finite
    assert jnp.all(jnp.isfinite(result))

    # Without rewards, values should stay within initial bounds
    min_val = jnp.min(q_arr)
    max_val = jnp.max(q_arr)
    assert jnp.all(result >= min_val - 1e-5)
    assert jnp.all(result <= max_val + 1e-5)


def test_sweep_q_jit_compilation():
    """Test that sweep.q can be JIT compiled."""
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=5)

    q_arr = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    mu = jnp.ones((2, 2)) / 2.0

    # JIT compile the method
    @jax.jit
    def propagate(q, mu):
        return sweeper.q(q, mdp, mu)

    result = propagate(q_arr, mu)

    # Should return trajectory
    assert result.shape == (5, 2, 2)
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result[0], q_arr)


def test_sweep_q_vmap():
    """Test that sweep.q works with vmap for batch processing."""
    transition = jnp.array([
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, 0.5], [0.5, 0.5]],
    ])

    mdp = sweep.TabularMDP(transition=transition)
    sweeper = sweep.Sweep(n_step=3)

    # Batch of Q-values
    batch_size = 4
    q_batch = jnp.stack([
        jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        jnp.array([[0.5, 0.5], [0.5, 0.5]]),
        jnp.array([[2.0, 1.0], [1.0, 2.0]]),
        jnp.array([[0.0, 0.0], [1.0, 1.0]]),
    ])

    mu = jnp.ones((2, 2)) / 2.0

    # Vmap over batch dimension
    propagate_batch = jax.vmap(lambda q: sweeper.q(q, mdp, mu))
    results = propagate_batch(q_batch)

    # Should return batch of trajectories
    assert results.shape == (batch_size, 3, 2, 2)
    assert jnp.all(jnp.isfinite(results))

    # First element of each trajectory should match initial values
    for i in range(batch_size):
        assert jnp.allclose(results[i, 0], q_batch[i])


def test_sweep_q_sum_vs_explicit_power_series():
    """Test that sweep.q sum equals explicit computation sum_{k=0}^{n-1} (P^mu)^k q_arr."""
    # Simple 3-state, 2-action MDP
    transition = jnp.array([
        [[0.5, 0.3, 0.2],
         [0.2, 0.5, 0.3],
         [0.3, 0.2, 0.5]],
        [[0.4, 0.4, 0.2],
         [0.3, 0.3, 0.4],
         [0.2, 0.4, 0.4]],
    ])

    mdp = sweep.TabularMDP(transition=transition)

    # Initial Q values
    q_arr = jnp.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
    ])

    # Policy
    mu = jnp.array([
        [0.7, 0.6, 0.8],
        [0.3, 0.4, 0.2],
    ])

    n_step = 5

    # Method 1: Using sweep
    sweeper = sweep.Sweep(n_step=n_step)
    trajectory = sweeper.q(q_arr, mdp, mu)
    sweep_sum = trajectory.sum(axis=0)

    # Method 2: Explicit computation sum_{k=0}^{n-1} (P^mu)^k q_arr
    # where propagation operator is: Q'(a,s) = Σ_s' P(s'|s,a) Σ_u μ(u|s') Q(u,s')
    def propagate_once(q):
        return jnp.einsum("axs,ux,ux->as", transition, mu, q)

    explicit_sum = jnp.zeros_like(q_arr)
    current_q = q_arr.copy()
    for k in range(n_step):
        explicit_sum += current_q
        current_q = propagate_once(current_q)

    # Should be equal
    assert jnp.allclose(sweep_sum, explicit_sum, rtol=1e-5)
