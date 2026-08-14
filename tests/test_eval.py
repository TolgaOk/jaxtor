"""Tests for tabular evaluator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import dataclass
from jaxtor.env import tabular
from jaxtor.eval import tabular as tabular_eval
from jaxtor.eval.tabular import Eval as TabularEval


# =============================================================================
# Fake agents
# =============================================================================


class TableAgent:
    """Agent that returns a preloaded Q-table for any observation."""

    @dataclass
    class State:
        q: jax.Array

    def q_vals(self, obs: jax.Array, state: TableAgent.State) -> jax.Array:
        """Return the stored Q-table for the requested state indices."""
        del obs
        return state.q


# =============================================================================
# Helpers
# =============================================================================


def _opt_q(mdp: tabular.Mdp, gamma: float, n_iters: int = 2000) -> jax.Array:
    """Compute optimal Q-values via Bellman iteration."""
    q = jnp.zeros((mdp.action_size, mdp.state_size))
    for _ in range(n_iters):
        q = tabular_eval._bellman_optimality(mdp, q, gamma)
    return q


def _make_mdp(
    state_size: int,
    action_size: int,
    terminal: jax.Array | None = None,
    reward: jax.Array | None = None,
) -> tabular.Mdp:
    """Build a minimal MDP with identity transitions for testing."""
    S, A = state_size, action_size
    transition = jnp.eye(S)[None, :, :].repeat(A, axis=0)
    if reward is None:
        reward = jnp.zeros((A, S, S))
    if terminal is None:
        terminal = jnp.zeros(S)
    initial = jnp.ones(S) / S
    return tabular.Mdp(
        transition=transition,
        reward=reward,
        initial=initial,
        terminal=terminal,
    )


# =============================================================================
# Tabular Eval Tests
# =============================================================================


def test_tabular_converged_agent_zero_error():
    """Optimal Q-values produce zero error across all metrics."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.allclose(metrics.diff_l1, 0.0, atol=1e-5)
    assert jnp.allclose(metrics.diff_linf, 0.0, atol=1e-5)
    assert jnp.allclose(metrics.bellman_l1, 0.0, atol=1e-2)
    assert jnp.allclose(metrics.bellman_linf, 0.0, atol=1e-2)
    assert jnp.allclose(metrics.value_l1, 0.0, atol=1e-5)
    assert jnp.allclose(metrics.value_linf, 0.0, atol=1e-5)


def test_tabular_known_offset_from_optimal():
    """Constant offset from optimal gives predictable L1 and Linf errors."""
    key = jax.random.PRNGKey(1)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    offset = 0.5
    shifted_q = opt_q + offset

    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    prev_state = TableAgent.State(q=shifted_q)
    agent_state = TableAgent.State(q=shifted_q)
    state = evaluator.init(prev_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    # L1 sums |error| over actions and averages over non-terminal states
    expected_l1 = offset * mdp.action_size
    assert jnp.allclose(metrics.value_l1, expected_l1, atol=1e-5)
    assert jnp.allclose(metrics.value_linf, offset, atol=1e-5)


def test_tabular_terminal_states_masked():
    """Metrics ignore terminal states even when they have large Q-error."""
    key = jax.random.PRNGKey(2)

    # 3-state gridworld: P _ @ — state 2 is terminal
    config = tabular.gridworld.Config(
        board=["#####", "#P @#", "#####"],
        p_slip=0.0,
    )
    env = tabular.gridworld.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)

    # Inject large error ONLY in terminal state (index 2)
    bad_q = opt_q.at[:, 2].set(999.0)

    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    prev_state = TableAgent.State(q=opt_q)
    agent_state = TableAgent.State(q=bad_q)
    state = evaluator.init(prev_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    # Value and diff metrics only reflect non-terminal states (which match opt_q)
    assert jnp.allclose(metrics.value_l1, 0.0, atol=1e-4)
    assert jnp.allclose(metrics.value_linf, 0.0, atol=1e-4)
    assert jnp.allclose(metrics.diff_l1, 0.0, atol=1e-4)
    assert jnp.allclose(metrics.diff_linf, 0.0, atol=1e-4)


def test_tabular_step_counter_increments():
    """The JAX scalar step advances after each evaluation."""
    key = jax.random.PRNGKey(3)

    config = tabular.garnet.Config(state_size=5, action_size=2)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    assert isinstance(state.step, jax.Array)
    assert state.step.shape == ()
    assert int(state.step) == 0

    _, state = evaluator.evaluate(agent_state, state)
    assert state.step == 1

    _, state = evaluator.evaluate(agent_state, state)
    assert state.step == 2


def test_tabular_previous_q_updated():
    """Evaluator state retains only the most recently evaluated Q-values."""
    key = jax.random.PRNGKey(4)

    config = tabular.garnet.Config(state_size=5, action_size=2)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    q1 = jnp.ones((mdp.action_size, mdp.state_size))
    q2 = jnp.ones((mdp.action_size, mdp.state_size)) * 2.0
    opt_q = _opt_q(mdp, gamma)

    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    state = evaluator.init(TableAgent.State(q=q1))
    _, state = evaluator.evaluate(TableAgent.State(q=q2), state)

    assert jnp.array_equal(state.prev_q, q2)


def test_tabular_same_agent_twice_zero_diff():
    """Evaluating same agent state consecutively gives zero diff metrics."""
    key = jax.random.PRNGKey(5)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q * 0.5)
    state = evaluator.init(agent_state)

    # First evaluation stores the current Q-values.
    _, state = evaluator.evaluate(agent_state, state)
    # Re-evaluating the same values produces zero change.
    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.allclose(metrics.diff_l1, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.diff_linf, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.pi_diff_l1, 0.0, atol=1e-6)
    assert jnp.allclose(metrics.pi_diff_linf, 0.0, atol=1e-6)


def test_tabular_bellman_fixed_point():
    """Optimal Q-values are a fixed point of the Bellman operator."""
    key = jax.random.PRNGKey(6)

    config = tabular.garnet.Config(state_size=10, action_size=4)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.allclose(metrics.bellman_l1, 0.0, atol=1e-2)
    assert jnp.allclose(metrics.bellman_linf, 0.0, atol=1e-2)


def test_tabular_policy_change_detected():
    """Changing Q-values enough to flip greedy actions produces nonzero pi_diff."""
    key = jax.random.PRNGKey(7)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    # Q1: action 0 is best everywhere
    q1 = jnp.zeros((mdp.action_size, mdp.state_size))
    q1 = q1.at[0, :].set(1.0)

    # Q2: action 1 is best everywhere — greedy policy flips
    q2 = jnp.zeros((mdp.action_size, mdp.state_size))
    q2 = q2.at[1, :].set(1.0)

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    state = evaluator.init(TableAgent.State(q=q1))
    metrics, state = evaluator.evaluate(TableAgent.State(q=q2), state)

    assert metrics.pi_diff_l1 > 0
    assert metrics.pi_diff_linf > 0


def test_tabular_jit_compilation():
    """Verify evaluate() can be JIT compiled."""
    key = jax.random.PRNGKey(8)

    config = tabular.garnet.Config(state_size=5, action_size=2)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.99

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    jit_evaluate = jax.jit(evaluator.evaluate)
    metrics, state = jit_evaluate(agent_state, state)

    assert metrics.iteration == 1
    assert metrics.bellman_l1.shape == ()


def test_tabular_zero_discount():
    """With gamma=0, optimal Q equals immediate reward; bellman error is zero."""
    key = jax.random.PRNGKey(9)

    config = tabular.garnet.Config(state_size=5, action_size=2)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp
    gamma = 0.0

    opt_q = _opt_q(mdp, gamma)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=gamma, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.allclose(metrics.bellman_l1, 0.0, atol=1e-5)
    assert jnp.allclose(metrics.value_l1, 0.0, atol=1e-5)


# =============================================================================
# Level 1: Edge Cases
# =============================================================================


def test_tabular_all_terminal_produces_nan():
    """All-terminal MDP has n_non_term=0, so L1 metrics are NaN (0/0)."""
    mdp = _make_mdp(state_size=3, action_size=2, terminal=jnp.ones(3))

    opt_q = jnp.zeros((2, 3))
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=0.99, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.isnan(metrics.diff_l1)
    assert jnp.isnan(metrics.bellman_l1)
    assert jnp.isnan(metrics.value_l1)


def test_tabular_value_norm_inf_when_optimal_is_zero():
    """value_norm is inf when opt_q is all zeros and agent Q is nonzero."""
    mdp = _make_mdp(state_size=3, action_size=2)

    opt_q = jnp.zeros((2, 3))
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=0.99, agent=agent, opt_q=opt_q)

    agent_q = jnp.ones((2, 3))
    agent_state = TableAgent.State(q=agent_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    assert jnp.isinf(metrics.value_norm)


def test_tabular_single_state_single_action():
    """Degenerate 1-state 1-action MDP produces valid scalar metrics."""
    S, A = 1, 1
    transition = jnp.ones((A, S, S))
    reward = jnp.array([[[0.5]]])
    mdp = tabular.Mdp(
        transition=transition,
        reward=reward,
        initial=jnp.ones(S),
        terminal=jnp.zeros(S),
    )

    opt_q = _opt_q(mdp, gamma=0.99)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=0.99, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    # All metrics should be scalar with no crashes
    assert metrics.diff_l1.shape == ()
    assert metrics.bellman_l1.shape == ()
    assert metrics.value_l1.shape == ()
    assert metrics.pi_eval_rho.shape == ()
    assert jnp.allclose(metrics.value_l1, 0.0, atol=1e-2)


# =============================================================================
# Level 2: Intermediate Dynamics
# =============================================================================


def test_tabular_non_term_mask_shape_and_values():
    """non_term mask has shape (1, S) and correctly flags terminal states."""
    key = jax.random.PRNGKey(20)

    config = tabular.gridworld.Config(
        board=["#####", "#P @#", "#####"],
        p_slip=0.0,
    )
    env = tabular.gridworld.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    non_term = (1 - mdp.terminal)[None, :]

    assert non_term.shape == (1, mdp.state_size)
    # State 2 is terminal (@), states 0 and 1 are not
    assert non_term[0, 0] == 1.0
    assert non_term[0, 1] == 1.0
    assert non_term[0, 2] == 0.0
    assert jnp.sum(non_term) == 2.0


def test_tabular_q_vals_returns_full_table():
    """agent.q_vals is called with all state indices and returns (A, S) table."""
    key = jax.random.PRNGKey(21)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    q = jnp.arange(15, dtype=jnp.float32).reshape(3, 5)
    agent = TableAgent()
    agent_state = TableAgent.State(q=q)

    all_states = jnp.arange(mdp.state_size)
    result = agent.q_vals(all_states, agent_state)

    assert result.shape == (mdp.action_size, mdp.state_size)
    assert jnp.array_equal(result, q)


def test_tabular_bellman_target_shape():
    """Bellman operator preserves Q-table shape (A, S)."""
    key = jax.random.PRNGKey(22)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    q = jnp.ones((mdp.action_size, mdp.state_size))
    target = tabular_eval._bellman_optimality(mdp, q, 0.99)

    assert target.shape == (mdp.action_size, mdp.state_size)


def test_tabular_greedy_policy_sums_to_one():
    """Greedy policy over Q-values sums to 1 over actions for each state."""
    key = jax.random.PRNGKey(23)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    opt_q = _opt_q(mdp, 0.99)
    pi = tabular_eval._greedy_policy(opt_q)

    assert pi.shape == (mdp.action_size, mdp.state_size)
    assert jnp.allclose(jnp.sum(pi, axis=0), 1.0)


def test_tabular_policy_eval_returns_q_shape():
    """policy_evaluation.q returns shape (A, S) matching the MDP."""
    key = jax.random.PRNGKey(24)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    opt_q = _opt_q(mdp, 0.99)
    pi = tabular_eval._greedy_policy(opt_q)
    greedy_q = tabular_eval._evaluate_policy(mdp, pi, 0.99)

    assert greedy_q.shape == (mdp.action_size, mdp.state_size)


def test_tabular_pi_eval_rho_uses_initial_distribution():
    """pi_eval_rho weights greedy values by MDP initial distribution."""
    key = jax.random.PRNGKey(25)

    config = tabular.garnet.Config(state_size=5, action_size=3)
    env = tabular.garnet.make(config)
    env_state = env.init(key)
    mdp = env_state.mdp

    opt_q = _opt_q(mdp, 0.99)
    agent = TableAgent()
    evaluator = TabularEval(mdp=mdp, gamma=0.99, agent=agent, opt_q=opt_q)

    agent_state = TableAgent.State(q=opt_q)
    state = evaluator.init(agent_state)

    metrics, state = evaluator.evaluate(agent_state, state)

    # Recompute pi_eval_rho manually
    pi = tabular_eval._greedy_policy(opt_q)
    greedy_q = tabular_eval._evaluate_policy(mdp, pi, 0.99)
    greedy_v = jnp.max(greedy_q, axis=0)
    expected_rho = jnp.sum(mdp.initial * greedy_v)

    assert jnp.allclose(metrics.pi_eval_rho, expected_rho, atol=1e-4)
