"""Tests for InducedMarkovChain single-step sampler."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jaxtor.sampler import imc, mc
from jaxtor.env import tabular


class GoRightAgent:
    """Agent that always takes the 'right' action (action index 1)."""

    @dataclass
    class State:
        pass

    def act(self, obs, state):
        action = jax.nn.one_hot(1, 4)
        return action, state


class CountingAgent:
    """Agent with a counter in state that increments each step."""

    @dataclass
    class State:
        counter: int

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    def act(self, obs, state):
        action = jax.nn.one_hot(0, self.action_size)
        new_state = state.replace(counter=state.counter + 1)
        return action, new_state


class MDPAdapter:
    """Adapter to make TabularEnv compatible with mc.Env protocol."""

    def __init__(self, mdp_env):
        self.mdp_env = mdp_env

    def init(self, key):
        return self.mdp_env.init(key)

    def reset(self, key, env_state):
        obs, env_state = self.mdp_env.reset(key, env_state)
        return jax.nn.one_hot(obs, env_state.mdp.state_size), env_state

    def step(self, key, act, env_state):
        step_result, next_mdp_state = self.mdp_env.step(key, jnp.argmax(act), env_state)
        step_result = tabular.Step(
            nobs=jax.nn.one_hot(step_result.nobs, next_mdp_state.mdp.state_size),
            rew=step_result.rew,
            term=step_result.term,
            trun=step_result.trun,
        )
        return step_result, next_mdp_state


def test_single_step_sample():
    """Test single-step IMC sampling."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transition, next_state = imc_step.sample(state)

    assert transition.obs.shape == (10,)
    assert transition.act.shape == (4,)
    assert transition.rew.shape == ()
    assert transition.term.shape == ()
    assert transition.trun.shape == ()
    assert transition.nobs.shape == (10,)


def test_single_step_state_update():
    """Test that single-step sample updates state correctly."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = CountingAgent.State(counter=0)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    assert state.agent.counter == 0

    transition, state = imc_step.sample(state)
    assert state.agent.counter == 1

    transition, state = imc_step.sample(state)
    assert state.agent.counter == 2


def test_single_step_consecutive_observations():
    """Test that consecutive single steps have matching observations."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    t1, state = imc_step.sample(state)
    t2, state = imc_step.sample(state)

    done = jnp.logical_or(t1.term, t1.trun)
    if not done:
        assert jnp.allclose(t1.nobs, t2.obs)


def test_init_returns_correct_structure():
    """Test that init returns correct IMC State structure."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = CountingAgent.State(counter=42)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    assert hasattr(state, "mc")
    assert hasattr(state, "agent")
    assert state.agent.counter == 42
    assert state.mc.last_obs.shape == (10,)


def test_jit_compilation_single_step():
    """Verify single-step sample() can be JIT compiled."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(imc_step.sample)

    transition, next_state = jit_sample(state)

    assert transition.obs.shape == (10,)
    assert transition.act.shape == (4,)


def test_vmap_over_batch_init():
    """Test vmap works for initializing batch of environments."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    assert mc_states.last_obs.shape == (num_envs, 10)
    assert mc_states.eps_rew_queue.shape == (num_envs, 5)


def test_vmap_single_step():
    """Test vmap works for single-step sampling from batch."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    imc_states = init_imc_vmap(mc_states, agent_states)

    sample_vmap = jax.vmap(imc_step.sample)
    transitions, next_states = sample_vmap(imc_states)

    assert transitions.obs.shape == (num_envs, 10)
    assert transitions.act.shape == (num_envs, 4)
    assert transitions.rew.shape == (num_envs,)


def test_metrics_computation():
    """Test metrics computation and queue refresh."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    mdp_env = tabular.gridworld.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    for _ in range(5):
        _, state = imc_step.sample(state)

    metrics, refreshed_state = imc_step.metrics(state)

    assert hasattr(metrics, "avg_eps_rew")
    assert hasattr(metrics, "avg_eps_len")
    assert jnp.all(jnp.isnan(refreshed_state.mc.eps_rew_queue))
    assert jnp.all(jnp.isnan(refreshed_state.mc.eps_len_queue))


@dataclass
class RandomAgent:
    """Agent that samples actions uniformly at random."""

    action_size: int

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        key, action_key = jax.random.split(state.key)
        action = jax.nn.one_hot(
            jax.random.randint(action_key, (), 0, self.action_size),
            self.action_size,
        )
        return action, state.replace(key=key)


def chi_square_uniform_test(counts, expected_count, alpha=0.01):
    """Test if counts follow uniform distribution."""
    from scipy import stats

    expected = jnp.full_like(counts, expected_count, dtype=float)
    _, p_value = stats.chisquare(counts, expected)
    return p_value > alpha


@pytest.mark.statistical
def test_initial_state_distribution_uniformity():
    """Verify Garnet MDP samples initial states uniformly via chi-square test."""
    key = jax.random.PRNGKey(42)
    state_size = 10
    num_samples = 1000

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)

    initial_state_counts = jnp.zeros(state_size)
    mc_state = mc_sampler.init(key)

    for _ in range(num_samples):
        key, reset_key = jax.random.split(key)
        obs, _ = adapted_env.reset(reset_key, mc_state.env)
        state_idx = jnp.argmax(obs)
        initial_state_counts = initial_state_counts.at[state_idx].add(1)

    expected_count = num_samples / state_size
    assert chi_square_uniform_test(initial_state_counts, expected_count, alpha=0.01)

    std = float(jnp.sqrt(expected_count * (1 - 1 / state_size)))
    assert bool(jnp.all(jnp.abs(initial_state_counts - expected_count) < 3 * std))


@pytest.mark.statistical
def test_empirical_transition_frequency():
    """Verify sampled transitions match MDP transition matrix with KL divergence < 0.1."""
    key = jax.random.PRNGKey(123)
    state_size = 5
    action_size = 3
    num_samples = 500

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=action_size,
        branch_size=2,
        max_episode_len=1000,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=5, env=adapted_env)

    mc_state = mc_sampler.init(key)
    mdp = mc_state.env.mdp

    test_state = 0
    test_action = 0
    true_probs = mdp.transition[test_action, :, test_state]

    next_state_counts = jnp.zeros(state_size)

    for _ in range(num_samples):
        key, _, _ = jax.random.split(key, 3)

        forced_state = jax.nn.one_hot(test_state, state_size)
        mc_state = mc_state.replace(
            env=mc_state.env.replace(last_state=forced_state),
            last_obs=forced_state,
        )

        action = jax.nn.one_hot(test_action, action_size)
        transition, mc_state = mc_sampler.sample(action, mc_state)

        next_state_idx = jnp.argmax(transition.nobs)
        next_state_counts = next_state_counts.at[next_state_idx].add(1)

    empirical_probs = next_state_counts / num_samples

    eps = 1e-10
    kl_div = float(
        jnp.sum(true_probs * jnp.log((true_probs + eps) / (empirical_probs + eps)))
    )

    assert kl_div < 0.1


@pytest.mark.statistical
def test_action_coverage_with_random_policy():
    """Verify RandomAgent samples each action at least 50% of expected frequency."""
    key = jax.random.PRNGKey(0)
    action_size = 4
    num_steps = 1000

    config = tabular.garnet.Config(
        state_size=10,
        action_size=action_size,
        max_episode_len=1000,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=5, env=adapted_env)
    agent = RandomAgent(action_size=action_size)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    key, agent_key = jax.random.split(key)
    mc_state = mc_sampler.init(key)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    action_counts = jnp.zeros(action_size)
    for _ in range(num_steps):
        transition, state = imc_step.sample(state)
        action_idx = jnp.argmax(transition.act)
        action_counts = action_counts.at[action_idx].add(1)

    expected_count = num_steps / action_size
    min_threshold = expected_count * 0.5

    assert bool(jnp.all(action_counts >= min_threshold))
    assert bool(jnp.all(action_counts < num_steps * 0.5))


@pytest.mark.statistical
def test_state_transition_markov_property():
    """Verify empirical transition distribution matches MDP matrix within 0.15 max diff."""
    key = jax.random.PRNGKey(0)
    state_size = 5
    action_size = 3
    num_samples = 200

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=action_size,
        max_episode_len=1000,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=5, env=adapted_env)

    mc_state = mc_sampler.init(key)
    mdp = mc_state.env.mdp

    test_state = 2
    test_action = 1

    true_probs = mdp.transition[test_action, :, test_state]

    next_state_counts = jnp.zeros(state_size)

    for i in range(num_samples):
        key, _ = jax.random.split(key)

        forced_state = jax.nn.one_hot(test_state, state_size)
        mc_state = mc_state.replace(
            env=mc_state.env.replace(last_state=forced_state),
            last_obs=forced_state,
            eps_idx=i % 10,
        )

        action = jax.nn.one_hot(test_action, action_size)
        transition, mc_state = mc_sampler.sample(action, mc_state)

        next_state_idx = jnp.argmax(transition.nobs)
        next_state_counts = next_state_counts.at[next_state_idx].add(1)

    empirical_probs = next_state_counts / num_samples

    max_diff = float(jnp.max(jnp.abs(empirical_probs - true_probs)))
    assert max_diff < 0.15
