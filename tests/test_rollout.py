"""Tests for Rollout trajectory collector."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jaxtor.sampler import imc, mc, rollout
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


def test_trajectory_consistency_gridworld():
    """Test trajectory consistency in a hallway gridworld."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["######", "#P  @#", "######"],
        p_slip=0.0,
        max_episode_len=10,
    )
    mdp_env = tabular.gridworld.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    assert transitions.obs.shape[0] == 5
    assert transitions.act.shape[0] == 5
    assert transitions.rew.shape[0] == 5
    assert transitions.term.shape[0] == 5
    assert transitions.trun.shape[0] == 5
    assert transitions.nobs.shape[0] == 5

    for i in range(4):
        done = jnp.logical_or(transitions.term[i], transitions.trun[i])
        if not done:
            assert jnp.allclose(transitions.nobs[i], transitions.obs[i + 1])


def test_terminal_reset_observation():
    """Verify that after terminal, next step starts from reset state."""
    key = jax.random.PRNGKey(42)

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
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    term_indices = jnp.where(transitions.term)[0]
    if len(term_indices) > 0:
        term_idx = int(term_indices[0])
        if term_idx < 4:
            assert not jnp.any(jnp.isnan(transitions.obs[term_idx + 1]))


def test_metrics_queue_refresh():
    """Test that metrics() clears the queues after computation."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, state = sampler.sample(state)

    metrics, refreshed_state = imc_step.metrics(state)

    assert jnp.all(jnp.isnan(refreshed_state.mc.eps_rew_queue))
    assert jnp.all(jnp.isnan(refreshed_state.mc.eps_len_queue))


def test_incomplete_episode_no_stats():
    """Test that rollout shorter than episode length has all NaN in queue."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=100,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=100, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=3)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, state = sampler.sample(state)

    assert jnp.all(jnp.isnan(state.mc.eps_rew_queue))
    assert jnp.all(jnp.isnan(state.mc.eps_len_queue))

    metrics, _ = imc_step.metrics(state)
    assert jnp.isnan(metrics.avg_eps_rew)
    assert jnp.isnan(metrics.avg_eps_len)


def test_single_complete_episode_stats():
    """Test rollout that completes exactly 1 episode has correct stats."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=2)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, state = sampler.sample(state)

    assert not jnp.isnan(state.mc.eps_rew_queue[0])
    assert not jnp.isnan(state.mc.eps_len_queue[0])

    assert state.mc.eps_len_queue[0] == 1

    metrics, _ = imc_step.metrics(state)
    assert jnp.isclose(metrics.avg_eps_len, 1.0)


def test_multiple_complete_episodes_stats():
    """Test rollout that completes 2-3 episodes has correct average stats."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    mdp_env = tabular.gridworld.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=10, queue_size=10, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=10)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, state = sampler.sample(state)

    num_completed = jnp.sum(~jnp.isnan(state.mc.eps_rew_queue))
    assert num_completed >= 2

    valid_lens = state.mc.eps_len_queue[~jnp.isnan(state.mc.eps_len_queue)]
    assert jnp.all(valid_lens == 1)

    metrics, _ = imc_step.metrics(state)
    assert jnp.isclose(metrics.avg_eps_len, 1.0)


def test_state_preservation_across_samples():
    """Test that multiple sample() calls preserve state continuity."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=20,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=20, queue_size=10, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions1, state = sampler.sample(state)
    last_obs_after_first = state.mc.last_obs

    transitions2, state = sampler.sample(state)

    assert jnp.allclose(transitions2.obs[0], last_obs_after_first)


def test_agent_state_updates():
    """Test that stateful agent state increments during rollout."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=10)

    mc_state = mc_sampler.init(key)
    agent_state = CountingAgent.State(counter=0)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    assert state.agent.counter == 0

    transitions, state = sampler.sample(state)

    assert state.agent.counter == 10

    transitions, state = sampler.sample(state)

    assert state.agent.counter == 20


def test_output_shapes_and_types():
    """Verify rollout structure has correct shapes."""
    key = jax.random.PRNGKey(0)

    state_size = 8
    action_size = 4
    seqlen = 15

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=action_size,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=seqlen)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    assert transitions.obs.shape == (seqlen, state_size)
    assert transitions.act.shape == (seqlen, action_size)
    assert transitions.rew.shape == (seqlen,)
    assert transitions.term.shape == (seqlen,)
    assert transitions.trun.shape == (seqlen,)
    assert transitions.nobs.shape == (seqlen, state_size)

    assert isinstance(transitions.obs, jnp.ndarray)
    assert isinstance(transitions.act, jnp.ndarray)
    assert isinstance(transitions.rew, jnp.ndarray)
    assert isinstance(transitions.term, jnp.ndarray)
    assert isinstance(transitions.trun, jnp.ndarray)
    assert isinstance(transitions.nobs, jnp.ndarray)


def test_jit_compilation_sample():
    """Verify sample() can be JIT compiled."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=10)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(sampler.sample)

    transitions, next_state = jit_sample(state)

    assert transitions.obs.shape == (10, 10)
    assert transitions.act.shape == (10, 4)


def test_jit_compilation_metrics():
    """Verify metrics() can be JIT compiled."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, state = sampler.sample(state)

    jit_metrics = jax.jit(imc_step.metrics)

    metrics, refreshed_state = jit_metrics(state)

    assert hasattr(metrics, "avg_eps_rew")
    assert hasattr(metrics, "avg_eps_len")
    assert jnp.all(jnp.isnan(refreshed_state.mc.eps_rew_queue))


def test_jit_multiple_calls():
    """Verify JIT compiled functions work correctly across multiple calls."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    mc_state = mc_sampler.init(key)
    agent_state = CountingAgent.State(counter=0)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(sampler.sample)

    for i in range(5):
        transitions, state = jit_sample(state)
        expected_counter = (i + 1) * 5
        assert state.agent.counter == expected_counter


def test_vmap_over_batch_sample():
    """Test vmap works for sampling from batch of environments."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=10)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    imc_states = init_imc_vmap(mc_states, agent_states)

    sample_vmap = jax.vmap(sampler.sample)
    transitions, next_states = sample_vmap(imc_states)

    assert transitions.obs.shape == (num_envs, 10, 10)
    assert transitions.act.shape == (num_envs, 10, 4)
    assert transitions.rew.shape == (num_envs, 10)
    assert transitions.term.shape == (num_envs, 10)
    assert transitions.trun.shape == (num_envs, 10)
    assert transitions.nobs.shape == (num_envs, 10, 10)


def test_vmap_with_stateful_agent():
    """Test vmap works with stateful agent over batch."""
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
    agent = CountingAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=10)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = CountingAgent.State(counter=jnp.arange(num_envs))

    init_imc_vmap = jax.vmap(imc_step.init)
    imc_states = init_imc_vmap(mc_states, agent_states)

    assert jnp.array_equal(imc_states.agent.counter, jnp.arange(num_envs))

    sample_vmap = jax.vmap(sampler.sample)
    transitions, next_states = sample_vmap(imc_states)

    expected_counters = jnp.arange(num_envs) + 10
    assert jnp.array_equal(next_states.agent.counter, expected_counters)


def test_vmap_metrics():
    """Test vmap works for metrics over batch."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

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
    sampler = rollout.Rollout(imc=imc_step, seqlen=5)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    imc_states = init_imc_vmap(mc_states, agent_states)

    sample_vmap = jax.vmap(sampler.sample)
    transitions, imc_states = sample_vmap(imc_states)

    metrics_vmap = jax.vmap(imc_step.metrics)
    metrics, refreshed_states = metrics_vmap(imc_states)

    assert metrics.avg_eps_rew.shape == (num_envs,)
    assert metrics.avg_eps_len.shape == (num_envs,)
    assert refreshed_states.mc.eps_rew_queue.shape == (num_envs, 5)

    assert jnp.all(jnp.isnan(refreshed_states.mc.eps_rew_queue))


def test_seqlen_one():
    """Test with seqlen=1 (single step rollout)."""
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
    sampler = rollout.Rollout(imc=imc_step, seqlen=1)

    mc_state = mc_sampler.init(key)
    agent_state = GoRightAgent.State()
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    assert transitions.obs.shape == (1, 10)
    assert transitions.rew.shape == (1,)


def test_unroll_parameter():
    """Test that _unroll parameter works correctly."""
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

    for unroll in [1, 2, 5]:
        sampler = rollout.Rollout(imc=imc_step, seqlen=10, _unroll=unroll)

        mc_state = mc_sampler.init(key)
        agent_state = GoRightAgent.State()
        state = imc_step.init(mc=mc_state, agent=agent_state)

        transitions, next_state = sampler.sample(state)

        assert transitions.obs.shape == (10, 10)


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
def test_state_visitation_coverage():
    """Verify RandomAgent visits at least 80% of states over many rollouts."""
    key = jax.random.PRNGKey(0)
    state_size = 10
    action_size = 4

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=action_size,
        branch_size=3,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=10, env=adapted_env)
    agent = RandomAgent(action_size=action_size)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=100)

    visited_states = jnp.zeros(state_size)
    mc_state = mc_sampler.init(key)

    for _ in range(10):
        key, agent_key = jax.random.split(key)
        agent_state = RandomAgent.State(key=agent_key)
        state = imc_step.init(mc=mc_state, agent=agent_state)

        transitions, state = sampler.sample(state)
        mc_state = state.mc

        state_indices = jnp.argmax(transitions.obs, axis=-1)
        for idx in state_indices:
            visited_states = visited_states.at[idx].set(1)

    coverage = float(jnp.sum(visited_states) / state_size)
    assert coverage >= 0.8


@pytest.mark.statistical
def test_reward_distribution_statistics():
    """Verify rewards are within [0, 1] bounds and mean is approximately 0.5."""
    key = jax.random.PRNGKey(0)
    min_reward = 0.0
    max_reward = 1.0
    num_samples = 1000

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        min_reward=min_reward,
        max_reward=max_reward,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=10, env=adapted_env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=num_samples)

    key, agent_key = jax.random.split(key)
    mc_state = mc_sampler.init(key)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, _ = sampler.sample(state)
    rewards = transitions.rew

    assert bool(jnp.all(rewards >= min_reward))
    assert bool(jnp.all(rewards <= max_reward))

    expected_mean = (min_reward + max_reward) / 2
    actual_mean = float(jnp.mean(rewards))
    assert abs(actual_mean - expected_mean) < 0.1


@pytest.mark.statistical
def test_episode_length_distribution_with_truncation():
    """Verify episode lengths are bounded by max_episode_len and some truncate."""
    key = jax.random.PRNGKey(0)
    max_episode_len = 20
    num_rollouts = 100

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=max_episode_len,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(
        max_episode_len=max_episode_len, queue_size=100, env=adapted_env
    )
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=max_episode_len * num_rollouts)

    key, agent_key = jax.random.split(key)
    mc_state = mc_sampler.init(key)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    _, state = sampler.sample(state)

    eps_lens = state.mc.eps_len_queue[~jnp.isnan(state.mc.eps_len_queue)]

    assert bool(jnp.all(eps_lens <= max_episode_len))
    assert bool(jnp.any(eps_lens == max_episode_len))

    mean_len = float(jnp.mean(eps_lens))
    assert 1 < mean_len <= max_episode_len


@pytest.mark.statistical
def test_deterministic_reproducibility():
    """Verify identical keys produce identical trajectories."""
    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=20)

    key1 = jax.random.PRNGKey(42)
    mc_state1 = mc_sampler.init(key1)
    key1, agent_key1 = jax.random.split(key1)
    agent_state1 = RandomAgent.State(key=agent_key1)
    state1 = imc_step.init(mc=mc_state1, agent=agent_state1)
    transitions1, _ = sampler.sample(state1)

    key2 = jax.random.PRNGKey(42)
    mc_state2 = mc_sampler.init(key2)
    key2, agent_key2 = jax.random.split(key2)
    agent_state2 = RandomAgent.State(key=agent_key2)
    state2 = imc_step.init(mc=mc_state2, agent=agent_state2)
    transitions2, _ = sampler.sample(state2)

    assert bool(jnp.allclose(transitions1.obs, transitions2.obs))
    assert bool(jnp.allclose(transitions1.act, transitions2.act))
    assert bool(jnp.allclose(transitions1.rew, transitions2.rew))
    assert bool(jnp.allclose(transitions1.nobs, transitions2.nobs))


@pytest.mark.statistical
def test_queue_statistics_convergence():
    """Verify metrics.avg_eps_rew and avg_eps_len match manually computed queue means."""
    key = jax.random.PRNGKey(0)
    max_episode_len = 10
    queue_size = 20

    config = tabular.garnet.Config(
        state_size=5,
        action_size=3,
        max_episode_len=max_episode_len,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(
        max_episode_len=max_episode_len, queue_size=queue_size, env=adapted_env
    )
    agent = RandomAgent(action_size=3)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=200)

    key, agent_key = jax.random.split(key)
    mc_state = mc_sampler.init(key)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    _, state = sampler.sample(state)

    metrics, _ = imc_step.metrics(state)

    valid_rew = state.mc.eps_rew_queue[~jnp.isnan(state.mc.eps_rew_queue)]
    valid_len = state.mc.eps_len_queue[~jnp.isnan(state.mc.eps_len_queue)]

    expected_avg_rew = float(jnp.mean(valid_rew))
    expected_avg_len = float(jnp.mean(valid_len))

    assert float(metrics.avg_eps_rew) == pytest.approx(expected_avg_rew, rel=1e-5)
    assert float(metrics.avg_eps_len) == pytest.approx(expected_avg_len, rel=1e-5)


@pytest.mark.statistical
def test_long_rollout_stability():
    """Verify 1000-step rollout has no NaN values and valid state indices."""
    key = jax.random.PRNGKey(0)
    state_size = 10
    seqlen = 1000

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=4,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=100, env=adapted_env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=seqlen)

    key, agent_key = jax.random.split(key)
    mc_state = mc_sampler.init(key)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(mc=mc_state, agent=agent_state)

    transitions, _ = sampler.sample(state)

    assert not bool(jnp.any(jnp.isnan(transitions.obs)))
    assert not bool(jnp.any(jnp.isnan(transitions.rew)))
    assert not bool(jnp.any(jnp.isnan(transitions.nobs)))

    state_indices = jnp.argmax(transitions.obs, axis=-1)
    assert bool(jnp.all(state_indices >= 0))
    assert bool(jnp.all(state_indices < state_size))

    assert transitions.obs.shape == (seqlen, state_size)
    assert transitions.rew.shape == (seqlen,)


@pytest.mark.statistical
def test_batch_consistency_under_vmap():
    """Verify vmapped batch produces valid, independent trajectories."""
    key = jax.random.PRNGKey(0)
    num_envs = 16

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=20,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=20, queue_size=5, env=adapted_env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    sampler = rollout.Rollout(imc=imc_step, seqlen=50)

    keys = jax.random.split(key, num_envs + 1)
    key, env_keys = keys[0], keys[1:]

    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(env_keys)

    agent_keys = jax.random.split(key, num_envs)
    agent_states = RandomAgent.State(key=agent_keys)

    init_imc_vmap = jax.vmap(imc_step.init)
    imc_states = init_imc_vmap(mc_states, agent_states)

    sample_vmap = jax.vmap(sampler.sample)
    transitions, _ = sample_vmap(imc_states)

    assert transitions.obs.shape == (num_envs, 50, 10)
    assert not bool(jnp.any(jnp.isnan(transitions.obs)))
    assert not bool(jnp.any(jnp.isnan(transitions.rew)))

    first_obs = transitions.obs[:, 0, :]
    first_obs_indices = jnp.argmax(first_obs, axis=-1)
    unique_starts = len(jnp.unique(first_obs_indices))
    assert unique_starts > 1

    batch_mean_rew = float(jnp.mean(transitions.rew))
    assert 0.2 < batch_mean_rew < 0.8


def test_vec_rollout_output_shapes():
    """Test VecRollout produces correct output shapes."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10
    state_size = 10
    action_size = 4

    config = tabular.garnet.Config(
        state_size=state_size,
        action_size=action_size,
        max_episode_len=50,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=adapted_env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    transitions, next_states = vsampler.sample(states)

    assert transitions.obs.shape == (num_envs, seqlen, state_size)
    assert transitions.act.shape == (num_envs, seqlen, action_size)
    assert transitions.rew.shape == (num_envs, seqlen)
    assert transitions.term.shape == (num_envs, seqlen)
    assert transitions.trun.shape == (num_envs, seqlen)
    assert transitions.nobs.shape == (num_envs, seqlen, state_size)


def test_vec_rollout_state_updates():
    """Test VecRollout correctly updates batched states."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10

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
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = CountingAgent.State(counter=jnp.zeros(num_envs, dtype=jnp.int32))

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    assert jnp.all(states.agent.counter == 0)

    transitions, states = vsampler.sample(states)

    assert jnp.all(states.agent.counter == seqlen)


def test_vec_rollout_metrics():
    """Test VecRollout metrics computation."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

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
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=5, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    transitions, states = vsampler.sample(states)
    metrics, refreshed_states = jax.vmap(imc_step.metrics)(states)

    assert metrics.avg_eps_rew.shape == (num_envs,)
    assert metrics.avg_eps_len.shape == (num_envs,)
    assert jnp.all(jnp.isnan(refreshed_states.mc.eps_rew_queue))


def test_vec_rollout_jit_compilation():
    """Test VecRollout can be JIT compiled."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10

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
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    jit_sample = jax.jit(vsampler.sample)
    jit_metrics = jax.jit(jax.vmap(imc_step.metrics))

    transitions, states = jit_sample(states)
    metrics, states = jit_metrics(states)

    assert transitions.obs.shape == (num_envs, seqlen, 10)
    assert metrics.avg_eps_rew.shape == (num_envs,)


def test_vec_rollout_multiple_samples():
    """Test VecRollout works correctly across multiple sample calls."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 5

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
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = CountingAgent.State(counter=jnp.zeros(num_envs, dtype=jnp.int32))

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    jit_sample = jax.jit(vsampler.sample)

    for i in range(5):
        transitions, states = jit_sample(states)
        expected_counter = (i + 1) * seqlen
        assert jnp.all(states.agent.counter == expected_counter)


def test_vec_rollout_trajectory_consistency():
    """Test consecutive observations match within VecRollout trajectories."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10

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
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs)
    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(keys)

    agent_states = jax.tree.map(
        lambda x: jnp.stack([x] * num_envs), GoRightAgent.State()
    )

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    transitions, _ = vsampler.sample(states)

    for env_idx in range(num_envs):
        for step_idx in range(seqlen - 1):
            done = jnp.logical_or(
                transitions.term[env_idx, step_idx],
                transitions.trun[env_idx, step_idx],
            )
            if not done:
                assert jnp.allclose(
                    transitions.nobs[env_idx, step_idx],
                    transitions.obs[env_idx, step_idx + 1],
                )


@pytest.mark.statistical
def test_vec_rollout_independence():
    """Test VecRollout environments are independent."""
    key = jax.random.PRNGKey(0)
    num_envs = 8
    seqlen = 50

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=20,
    )
    mdp_env = tabular.garnet.make(config)
    adapted_env = MDPAdapter(mdp_env)

    mc_sampler = mc.MarkovChain(max_episode_len=20, queue_size=10, env=adapted_env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    vsampler = rollout.VecRollout(imc=imc_step, seqlen=seqlen, num_envs=num_envs)

    keys = jax.random.split(key, num_envs + 1)
    key, env_keys = keys[0], keys[1:]

    init_vmap = jax.vmap(mc_sampler.init)
    mc_states = init_vmap(env_keys)

    agent_keys = jax.random.split(key, num_envs)
    agent_states = RandomAgent.State(key=agent_keys)

    init_imc_vmap = jax.vmap(imc_step.init)
    states = init_imc_vmap(mc_states, agent_states)

    transitions, _ = vsampler.sample(states)

    first_obs_indices = jnp.argmax(transitions.obs[:, 0, :], axis=-1)
    unique_starts = len(jnp.unique(first_obs_indices))
    assert unique_starts > 1

    env_rewards = jnp.sum(transitions.rew, axis=1)
    reward_variance = jnp.var(env_rewards)
    assert reward_variance > 0
