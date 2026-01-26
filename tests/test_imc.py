"""Tests for InducedMarkovChain sampler."""

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

    def act(self, key, obs, state):
        return jnp.array(1), state


class CountingAgent:
    """Agent with a counter in state that increments each step."""

    @dataclass
    class State:
        counter: int

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    def act(self, key, obs, state):
        action = jnp.array(0)
        new_state = state.replace(counter=state.counter + 1)
        return action, new_state


# =============================================================================
# IMC Tests
# =============================================================================


def test_single_step_sample():
    """Test single-step IMC sampling."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc_step.init(key, mc_state, agent_state)

    transition, next_state = imc_step.sample(state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()
    assert transition.rew.shape == ()
    assert transition.term.shape == ()
    assert transition.trun.shape == ()
    assert transition.nobs.shape == ()


def test_single_step_state_update():
    """Test that single-step sample updates state correctly."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = CountingAgent.State(counter=0)
    state = imc_step.init(key, mc_state, agent_state)

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
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc_step.init(key, mc_state, agent_state)

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
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = CountingAgent.State(counter=42)
    state = imc_step.init(key, mc_state, agent_state)

    assert hasattr(state, "mc")
    assert hasattr(state, "agent")
    assert state.agent.counter == 42
    assert state.mc.last_obs.shape == ()


def test_jit_compilation_single_step():
    """Verify single-step sample() can be JIT compiled."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc_step.init(key, mc_state, agent_state)

    jit_sample = jax.jit(imc_step.sample)

    transition, next_state = jit_sample(state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()


# =============================================================================
# IMC + VecMC Integration Tests
# =============================================================================


def test_imc_with_vecmc():
    """Test IMC composed with VecMC for batched sampling."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    class BatchedGoRightAgent:
        @dataclass
        class State:
            pass

        def act(self, key, obs, state):
            batch_size = obs.shape[0]
            return jnp.ones(batch_size, dtype=jnp.int32), state

    agent = BatchedGoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=vec_mc)

    env_state = env.init(key)
    mc_states = vec_mc.init(key, env_state)
    agent_state = BatchedGoRightAgent.State()
    state = imc_step.init(key, mc_states, agent_state)

    transition, next_state = imc_step.sample(state)

    assert transition.obs.shape == (num_envs,)
    assert transition.act.shape == (num_envs,)
    assert transition.rew.shape == (num_envs,)


def test_imc_with_vecmc_metrics():
    """Test metrics through IMC + VecMC."""
    key = jax.random.PRNGKey(0)
    num_envs = 4

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=10, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    class BatchedGoRightAgent:
        @dataclass
        class State:
            pass

        def act(self, key, obs, state):
            batch_size = obs.shape[0]
            return jnp.ones(batch_size, dtype=jnp.int32), state

    agent = BatchedGoRightAgent()
    imc_step = imc.InducedMarkovChain(agent=agent, mc=vec_mc)

    env_state = env.init(key)
    mc_states = vec_mc.init(key, env_state)
    agent_state = BatchedGoRightAgent.State()
    state = imc_step.init(key, mc_states, agent_state)

    for _ in range(10):
        _, state = imc_step.sample(state)

    # Get metrics from vec_mc
    metrics, refreshed_mc_state = vec_mc.metrics(state.mc)

    assert metrics.avg_eps_rew.shape == ()
    assert metrics.avg_eps_len.shape == ()


# =============================================================================
# Statistical Tests
# =============================================================================


@dataclass
class RandomAgent:
    """Agent that samples actions uniformly at random."""

    action_size: int

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, key, obs, state):
        key, action_key = jax.random.split(state.key)
        action = jax.random.randint(action_key, (), 0, self.action_size)
        return action, state.replace(key=key)


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
    env = tabular.garnet.make(config)

    mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=5, env=env)
    agent = RandomAgent(action_size=action_size)
    imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)

    key, agent_key = jax.random.split(key)
    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc_step.init(key, mc_state, agent_state)

    action_counts = jnp.zeros(action_size)
    for _ in range(num_steps):
        transition, state = imc_step.sample(state)
        action_idx = transition.act
        action_counts = action_counts.at[action_idx].add(1)

    expected_count = num_steps / action_size
    min_threshold = expected_count * 0.5

    assert bool(jnp.all(action_counts >= min_threshold))
    assert bool(jnp.all(action_counts < num_steps * 0.5))
