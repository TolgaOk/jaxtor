"""Tests for Roll trajectory collector."""

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
# Roll Basic Tests
# =============================================================================


def test_roll_output_shapes():
    """Test Roll produces correct output shapes."""
    key = jax.random.PRNGKey(0)
    seqlen = 10

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=seqlen)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    assert transitions.obs.shape == (seqlen,)
    assert transitions.act.shape == (seqlen,)
    assert transitions.rew.shape == (seqlen,)
    assert transitions.term.shape == (seqlen,)
    assert transitions.trun.shape == (seqlen,)
    assert transitions.nobs.shape == (seqlen,)


def test_roll_trajectory_consistency():
    """Test consecutive observations match within trajectories."""
    key = jax.random.PRNGKey(0)

    config = tabular.gridworld.Config(
        board=["######", "#P  @#", "######"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc_sampler = mc.Mc(max_episode_len=10, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=5)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    for i in range(4):
        done = jnp.logical_or(transitions.term[i], transitions.trun[i])
        if not done:
            assert jnp.allclose(transitions.nobs[i], transitions.obs[i + 1])


def test_roll_state_continuity():
    """Test state preservation across multiple sample calls."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=20,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=20, queue_size=10, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=5)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    transitions1, state = sampler.sample(state)
    last_obs_after_first = state.mc.last_obs

    transitions2, state = sampler.sample(state)

    assert jnp.allclose(transitions2.obs[0], last_obs_after_first)


def test_roll_agent_state_updates():
    """Test stateful agent state updates during rollout."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=10)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = CountingAgent.State(counter=0)
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    assert state.agent.counter == 0

    transitions, state = sampler.sample(state)
    assert state.agent.counter == 10

    transitions, state = sampler.sample(state)
    assert state.agent.counter == 20


def test_roll_jit_compilation():
    """Verify Roll sample() can be JIT compiled."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=10)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(sampler.sample)

    transitions, next_state = jit_sample(state)

    assert transitions.obs.shape == (10,)
    assert transitions.act.shape == (10,)


def test_roll_seqlen_one():
    """Test with seqlen=1 (single step rollout)."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=1)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    assert transitions.obs.shape == (1,)
    assert transitions.rew.shape == (1,)


def test_roll_unroll_parameter():
    """Test that _unroll parameter works correctly."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    for unroll in [1, 2, 5]:
        sampler = rollout.Roll(imc=imc_step, seqlen=10, _unroll=unroll)

        env_state = env.init(key)
        mc_state = mc_sampler.init(key, env_state)
        agent_state = GoRightAgent.State()
        state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

        transitions, next_state = sampler.sample(state)

        assert transitions.obs.shape == (10,)


# =============================================================================
# Roll + VecMc Tests
# =============================================================================


def test_roll_with_vecmc():
    """Test Roll with VecMc for batched trajectory collection."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    class BatchedGoRightAgent:
        @dataclass
        class State:
            pass

        def act(self, key, obs, state):
            batch_size = obs.shape[0]
            return jnp.ones(batch_size, dtype=jnp.int32), state

    agent = BatchedGoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=vec_mc)
    sampler = rollout.Roll(imc=imc_step, seqlen=seqlen)

    env_state = env.init(key)
    mc_states = vec_mc.init(key, env_state)
    agent_state = BatchedGoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_states, agent=agent_state)

    transitions, next_state = sampler.sample(state)

    # Shape: (seqlen, n_env)
    assert transitions.obs.shape == (seqlen, num_envs)
    assert transitions.act.shape == (seqlen, num_envs)
    assert transitions.rew.shape == (seqlen, num_envs)


def test_roll_with_vecmc_jit():
    """Test JIT compilation of Roll + VecMc."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 10

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    class BatchedGoRightAgent:
        @dataclass
        class State:
            pass

        def act(self, key, obs, state):
            batch_size = obs.shape[0]
            return jnp.ones(batch_size, dtype=jnp.int32), state

    agent = BatchedGoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=vec_mc)
    sampler = rollout.Roll(imc=imc_step, seqlen=seqlen)

    env_state = env.init(key)
    mc_states = vec_mc.init(key, env_state)
    agent_state = BatchedGoRightAgent.State()
    state = imc.Imc.State(key=key, mc=mc_states, agent=agent_state)

    jit_sample = jax.jit(sampler.sample)

    transitions, next_state = jit_sample(state)

    assert transitions.obs.shape == (seqlen, num_envs)


def test_roll_with_vecmc_multiple_samples():
    """Test multiple sample calls with Roll + VecMc."""
    key = jax.random.PRNGKey(0)
    num_envs = 4
    seqlen = 5

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMC(mc=mc_sampler, n_env=num_envs)

    class BatchedCountingAgent:
        @dataclass
        class State:
            counter: jnp.ndarray

        def __init__(self, action_size: int = 4):
            self.action_size = action_size

        def act(self, key, obs, state):
            batch_size = obs.shape[0]
            new_state = state.replace(counter=state.counter + 1)
            return jnp.zeros(batch_size, dtype=jnp.int32), new_state

    agent = BatchedCountingAgent(action_size=4)
    imc_step = imc.Imc(agent=agent, mc=vec_mc)
    sampler = rollout.Roll(imc=imc_step, seqlen=seqlen)

    env_state = env.init(key)
    mc_states = vec_mc.init(key, env_state)
    agent_state = BatchedCountingAgent.State(counter=jnp.zeros(num_envs, dtype=jnp.int32))
    state = imc.Imc.State(key=key, mc=mc_states, agent=agent_state)

    jit_sample = jax.jit(sampler.sample)

    for i in range(5):
        transitions, state = jit_sample(state)
        expected_counter = (i + 1) * seqlen
        assert jnp.all(state.agent.counter == expected_counter)


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
def test_long_roll_stability():
    """Verify 1000-step rollout has no NaN values."""
    key = jax.random.PRNGKey(0)
    seqlen = 1000

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=100, env=env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=seqlen)

    key, agent_key = jax.random.split(key)
    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc.Imc.State(key=key, mc=mc_state, agent=agent_state)

    transitions, _ = sampler.sample(state)

    assert not bool(jnp.any(jnp.isnan(transitions.obs)))
    assert not bool(jnp.any(jnp.isnan(transitions.rew)))
    assert not bool(jnp.any(jnp.isnan(transitions.nobs)))


@pytest.mark.statistical
def test_deterministic_reproducibility():
    """Verify identical keys produce identical trajectories."""
    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = RandomAgent(action_size=4)
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)
    sampler = rollout.Roll(imc=imc_step, seqlen=20)

    # First run
    key1 = jax.random.PRNGKey(42)
    env_state1 = env.init(key1)
    mc_state1 = mc_sampler.init(key1, env_state1)
    key1, agent_key1 = jax.random.split(key1)
    agent_state1 = RandomAgent.State(key=agent_key1)
    state1 = imc.Imc.State(key=key1, mc=mc_state1, agent=agent_state1)
    transitions1, _ = sampler.sample(state1)

    # Second run with same key
    key2 = jax.random.PRNGKey(42)
    env_state2 = env.init(key2)
    mc_state2 = mc_sampler.init(key2, env_state2)
    key2, agent_key2 = jax.random.split(key2)
    agent_state2 = RandomAgent.State(key=agent_key2)
    state2 = imc.Imc.State(key=key2, mc=mc_state2, agent=agent_state2)
    transitions2, _ = sampler.sample(state2)

    assert bool(jnp.allclose(transitions1.obs, transitions2.obs))
    assert bool(jnp.allclose(transitions1.act, transitions2.act))
    assert bool(jnp.allclose(transitions1.rew, transitions2.rew))
