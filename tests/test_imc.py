"""Tests for InducedMc sampler."""

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
        key: chex.PRNGKey

    def act(self, obs, state):
        return jnp.array(1), state


class CountingAgent:
    """Agent with a counter in state that increments each step."""

    @dataclass
    class State:
        key: chex.PRNGKey
        counter: int

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    def act(self, obs, state):
        action = jnp.array(0)
        new_state = state.replace(counter=state.counter + 1)
        return action, new_state


class RandomAgent:
    """Agent that takes random actions."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    def act(self, obs, state):
        key, subkey = jax.random.split(state.key)
        action = jax.random.randint(subkey, (), 0, self.action_size)
        return action, state.replace(key=key)


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

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State(key=key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

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

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = CountingAgent(action_size=4)
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = CountingAgent.State(key=key, counter=0)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

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

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State(key=key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    t1, state = imc_step.sample(state)
    t2, state = imc_step.sample(state)

    done = jnp.logical_or(t1.term, t1.trun)
    if not done:
        assert jnp.allclose(t1.nobs, t2.obs)


def test_state_construction():
    """Test that Imc.State can be constructed directly."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_episode_len=50,
    )
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = CountingAgent(action_size=4)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = CountingAgent.State(key=key, counter=42)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

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

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = GoRightAgent()
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = GoRightAgent.State(key=key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(imc_step.sample)

    transition, next_state = jit_sample(state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()


# =============================================================================
# IMC + VecMc Integration Tests
# =============================================================================


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

    mc_sampler = mc.Mc(max_episode_len=1000, queue_size=5, env=env)
    agent = RandomAgent(action_size=action_size)
    imc_step = imc.Imc(agent=agent, mc=mc_sampler)

    key, agent_key = jax.random.split(key)
    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = RandomAgent.State(key=agent_key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    action_counts = jnp.zeros(action_size)
    for _ in range(num_steps):
        transition, state = imc_step.sample(state)
        action_idx = transition.act
        action_counts = action_counts.at[action_idx].add(1)

    expected_count = num_steps / action_size
    min_threshold = expected_count * 0.5

    assert bool(jnp.all(action_counts >= min_threshold))
    assert bool(jnp.all(action_counts < num_steps * 0.5))


# =============================================================================
# VecMc + Vector Observations Tests
# =============================================================================


class VectorObsEnv:
    """Fake environment with vector observations."""

    def __init__(self, obs_dim: int = 4, action_size: int = 2):
        self.obs_dim = obs_dim
        self.action_size = action_size

    @dataclass
    class State:
        key: chex.PRNGKey

    @dataclass
    class Step:
        nobs: chex.Array
        rew: chex.Numeric
        term: chex.Numeric
        trun: chex.Numeric

    def init(self, key):
        return self.State(key=key)

    def reset(self, key, state):
        obs = jax.random.normal(key, (self.obs_dim,))
        return obs, state.replace(key=key)

    def step(self, key, act, state):
        k1, k2 = jax.random.split(key)
        nobs = jax.random.normal(k1, (self.obs_dim,))
        return (
            self.Step(
                nobs=nobs,
                rew=jnp.float32(0.0),
                term=jnp.bool_(False),
                trun=jnp.bool_(False),
            ),
            state.replace(key=k2),
        )


class VecObsAgent:
    """Agent that handles vector observations, returns scalar action per env."""

    def __init__(self, action_size: int = 2):
        self.action_size = action_size

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        key, subkey = jax.random.split(state.key)
        batch_shape = obs.shape[:-1]
        action = jax.random.randint(subkey, batch_shape, 0, self.action_size)
        return action, state.replace(key=key)


def test_vecmc_vector_obs_single_step():
    """Test Imc with VecMc and vector observations."""
    key = jax.random.PRNGKey(0)
    n_env = 4
    obs_dim = 4

    env = VectorObsEnv(obs_dim=obs_dim, action_size=2)
    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMc(mc=mc_sampler)
    agent = VecObsAgent(action_size=2)
    imc_step = imc.Imc(agent=agent, mc=vec_mc)

    key, env_key = jax.random.split(key)
    env_state = env.init(env_key)
    mc_keys = jax.random.split(key, n_env)
    mc_state = vec_mc.init(mc_keys, env_state)

    key, agent_key = jax.random.split(key)
    agent_state = VecObsAgent.State(key=agent_key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    transition, next_state = imc_step.sample(state)

    assert transition.obs.shape == (n_env, obs_dim)
    assert transition.act.shape == (n_env,)
    assert transition.rew.shape == (n_env,)
    assert transition.nobs.shape == (n_env, obs_dim)


def test_vecmc_vector_obs_jit():
    """Verify Imc with VecMc and vector observations works under JIT."""
    key = jax.random.PRNGKey(1)
    n_env = 4
    obs_dim = 4

    env = VectorObsEnv(obs_dim=obs_dim, action_size=2)
    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMc(mc=mc_sampler)
    agent = VecObsAgent(action_size=2)
    imc_step = imc.Imc(agent=agent, mc=vec_mc)

    key, env_key = jax.random.split(key)
    env_state = env.init(env_key)
    mc_keys = jax.random.split(key, n_env)
    mc_state = vec_mc.init(mc_keys, env_state)

    key, agent_key = jax.random.split(key)
    agent_state = VecObsAgent.State(key=agent_key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    jit_sample = jax.jit(imc_step.sample)
    transition, next_state = jit_sample(state)

    assert transition.obs.shape == (n_env, obs_dim)
    assert transition.act.shape == (n_env,)


# =============================================================================
# Chex Shape Assertion Tests
# =============================================================================


class WrongBatchAgent:
    """Agent that returns action with mismatched batch dimension."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        return jnp.zeros(3), state


def test_chex_imc_mismatched_batch_action():
    """Assert Imc raises on mismatched batch dimension between obs and action."""
    key = jax.random.PRNGKey(0)
    n_env = 4

    env = VectorObsEnv(obs_dim=4, action_size=2)
    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec_mc = mc.VecMc(mc=mc_sampler)
    agent = WrongBatchAgent()
    imc_step = imc.Imc(agent=agent, mc=vec_mc)

    key, env_key = jax.random.split(key)
    env_state = env.init(env_key)
    mc_keys = jax.random.split(key, n_env)
    mc_state = vec_mc.init(mc_keys, env_state)

    agent_state = WrongBatchAgent.State(key=key)
    state = imc.Imc.State(mc=mc_state, agent=agent_state)

    with pytest.raises(AssertionError):
        imc_step.sample(state)
