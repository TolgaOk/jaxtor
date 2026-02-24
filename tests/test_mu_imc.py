"""Tests for MuImc sampler."""

import jax
import jax.numpy as jnp
import chex
from chex import dataclass
from jaxtor.sampler import mu_imc, mc, Roll
from jaxtor.env import tabular


# =============================================================================
# Test agents
# =============================================================================


class ConstantMuAgent:
    """Agent that always takes action 1 with a fixed log-probability."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        action = jnp.array(1)
        log_prob = jnp.float32(-0.5)
        return action, log_prob, state


class RandomMuAgent:
    """Agent that takes random discrete actions and returns log-prob."""

    def __init__(self, action_size: int = 4):
        self.action_size = action_size

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        key, act_key = jax.random.split(state.key)
        action = jax.random.randint(act_key, (), 0, self.action_size)
        log_prob = jnp.float32(-jnp.log(self.action_size))
        return action, log_prob, state.replace(key=key)


class VecGaussianMuAgent:
    """Agent for vectorized envs, handles batched observations."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        key, noise_key = jax.random.split(state.key)
        action = jax.random.normal(noise_key, obs.shape)
        log_prob = jnp.sum(
            -0.5 * (action**2 + jnp.log(2 * jnp.pi)),
            axis=-1,
        )
        return action, log_prob, state.replace(key=key)


# =============================================================================
# Basic MuImc tests
# =============================================================================


def test_single_step_has_log_mu():
    """MuImc transition includes log_mu field."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(state_size=10, action_size=4, max_episode_len=50)
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = ConstantMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = ConstantMuAgent.State(key=key)
    state = mu_imc.MuImc.State(mc=mc_state, agent=agent_state)

    transition, _ = step.sample(state)

    assert hasattr(transition, "log_mu")
    assert transition.log_mu.shape == ()
    assert jnp.allclose(transition.log_mu, jnp.float32(-0.5))


def test_transition_fields_match_mc():
    """MuImc transition has same obs/act/rew/term/trun/nobs as Mc."""
    key = jax.random.PRNGKey(1)

    config = tabular.garnet.Config(state_size=10, action_size=4, max_episode_len=50)
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = ConstantMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    state = mu_imc.MuImc.State(mc=mc_state, agent=ConstantMuAgent.State(key=key))

    transition, _ = step.sample(state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()
    assert transition.rew.shape == ()
    assert transition.term.shape == ()
    assert transition.trun.shape == ()
    assert transition.nobs.shape == ()


def test_state_update():
    """MuImc updates agent state across steps."""
    key = jax.random.PRNGKey(2)

    config = tabular.garnet.Config(state_size=10, action_size=4, max_episode_len=50)
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = RandomMuAgent(action_size=4)
    step = mu_imc.MuImc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    state = mu_imc.MuImc.State(mc=mc_state, agent=RandomMuAgent.State(key=key))

    t1, state = step.sample(state)
    t2, state = step.sample(state)

    # Agent key should advance each step
    assert t1.act.shape == ()
    assert t2.act.shape == ()


def test_init_method():
    """MuImc.init creates state from mc and agent states."""
    key = jax.random.PRNGKey(3)

    config = tabular.garnet.Config(state_size=10, action_size=4, max_episode_len=50)
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = ConstantMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    agent_state = ConstantMuAgent.State(key=key)

    state = step.init(mc_state, agent_state)
    assert hasattr(state, "mc")
    assert hasattr(state, "agent")


# =============================================================================
# JIT tests
# =============================================================================


def test_jit_single_step():
    """MuImc.sample works under JIT."""
    key = jax.random.PRNGKey(4)

    config = tabular.garnet.Config(state_size=10, action_size=4, max_episode_len=50)
    env = tabular.garnet.make(config)

    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    agent = ConstantMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=mc_sampler)

    env_state = env.init(key)
    mc_state = mc_sampler.init(key, env_state)
    state = mu_imc.MuImc.State(mc=mc_state, agent=ConstantMuAgent.State(key=key))

    jit_sample = jax.jit(step.sample)
    transition, _ = jit_sample(state)

    assert transition.obs.shape == ()
    assert jnp.allclose(transition.log_mu, jnp.float32(-0.5))


# =============================================================================
# VecMc + Roll composition tests
# =============================================================================


class VectorObsEnv:
    """Fake environment with vector observations."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 2):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

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
                rew=jnp.float32(1.0),
                term=jnp.bool_(False),
                trun=jnp.bool_(False),
            ),
            state.replace(key=k2),
        )


def test_vecmc_log_mu_shape():
    """MuImc + VecMc produces batched log_mu."""
    key = jax.random.PRNGKey(5)
    n_env = 4
    obs_dim = 4

    env = VectorObsEnv(obs_dim=obs_dim)
    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec = mc.VecMc(mc=mc_sampler)
    agent = VecGaussianMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=vec)

    key, env_key, agent_key = jax.random.split(key, 3)
    env_state = env.init(env_key)
    mc_state = vec.init(jax.random.split(key, n_env), env_state)
    state = mu_imc.MuImc.State(
        mc=mc_state, agent=VecGaussianMuAgent.State(key=agent_key)
    )

    transition, _ = step.sample(state)

    assert transition.obs.shape == (n_env, obs_dim)
    assert transition.act.shape == (n_env, obs_dim)
    assert transition.log_mu.shape == (n_env,)


def test_roll_composition():
    """MuImc composes with Roll to produce (n_env, seqlen) log_mu."""
    key = jax.random.PRNGKey(6)
    n_env = 4
    seqlen = 8
    obs_dim = 4

    env = VectorObsEnv(obs_dim=obs_dim)
    mc_sampler = mc.Mc(max_episode_len=50, queue_size=5, env=env)
    vec = mc.VecMc(mc=mc_sampler)
    agent = VecGaussianMuAgent()
    step = mu_imc.MuImc(agent=agent, mc=vec)
    roll = Roll(imc=step, seqlen=seqlen, seq_axis=1)

    key, env_key, agent_key = jax.random.split(key, 3)
    env_state = env.init(env_key)
    mc_state = vec.init(jax.random.split(key, n_env), env_state)
    state = mu_imc.MuImc.State(
        mc=mc_state, agent=VecGaussianMuAgent.State(key=agent_key)
    )

    trans, _ = jax.jit(roll.sample)(state)

    assert trans.obs.shape == (n_env, seqlen, obs_dim)
    assert trans.act.shape == (n_env, seqlen, obs_dim)
    assert trans.rew.shape == (n_env, seqlen)
    assert trans.log_mu.shape == (n_env, seqlen)
    # log-probs should be negative (log of probability < 1)
    assert jnp.all(trans.log_mu < 0.0)
