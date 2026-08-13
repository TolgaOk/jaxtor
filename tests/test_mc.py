"""Tests for Mc sampler."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jaxtor.sampler import mc
from jaxtor.sampler.mc import Mc
from jaxtor.env import tabular


# =============================================================================
# Mc Tests
# =============================================================================


def test_mc_init():
    """Test Mc initialization."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_eps_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_eps_len=50, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)
    obs = sampler.observe(state)

    assert obs.shape == ()
    assert jnp.array_equal(obs, state.last_obs)
    assert state.last_obs.shape == ()
    assert state.eps_idx.shape == ()
    assert state.eps_idx == 0


@pytest.mark.parametrize(
    ("max_eps_len", "message"),
    [(0, "max_eps_len must be positive")],
)
def test_mc_rejects_nonpositive_static_configuration(
    max_eps_len,
    message,
):
    """The episode limit is validated when ``Mc`` is configured."""
    env = tabular.garnet.make(tabular.garnet.Config(state_size=2, action_size=2))

    with pytest.raises(ValueError, match=message):
        Mc(
            max_eps_len=max_eps_len,
            env=env,
        )


def test_mc_single_sample():
    """Test single sample from the sampler."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_eps_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_eps_len=50, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    action = jnp.array(1)
    transition, next_state = sampler.sample(action, state)

    assert transition.obs.shape == ()
    assert transition.act.shape == ()
    assert transition.rew.shape == ()
    assert transition.nobs.shape == ()


def test_mc_jit_compilation():
    """Verify sample() can be JIT compiled."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_eps_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_eps_len=50, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    jit_sample = jax.jit(sampler.sample)

    action = jnp.array(1)
    transition, next_state = jit_sample(action, state)

    assert transition.obs.shape == ()


def test_mc_consecutive_observations():
    """Test consecutive observations match."""
    key = jax.random.PRNGKey(0)

    config = tabular.garnet.Config(
        state_size=10,
        action_size=4,
        max_eps_len=50,
    )
    env = tabular.garnet.make(config)

    sampler = Mc(max_eps_len=50, env=env)
    env_state = env.init(key)
    state = sampler.init(key, env_state)

    action = jnp.array(1)
    t1, state = sampler.sample(action, state)
    t2, state = sampler.sample(action, state)

    done = jnp.logical_or(t1.term, t1.trun)
    if not done:
        assert jnp.allclose(t1.nobs, t2.obs)


# =============================================================================
# Chex Shape Assertion Tests
# =============================================================================


class NonScalarRewardEnv:
    """Env that returns non-scalar reward."""

    @dataclass
    class State:
        key: chex.PRNGKey

    @dataclass
    class Step:
        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def reset(self, key, state):
        return jnp.array(0), state.replace(key=key)

    def step(self, key, act, state):
        return (
            self.Step(
                nobs=jnp.array(0),
                rew=jnp.array([1.0, 2.0]),
                term=jnp.bool_(False),
                trun=jnp.bool_(False),
            ),
            state.replace(key=key),
        )


class MismatchedObsEnv:
    """Env that returns nobs with different shape than obs."""

    @dataclass
    class State:
        key: chex.PRNGKey

    @dataclass
    class Step:
        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def reset(self, key, state):
        return jnp.array(0), state.replace(key=key)

    def step(self, key, act, state):
        return (
            self.Step(
                nobs=jnp.array([0, 1]),
                rew=jnp.float32(0.0),
                term=jnp.bool_(False),
                trun=jnp.bool_(False),
            ),
            state.replace(key=key),
        )


def test_chex_mc_nonscalar_reward():
    """Assert Mc.sample raises on non-scalar reward from environment."""
    key = jax.random.PRNGKey(0)

    env = NonScalarRewardEnv()
    sampler = Mc(max_eps_len=50, env=env)
    state = sampler.init(key, NonScalarRewardEnv.State(key=key))

    with pytest.raises(AssertionError):
        sampler.sample(jnp.array(0), state)


def test_chex_mc_mismatched_obs_nobs():
    """Assert Mc.sample raises when nobs shape differs from obs."""
    key = jax.random.PRNGKey(0)

    env = MismatchedObsEnv()
    sampler = Mc(max_eps_len=50, env=env)
    state = sampler.init(key, MismatchedObsEnv.State(key=key))

    with pytest.raises(AssertionError):
        sampler.sample(jnp.array(0), state)


def test_chex_vecmc_wrong_batch_action():
    """Assert VecMc.sample raises when action has wrong batch dimension."""
    key = jax.random.PRNGKey(0)
    n_env = 4

    config = tabular.garnet.Config(state_size=10, action_size=4, max_eps_len=50)
    env = tabular.garnet.make(config)

    sampler = Mc(max_eps_len=50, env=env)
    vec_mc = mc.VecMc(mc=sampler)

    env_state = env.init(key)
    keys = jax.random.split(key, n_env)
    state = vec_mc.init(keys, env_state)

    wrong_action = jnp.zeros(3)
    with pytest.raises(AssertionError):
        vec_mc.sample(wrong_action, state)
