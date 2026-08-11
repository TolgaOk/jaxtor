"""Tests for the MJX environment adapter.

The headline tests assert one-to-one numerical parity between each jaxtor MJX
environment and its ``gymnasium`` v5 counterpart (observation, reward,
termination) under ``float64``, parametrized over all registered envs. The
remaining tests cover the Env protocol, sampler integration (Mc/VecMc/Roll),
and JIT compatibility in default ``float32``.
"""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pytest
from chex import dataclass

import gymnasium
from mujoco import mjx as _mjx

from jaxtor.env import mjx
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.rollout import Roll


# =============================================================================
# Fixtures
# =============================================================================

NUM_ENVS = 4
HOPPER_OBS = 11
HOPPER_NU = 3

# (gym/mjx name, action dim, observation dim, can terminate, clips qvel)
ENV_SPECS = [
    pytest.param("Hopper-v5", 3, 11, True, True, id="hopper"),
    pytest.param("Walker2d-v5", 6, 17, True, True, id="walker2d"),
    pytest.param("HalfCheetah-v5", 6, 17, False, False, id="halfcheetah"),
    pytest.param("Swimmer-v5", 2, 8, False, False, id="swimmer"),
]


@dataclass
class AgentState:
    key: chex.PRNGKey


class RandomAgent:
    """Agent sampling batched random continuous actions (vectorized path)."""

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        key, subkey = jrd.split(state.key)
        act = jrd.uniform(subkey, (obs.shape[0], HOPPER_NU), minval=-1.0, maxval=1.0)
        return self.Decision(act=act), AgentState(key=key)


class ScalarRandomAgent:
    """Agent sampling a single random continuous action (scalar path)."""

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        key, subkey = jrd.split(state.key)
        act = jrd.uniform(subkey, (HOPPER_NU,), minval=-1.0, maxval=1.0)
        return self.Decision(act=act), AgentState(key=key)


def _sync_state(env, qpos, qvel):
    """Build an MjxEnv.State from explicit qpos/qvel (forwarded)."""
    data = _mjx.make_data(env.mjx_model).replace(
        qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel)
    )
    return env.State(data=_mjx.forward(env.mjx_model, data))


# =============================================================================
# One-to-one parity with Gymnasium v5 (float64)
# =============================================================================


@pytest.mark.parametrize("name, nu, obs_dim, terminates, clips", ENV_SPECS)
def test_dynamics_parity_free_rollout(name, nu, obs_dim, terminates, clips):
    """MJX env matches its Gymnasium v5 counterpart to ~machine precision."""
    with jax.enable_x64():
        raw = gymnasium.make(name).unwrapped
        raw.reset(seed=0)
        env = mjx.make(name)
        state = _sync_state(env, raw.data.qpos.copy(), raw.data.qvel.copy())
        step_fn = jax.jit(env.step)
        key = jrd.PRNGKey(0)
        rng = np.random.default_rng(7)

        # Short horizon: per-step dynamics are exact, but chaotic envs
        # (HalfCheetah) amplify float noise over a free rollout.
        max_obs = max_rew = 0.0
        term_mismatch = 0
        steps = 0
        for _ in range(16):
            act = rng.uniform(-1.0, 1.0, size=nu)
            obs_g, rew_g, term_g, _, _ = raw.step(act)
            result, state = step_fn(key, jnp.asarray(act), state)
            max_obs = max(max_obs, float(np.abs(obs_g - np.asarray(result.nobs)).max()))
            max_rew = max(max_rew, abs(rew_g - float(result.rew)))
            term_mismatch += int(term_g != bool(result.term))
            steps += 1
            if term_g:
                break

        assert steps >= 5
        assert max_obs < 1e-9
        assert max_rew < 1e-9
        assert term_mismatch == 0


@pytest.mark.parametrize("name, nu, obs_dim, terminates, clips", ENV_SPECS)
def test_obs_termination_parity_random_states(name, nu, obs_dim, terminates, clips):
    """Observation and termination match Gymnasium across diverse states."""
    with jax.enable_x64():
        raw = gymnasium.make(name).unwrapped
        raw.reset(seed=2)
        env = mjx.make(name)
        rng = np.random.default_rng(3)

        max_obs = 0.0
        term_mismatch = 0
        saw_unhealthy = False
        for _ in range(50):
            qpos = raw.init_qpos + rng.uniform(-0.6, 0.6, size=raw.model.nq)
            qvel = raw.init_qvel + rng.uniform(-15.0, 15.0, size=raw.model.nv)
            raw.set_state(qpos, qvel)

            obs_g = raw._get_obs()
            data = _mjx.forward(
                env.mjx_model,
                _mjx.make_data(env.mjx_model).replace(
                    qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel)
                ),
            )
            obs_m = np.asarray(env.task.obs(data))
            term_m = bool(env.task.terminal(data))

            max_obs = max(max_obs, np.abs(obs_g - obs_m).max())
            if terminates:
                term_g = not raw.is_healthy
                term_mismatch += int(term_g != term_m)
                saw_unhealthy = saw_unhealthy or term_g
            else:
                term_mismatch += int(term_m)

        assert obs_m.shape == (obs_dim,)
        assert max_obs < 1e-9
        assert term_mismatch == 0
        assert saw_unhealthy == terminates


@pytest.mark.parametrize("name, nu, obs_dim, terminates, clips", ENV_SPECS)
def test_qvel_clip_parity(name, nu, obs_dim, terminates, clips):
    """Observation velocity clipping to [-10, 10] matches Gymnasium."""
    if not clips:
        pytest.skip(f"{name} does not clip observation velocities")
    with jax.enable_x64():
        raw = gymnasium.make(name).unwrapped
        raw.reset(seed=5)
        env = mjx.make(name)

        qpos = raw.init_qpos.copy()
        qvel = np.full(raw.model.nv, 50.0)
        raw.set_state(qpos, qvel)
        obs_g = raw._get_obs()

        data = _mjx.forward(
            env.mjx_model,
            _mjx.make_data(env.mjx_model).replace(
                qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel)
            ),
        )
        obs_m = np.asarray(env.task.obs(data))

        assert np.isclose(obs_m[obs_dim - raw.model.nv :], 10.0).all()
        assert np.abs(obs_g - obs_m).max() < 1e-9


@pytest.mark.statistical
@pytest.mark.parametrize(
    "name, qvel_gaussian, scale",
    [
        ("Hopper-v5", False, 5e-3),
        ("Walker2d-v5", False, 5e-3),
        ("HalfCheetah-v5", True, 0.1),
        ("Swimmer-v5", False, 0.1),
    ],
)
def test_reset_noise_distribution_matches_gymnasium(name, qvel_gaussian, scale):
    """Reset qpos (uniform) and qvel (uniform/Gaussian) spread matches v5."""
    env = mjx.make(name)
    state = env.init(jrd.PRNGKey(0))
    keys = jrd.split(jrd.PRNGKey(1), 4000)

    def reset_deltas(key):
        _, reset_state = env.reset(key, state)
        return (
            reset_state.data.qpos - env.init_qpos,
            reset_state.data.qvel - env.init_qvel,
        )

    dqpos, dqvel = jax.vmap(reset_deltas)(keys)
    uniform_std = scale / np.sqrt(3.0)

    assert np.isclose(np.std(np.asarray(dqpos)), uniform_std, rtol=0.1)
    expected = scale if qvel_gaussian else uniform_std
    assert np.isclose(np.std(np.asarray(dqvel)), expected, rtol=0.1)


# =============================================================================
# Env protocol
# =============================================================================


@pytest.mark.parametrize(
    "name, obs_dim",
    [
        ("Hopper-v5", 11),
        ("Walker2d-v5", 17),
        ("HalfCheetah-v5", 17),
        ("Swimmer-v5", 8),
    ],
)
def test_make_returns_env_with_obs_shape(name, obs_dim):
    """make() builds an env whose initial observation has the expected shape."""
    env = mjx.make(name)
    state = env.init(jrd.PRNGKey(0))
    assert env.obs(state).shape == (obs_dim,)


def test_make_unknown_name_raises():
    """make() raises ValueError for an unregistered environment name."""
    with pytest.raises(ValueError):
        mjx.make("Nonexistent-v0")


def test_step_shapes_scalar():
    """step() returns per-env scalar rew/term/trun and a vector observation."""
    env = mjx.make("Hopper-v5")
    key = jrd.PRNGKey(1)
    state = env.init(key)
    result, new_state = env.step(key, jnp.zeros(HOPPER_NU), state)

    assert result.nobs.shape == (HOPPER_OBS,)
    assert result.rew.shape == ()
    assert result.term.shape == ()
    assert result.trun.shape == ()
    assert not bool(result.trun)


def test_reset_applies_noise():
    """reset() perturbs the default pose, so observations differ from init."""
    env = mjx.make("Hopper-v5")
    key = jrd.PRNGKey(2)
    state = env.init(key)
    init_obs = env.obs(state)
    reset_obs, _ = env.reset(key, state)

    assert reset_obs.shape == (HOPPER_OBS,)
    assert not jnp.allclose(init_obs, reset_obs)


def test_reset_is_deterministic_in_key():
    """reset() with the same key gives identical observations; different keys differ."""
    env = mjx.make("Hopper-v5")
    state = env.init(jrd.PRNGKey(0))

    obs_a, _ = env.reset(jrd.PRNGKey(3), state)
    obs_b, _ = env.reset(jrd.PRNGKey(3), state)
    obs_c, _ = env.reset(jrd.PRNGKey(4), state)

    assert jnp.array_equal(obs_a, obs_b)
    assert not jnp.array_equal(obs_a, obs_c)


# =============================================================================
# Sampler integration (float32)
# =============================================================================


def test_mc_scalar_path():
    """Plain Mc samples transitions with scalar reward/termination."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_episode_len=1000, env=env)
    key = jrd.PRNGKey(6)
    env_state = env.init(key)
    mc_state = mc.init(key, env_state)

    transition, _ = mc.sample(jnp.zeros(HOPPER_NU), mc_state)
    assert transition.obs.shape == (HOPPER_OBS,)
    assert transition.rew.shape == ()
    assert transition.term.shape == ()


def test_vecmc_sample_shapes():
    """VecMc samples batched transitions with correct per-env shapes."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_episode_len=1000, env=env)
    vec_mc = VecMc(mc=mc)
    key = jrd.PRNGKey(7)
    env_state = env.init(key)
    keys = jrd.split(key, NUM_ENVS)
    mc_state = vec_mc.init(keys, env_state)

    actions = jrd.uniform(key, (NUM_ENVS, HOPPER_NU), minval=-1.0, maxval=1.0)
    transition, _ = vec_mc.sample(actions, mc_state)

    assert transition.obs.shape == (NUM_ENVS, HOPPER_OBS)
    assert transition.act.shape == (NUM_ENVS, HOPPER_NU)
    assert transition.rew.shape == (NUM_ENVS,)
    assert transition.term.shape == (NUM_ENVS,)


def test_jit_step():
    """env.step compiles and runs under jax.jit."""
    env = mjx.make("Hopper-v5")
    key = jrd.PRNGKey(8)
    state = env.init(key)
    result, _ = jax.jit(env.step)(key, jnp.zeros(HOPPER_NU), state)
    assert result.nobs.shape == (HOPPER_OBS,)


def test_roll_chain_jit():
    """Imc + VecMc + Roll collects batched trajectories under jit."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_episode_len=1000, env=env)
    vec_mc = VecMc(mc=mc)
    key = jrd.PRNGKey(9)

    env_state = env.init(key)
    k1, k2, k3 = jrd.split(key, 3)
    mc_state = vec_mc.init(jrd.split(k1, NUM_ENVS), env_state)

    imc = Imc(agent=RandomAgent(), mc=vec_mc)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    seqlen = 16
    roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
    trajectory, imc_state = jax.jit(roll.sample)(imc_state)

    assert trajectory.mc.obs.shape == (NUM_ENVS, seqlen, HOPPER_OBS)
    assert trajectory.dec.act.shape == (NUM_ENVS, seqlen, HOPPER_NU)
    assert trajectory.mc.rew.shape == (NUM_ENVS, seqlen)


def test_roll_single_env():
    """Imc + single-env Mc + Roll collects a trajectory."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_episode_len=1000, env=env)
    imc = Imc(agent=ScalarRandomAgent(), mc=mc)
    key = jrd.PRNGKey(30)

    k1, k2, k3 = jrd.split(key, 3)
    env_state = env.init(k1)
    mc_state = mc.init(k2, env_state)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k3))

    seqlen = 10
    roll = Roll(imc=imc, seqlen=seqlen)
    trajectory, _ = roll.sample(imc_state)

    assert trajectory.mc.obs.shape == (seqlen, HOPPER_OBS)
    assert trajectory.dec.act.shape == (seqlen, HOPPER_NU)
