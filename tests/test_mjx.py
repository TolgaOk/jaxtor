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
from typing import Protocol, TypeGuard

import gymnasium
from mujoco import mjx as _mjx

from jaxtor.env import mjx
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.rollout import Roll

pytestmark = pytest.mark.backend


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


class GymModel(Protocol):
    """MuJoCo model fields used by the parity tests."""

    nq: int
    nv: int


class GymData(Protocol):
    """MuJoCo data fields used by the parity tests."""

    qpos: np.ndarray
    qvel: np.ndarray


class GymMujocoEnv(Protocol):
    """Backend-specific Gymnasium capability exercised by parity tests."""

    data: GymData
    model: GymModel
    init_qpos: np.ndarray
    init_qvel: np.ndarray
    is_healthy: bool

    def reset(self, *, seed: int | None = None): ...

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None: ...

    def _get_obs(self) -> np.ndarray: ...


def _is_gym_mujoco(env: object) -> TypeGuard[GymMujocoEnv]:
    """Check the private MuJoCo surface absent from Gymnasium's base type."""
    return all(
        hasattr(env, name)
        for name in (
            "data",
            "model",
            "init_qpos",
            "init_qvel",
            "is_healthy",
            "reset",
            "step",
            "set_state",
            "_get_obs",
        )
    )


def _gym_mujoco(name: str) -> GymMujocoEnv:
    """Create and validate the concrete Gymnasium MuJoCo environment."""
    env = gymnasium.make(name).unwrapped
    if not _is_gym_mujoco(env):
        raise TypeError(f"{name!r} is not a Gymnasium MuJoCo environment")
    return env


@dataclass
class AgentState:
    key: chex.PRNGKey


class RandomAgent:
    """Agent sampling batched random continuous actions (vectorized path)."""

    State = AgentState

    def act(self, obs, state):
        key, subkey = jrd.split(state.key)
        act = jrd.uniform(subkey, (obs.shape[0], HOPPER_NU), minval=-1.0, maxval=1.0)
        return act, AgentState(key=key)


class ScalarRandomAgent:
    """Agent sampling a single random continuous action (scalar path)."""

    State = AgentState

    def act(self, obs, state):
        key, subkey = jrd.split(state.key)
        act = jrd.uniform(subkey, (HOPPER_NU,), minval=-1.0, maxval=1.0)
        return act, AgentState(key=key)


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
        raw = _gym_mujoco(name)
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
        raw = _gym_mujoco(name)
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
            assert obs_m.shape == (obs_dim,)
            term_m = bool(env.task.terminal(data))

            max_obs = max(max_obs, np.abs(obs_g - obs_m).max())
            if terminates:
                term_g = not raw.is_healthy
                term_mismatch += int(term_g != term_m)
                saw_unhealthy = saw_unhealthy or term_g
            else:
                term_mismatch += int(term_m)

        assert max_obs < 1e-9
        assert term_mismatch == 0
        assert saw_unhealthy == terminates


@pytest.mark.parametrize("name, nu, obs_dim, terminates, clips", ENV_SPECS)
def test_qvel_clip_parity(name, nu, obs_dim, terminates, clips):
    """Observation velocity clipping to [-10, 10] matches Gymnasium."""
    if not clips:
        pytest.skip(f"{name} does not clip observation velocities")
    with jax.enable_x64():
        raw = _gym_mujoco(name)
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


def test_mc_truncation_carries_a_fresh_reset_observation():
    """A sampler limit preserves true nobs and resets MJX before the next step."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_eps_len=1, env=env)
    key = jrd.PRNGKey(31)
    state = mc.init(key, env.init(key))

    transition, state = jax.jit(mc.sample)(jnp.zeros(HOPPER_NU), state)
    reset_obs = state.last_obs
    following, state = jax.jit(mc.sample)(jnp.zeros(HOPPER_NU), state)

    assert transition.trun
    assert state.eps_idx == 0
    assert not jnp.array_equal(transition.nobs, reset_obs)
    assert jnp.array_equal(reset_obs, following.obs)
    assert jnp.array_equal(state.last_obs, env.obs(state.env))


# =============================================================================
# Sampler integration (float32)
# =============================================================================


def test_mc_scalar_path():
    """Plain Mc samples transitions with scalar reward/termination."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_eps_len=1000, env=env)
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
    mc = Mc(max_eps_len=1000, env=env)
    vec_mc = VecMc(mc=mc)
    key = jrd.PRNGKey(7)
    keys = jrd.split(key, NUM_ENVS)
    env_state = jax.vmap(env.init)(keys)
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
    mc = Mc(max_eps_len=1000, env=env)
    vec_mc = VecMc(mc=mc)
    key = jrd.PRNGKey(9)

    k1, k2, k3 = jrd.split(key, 3)
    keys = jrd.split(k1, NUM_ENVS)
    mc_state = vec_mc.init(keys, jax.vmap(env.init)(keys))

    imc = Imc(agent=RandomAgent(), mc=vec_mc)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    seq_len = 16
    roll = Roll(imc=imc, seq_len=seq_len, seq_axis=1)
    seq, imc_state = jax.jit(roll.sample)(imc_state)

    assert seq.obs.shape == (NUM_ENVS, seq_len, HOPPER_OBS)
    assert seq.act.shape == (NUM_ENVS, seq_len, HOPPER_NU)
    assert seq.rew.shape == (NUM_ENVS, seq_len)


def test_roll_single_env():
    """Imc + single-env Mc + Roll collects a sequence."""
    env = mjx.make("Hopper-v5")
    mc = Mc(max_eps_len=1000, env=env)
    imc = Imc(agent=ScalarRandomAgent(), mc=mc)
    key = jrd.PRNGKey(30)

    k1, k2, k3 = jrd.split(key, 3)
    env_state = env.init(k1)
    mc_state = mc.init(k2, env_state)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k3))

    seq_len = 10
    roll = Roll(imc=imc, seq_len=seq_len)
    seq, _ = roll.sample(imc_state)

    assert seq.obs.shape == (seq_len, HOPPER_OBS)
    assert seq.act.shape == (seq_len, HOPPER_NU)
