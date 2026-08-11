"""Tests for the EnvPool backend (fast CPU MuJoCo through the GymEnv adapter).

EnvPool is an optional dependency, so the whole module skips when it is absent.
Coverage focuses on what EnvPool adds over ``jaxtor.env.gymnasium``: the
NEXT_STEP -> SAME_STEP autoreset conversion, subset vmap (a vmap narrower than
the pool), and key-driven seeding of a pool whose RNG is fixed at construction.
"""

import warnings

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import numpy as np
import pytest
from chex import dataclass

from jaxtor.env import envpool, gymnasium
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.rollout import Roll

try:
    envpool.import_envpool()
except ImportError:  # pragma: no cover - depends on the host environment
    pytest.skip("envpool is not installed", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

ENV_ID = "Hopper-v5"
NUM_ENVS = 4
OBS_DIM = 11
ACT_DIM = 3
MAX_EPISODE_LEN = 1000
QUEUE_SIZE = 10


@pytest.fixture(autouse=True)
def _close_backend_runtimes():
    """Keep the shared process-local runtime registry isolated across tests."""
    yield
    gymnasium.close_all_runtimes()


def _make_env(num_envs=NUM_ENVS, **kwargs):
    kwargs.setdefault("max_episode_steps", MAX_EPISODE_LEN)
    return envpool.make(ENV_ID, num_envs=num_envs, **kwargs)


def _init_vec(key, env, axis_size=None, max_len=MAX_EPISODE_LEN):
    """Init env + mc + vec_mc over ``axis_size`` envs; return (vec_mc, mc_state)."""
    axis_size = env.num_envs if axis_size is None else axis_size
    vec_mc = VecMc(mc=Mc(max_episode_len=max_len, queue_size=QUEUE_SIZE, env=env))
    env_state = env.init(key)
    return vec_mc, vec_mc.init(jrd.split(key, axis_size), env_state)


@dataclass
class AgentState:
    key: chex.PRNGKey


class RandomAgent:
    """Agent that samples random continuous actions (vectorized path)."""

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        key, subkey = jrd.split(state.key)
        actions = jrd.uniform(subkey, (obs.shape[0], ACT_DIM), minval=-1.0, maxval=1.0)
        return self.Decision(act=actions), AgentState(key=key)


# =============================================================================
# Construction Tests
# =============================================================================


def test_make_returns_env_with_shapes():
    """make() reports the per-env obs/action shapes of the MuJoCo env."""
    env = _make_env()

    assert env.num_envs == NUM_ENVS
    assert env.obs_shape == (OBS_DIM,)
    assert env.act_shape == (ACT_DIM,)


def test_init_returns_template_state():
    """Init returns a single-env template state."""
    env = _make_env()
    state = env.init(jrd.PRNGKey(0))

    assert state.obs.shape == (OBS_DIM,)
    assert state.reset_obs.shape == (OBS_DIM,)
    assert jnp.array_equal(state.obs, state.reset_obs)
    assert state.runtime.shape == (2,)
    assert state.runtime_id.shape == ()
    assert state.token.shape == ()


def test_unknown_env_id_raises():
    """An id EnvPool does not ship raises rather than silently falling back."""
    with pytest.raises(Exception, match="not supported"):
        envpool.make("NotAnEnv-v0", num_envs=1)


def test_num_envs_must_be_positive():
    """EnvPool requires a positive vector-pool size."""
    with pytest.raises(ValueError, match="num_envs must be positive"):
        envpool.make(ENV_ID, num_envs=0)


def test_vecmc_init_shapes():
    """VecMc.init produces per-env Mc states with correct shapes."""
    env = _make_env()
    _, mc_state = _init_vec(jrd.PRNGKey(0), env)

    assert mc_state.key.shape == (NUM_ENVS, 2)
    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.env.obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.eps_rew_queue.shape == (NUM_ENVS, QUEUE_SIZE)
    assert jnp.all(jnp.isnan(mc_state.eps_rew_queue))


# =============================================================================
# Seeding Tests
# =============================================================================


def test_same_key_gives_same_initial_obs():
    """Same key produces identical initial observations."""
    key = jrd.PRNGKey(42)
    obs1 = _make_env(2).init(key).obs
    obs2 = _make_env(2).init(key).obs

    assert jnp.array_equal(obs1, obs2)


def test_different_keys_give_different_initial_obs():
    """Different keys produce different initial observations.

    EnvPool fixes its RNG at construction, so this only holds because the
    runtime factory builds each pool from the key-derived seed. Without that,
    every key yields EnvPool's default stream and seeds are silently ignored.
    """
    obs1 = _make_env(2).init(jrd.PRNGKey(0)).obs
    obs2 = _make_env(2).init(jrd.PRNGKey(999)).obs

    assert not jnp.array_equal(obs1, obs2)


def test_reinit_same_env_is_stable():
    """Re-initializing one env with the same key reproduces the observation."""
    env = _make_env(2)

    assert jnp.array_equal(env.init(jrd.PRNGKey(7)).obs, env.init(jrd.PRNGKey(7)).obs)


def test_init_does_not_warn():
    """init() must not trip EnvPool's ignored-seed / abandoned-seed warnings."""
    env = _make_env(2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env.init(jrd.PRNGKey(0))

    seeding = [w for w in caught if "seed" in str(w.message).lower()]
    assert not seeding, [str(w.message) for w in seeding]


# =============================================================================
# Sampling Tests
# =============================================================================


def test_vecmc_sample_shapes():
    """VecMc.sample returns transitions with correct per-env shapes."""
    env = _make_env()
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(1), env)

    acts = jnp.zeros((NUM_ENVS, ACT_DIM))
    transition, _ = vec_mc.sample(acts, mc_state)

    assert transition.obs.shape == (NUM_ENVS, OBS_DIM)
    assert transition.act.shape == (NUM_ENVS, ACT_DIM)
    assert transition.rew.shape == (NUM_ENVS,)
    assert transition.term.shape == (NUM_ENVS,)
    assert transition.trun.shape == (NUM_ENVS,)
    assert transition.nobs.shape == (NUM_ENVS, OBS_DIM)


def test_vecmc_consecutive_observations_match():
    """Transition nobs matches next transition obs when the episode continues."""
    env = _make_env()
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(3), env)

    acts = jnp.zeros((NUM_ENVS, ACT_DIM))
    t1, mc_state = vec_mc.sample(acts, mc_state)
    t2, _ = vec_mc.sample(acts, mc_state)

    continuing = ~jnp.logical_or(t1.term, t1.trun)
    if jnp.any(continuing):
        assert jnp.allclose(t1.nobs[continuing], t2.obs[continuing])


def test_scalar_path_single_env():
    """num_envs=1 works through plain Mc without vmap."""
    env = _make_env(1)
    mc = Mc(max_episode_len=MAX_EPISODE_LEN, queue_size=5, env=env)
    key = jrd.PRNGKey(11)

    mc_state = mc.init(key, env.init(key))
    assert mc_state.last_obs.shape == (OBS_DIM,)

    _, mc_state = mc.sample(jnp.zeros(ACT_DIM), mc_state)
    assert mc_state.last_obs.shape == (OBS_DIM,)


def test_jit_vecmc_sample():
    """VecMc.sample works under jax.jit."""
    env = _make_env()
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(6), env)

    transition, _ = jax.jit(vec_mc.sample)(jnp.zeros((NUM_ENVS, ACT_DIM)), mc_state)

    assert transition.obs.shape == (NUM_ENVS, OBS_DIM)
    assert jnp.all(jnp.isfinite(transition.rew))


# =============================================================================
# Subset vmap Tests (the reason from_factory takes allow_subset)
# =============================================================================


def test_subset_vmap_uses_prefix_of_pool():
    """A vmap narrower than the pool steps only the first m envs."""
    env = _make_env(8)
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(0), env, axis_size=4)

    assert mc_state.last_obs.shape == (4, OBS_DIM)

    transition, _ = vec_mc.sample(jnp.zeros((4, ACT_DIM)), mc_state)
    assert transition.nobs.shape == (4, OBS_DIM)
    assert transition.rew.shape == (4,)


def test_subset_vmap_matches_full_pool_prefix():
    """The first m envs behave identically whether or not the rest are stepped."""
    key = jrd.PRNGKey(5)
    acts_full = jnp.zeros((8, ACT_DIM))

    env_full = _make_env(8)
    vec_full, st_full = _init_vec(key, env_full, axis_size=8)
    t_full, _ = vec_full.sample(acts_full, st_full)

    env_sub = _make_env(8)
    vec_sub, st_sub = _init_vec(key, env_sub, axis_size=4)
    t_sub, _ = vec_sub.sample(acts_full[:4], st_sub)

    assert jnp.allclose(t_full.nobs[:4], t_sub.nobs)
    assert jnp.allclose(t_full.rew[:4], t_sub.rew)


def test_vmap_axis_larger_than_pool_raises():
    """A vmap wider than the pool is rejected."""
    env = _make_env(4)
    with pytest.raises(ValueError, match=r"axis_size \(8\) > num_envs \(4\)"):
        _init_vec(jrd.PRNGKey(0), env, axis_size=8)


# =============================================================================
# Auto-reset Tests (NEXT_STEP -> SAME_STEP conversion)
# =============================================================================


def _run_until_done(vec_mc, mc_state, n_envs, max_steps=200):
    """Drive the envs hard until at least one episode ends.

    Returns ``(transition, state_after, done_mask)`` for the first step on which
    any env finished.
    """
    acts = jnp.full((n_envs, ACT_DIM), 0.9)
    for _ in range(max_steps):
        transition, mc_state = vec_mc.sample(acts, mc_state)
        done = np.array(jnp.logical_or(transition.term, transition.trun))
        if done.any():
            return transition, mc_state, done
    raise AssertionError(f"no episode ended within {max_steps} steps")


def test_terminal_obs_survives_autoreset():
    """On a done step, nobs is the terminal obs, not the post-reset obs.

    This is the whole point of the NEXT_STEP -> SAME_STEP conversion: EnvPool
    returns the terminal obs on the done step, and ``SameStep`` must hand it
    over as ``final_obs`` while the sampler carries the reset obs forward.
    """
    env = _make_env()
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(4), env)
    transition, mc_state, done = _run_until_done(vec_mc, mc_state, NUM_ENVS)

    terminal = np.array(transition.nobs)[done]
    carried = np.array(mc_state.last_obs)[done]

    assert not np.allclose(terminal, carried), (
        "terminal obs equals the post-reset obs -- final_obs was lost"
    )
    assert np.all(np.isfinite(terminal))


def test_terminated_env_restarts_near_init_state():
    """After a termination the env restarts, so its obs returns to healthy range."""
    env = _make_env()
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(4), env)
    _, mc_state, done = _run_until_done(vec_mc, mc_state, NUM_ENVS)

    # Hopper's reset state has torso height ~1.25 (obs[0]); a fallen hopper is
    # well below the 0.7 healthy threshold.
    restarted_height = np.array(mc_state.last_obs)[done][:, 0]
    assert np.all(restarted_height > 0.7)


def test_auto_reset_populates_episode_stats():
    """Repeated resets fill the episode-statistics queues."""
    env = _make_env(max_episode_steps=20)
    vec_mc, mc_state = _init_vec(jrd.PRNGKey(4), env, max_len=20)

    for _ in range(80):
        _, mc_state = vec_mc.sample(jnp.zeros((NUM_ENVS, ACT_DIM)), mc_state)

    assert not jnp.all(jnp.isnan(mc_state.eps_rew_queue))
    metrics, _ = vec_mc.metrics(mc_state)
    assert jnp.isfinite(metrics.avg_eps_rew)
    assert metrics.avg_eps_len > 0


# =============================================================================
# Full Roll Chain Tests
# =============================================================================


def test_roll_chain_jit():
    """Imc + VecMc + Roll collects trajectories under jit."""
    env = _make_env()
    key = jrd.PRNGKey(9)
    vec_mc, mc_state = _init_vec(key, env)

    imc = Imc(agent=RandomAgent(), mc=vec_mc)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=jrd.split(key)[1]))

    seqlen = 16
    roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
    jit_roll = jax.jit(roll.sample)

    trajectory, imc_state = jit_roll(imc_state)
    assert trajectory.mc.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)
    assert trajectory.dec.act.shape == (NUM_ENVS, seqlen, ACT_DIM)

    trajectory, imc_state = jit_roll(imc_state)
    assert jnp.all(jnp.isfinite(trajectory.mc.rew))
