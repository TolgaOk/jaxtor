"""Tests for Gymnasium environment adapter with custom_vmap."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from chex import dataclass
from jaxtor.env import gymnasium
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.rollout import Roll


# =============================================================================
# Fixtures
# =============================================================================

NUM_ENVS = 4
OBS_DIM = 4  # CartPole-v1 obs dimension
MAX_EPISODE_LEN = 500
QUEUE_SIZE = 10


def _make_env(num_envs=NUM_ENVS):
    return gymnasium.make("CartPole-v1", num_envs=num_envs)


def _make_mc(env):
    return Mc(max_episode_len=MAX_EPISODE_LEN, queue_size=QUEUE_SIZE, env=env)


def _init_vec(key, env):
    """Init env + mc + vec_mc, return (vec_mc, mc_state)."""
    mc = _make_mc(env)
    vec_mc = VecMc(mc=mc)
    env_state = env.init(key)
    keys = jrd.split(key, env._num_envs)
    mc_state = vec_mc.init(keys, env_state)
    return vec_mc, mc_state


@dataclass
class AgentState:
    key: chex.PRNGKey


class RandomAgent:
    """Agent that samples random discrete actions (vectorized path)."""

    State = AgentState

    def act(self, obs, state):
        key, subkey = jrd.split(state.key)
        actions = jrd.randint(subkey, (obs.shape[0],), 0, 2)
        return actions, AgentState(key=key)


class ScalarRandomAgent:
    """Agent that samples a single random discrete action (scalar path)."""

    State = AgentState

    def act(self, obs, state):
        key, subkey = jrd.split(state.key)
        action = jrd.randint(subkey, (), 0, 2)
        return action, AgentState(key=key)


# =============================================================================
# Init Tests
# =============================================================================


def test_init_returns_template_state():
    """Init returns a single-env template state."""
    env = _make_env()
    key = jrd.PRNGKey(0)
    state = env.init(key)

    assert state.obs.shape == (OBS_DIM,)
    assert state.reset_obs.shape == (OBS_DIM,)
    assert jnp.array_equal(state.obs, state.reset_obs)


def test_vecmc_init_shapes():
    """VecMc.init produces per-env Mc states with correct shapes."""
    env = _make_env()
    key = jrd.PRNGKey(0)
    vec_mc, mc_state = _init_vec(key, env)

    assert mc_state.key.shape == (NUM_ENVS, 2)
    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.env.obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.eps_rew_queue.shape == (NUM_ENVS, QUEUE_SIZE)
    assert mc_state.eps_len_queue.shape == (NUM_ENVS, QUEUE_SIZE)
    assert jnp.all(jnp.isnan(mc_state.eps_rew_queue))


def test_vecmc_init_per_env_obs():
    """VecMc.init gives each env its own initial observation."""
    env = _make_env()
    key = jrd.PRNGKey(0)
    vec_mc, mc_state = _init_vec(key, env)

    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)
    assert jnp.all(jnp.isfinite(mc_state.last_obs))


def test_default_num_envs_is_single():
    """make() with no num_envs creates a single-env adapter."""
    env = gymnasium.make("CartPole-v1")
    assert env._num_envs == 1

    key = jrd.PRNGKey(0)
    state = env.init(key)
    assert state.obs.shape == (OBS_DIM,)


def test_obs_accessor():
    """env.obs() returns the current observation from state."""
    env = _make_env()
    key = jrd.PRNGKey(20)
    state = env.init(key)

    obs = env.obs(state)
    assert obs.shape == (OBS_DIM,)
    assert jnp.array_equal(obs, state.obs)


def test_raw_step_reset_single_env():
    """Direct env.step() and env.reset() work without Mc wrapper."""
    env = gymnasium.make("CartPole-v1")
    key = jrd.PRNGKey(21)
    state = env.init(key)

    key, step_key = jrd.split(key)
    step_result, new_state = env.step(step_key, jnp.int32(0), state)

    assert step_result.nobs.shape == (OBS_DIM,)
    assert step_result.rew.shape == ()
    assert step_result.term.shape == ()
    assert step_result.trun.shape == ()
    assert new_state.obs.shape == (OBS_DIM,)
    assert new_state.reset_obs.shape == (OBS_DIM,)

    key, reset_key = jrd.split(key)
    reset_obs, reset_state = env.reset(reset_key, new_state)
    assert reset_obs.shape == (OBS_DIM,)


def test_raw_vmap_step_without_mc():
    """Direct vmap(env.step) works without VecMc."""
    env = _make_env()
    key = jrd.PRNGKey(22)
    env.init(key)
    init_obs = env._init_obs[0]
    batched_state = gymnasium.GymEnv.State(obs=init_obs, reset_obs=init_obs)

    keys = jrd.split(key, NUM_ENVS)
    acts = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    step_fn = jax.vmap(env.step)
    step_results, new_states = step_fn(keys, acts, batched_state)

    assert step_results.nobs.shape == (NUM_ENVS, OBS_DIM)
    assert step_results.rew.shape == (NUM_ENVS,)
    assert new_states.obs.shape == (NUM_ENVS, OBS_DIM)


def test_key_seeding_different_keys():
    """Different keys produce different initial observations."""
    env1 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    state1 = env1.init(jrd.PRNGKey(0))
    obs1 = env1._init_obs[0].copy()

    env2 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    state2 = env2.init(jrd.PRNGKey(999))
    obs2 = env2._init_obs[0].copy()

    assert not jnp.array_equal(obs1, obs2)


def test_key_seeding_reproducible():
    """Same key produces identical initial observations."""
    seed = jrd.PRNGKey(42)

    env1 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    env1.init(seed)
    obs1 = env1._init_obs[0].copy()

    env2 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    env2.init(seed)
    obs2 = env2._init_obs[0].copy()

    assert jnp.array_equal(obs1, obs2)


def test_per_env_keys_differ():
    """VecMc.init gives each env a distinct PRNG key."""
    env = _make_env()
    key = jrd.PRNGKey(23)
    _, mc_state = _init_vec(key, env)

    for i in range(NUM_ENVS):
        for j in range(i + 1, NUM_ENVS):
            assert not jnp.array_equal(mc_state.key[i], mc_state.key[j])


def test_keys_advance_after_sample():
    """PRNG keys in Mc state change after each sample call."""
    env = _make_env()
    key = jrd.PRNGKey(24)
    vec_mc, mc_state = _init_vec(key, env)

    old_keys = mc_state.key.copy()
    acts = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    _, mc_state = vec_mc.sample(acts, mc_state)

    assert not jnp.array_equal(old_keys, mc_state.key)


# =============================================================================
# VecMc.sample Tests
# =============================================================================


def test_vecmc_sample_shapes():
    """VecMc.sample returns transitions with correct per-env shapes."""
    env = _make_env()
    key = jrd.PRNGKey(1)
    vec_mc, mc_state = _init_vec(key, env)

    actions = jrd.randint(key, (NUM_ENVS,), 0, 2)
    transition, new_state = vec_mc.sample(actions, mc_state)

    assert transition.obs.shape == (NUM_ENVS, OBS_DIM)
    assert transition.act.shape == (NUM_ENVS,)
    assert transition.rew.shape == (NUM_ENVS,)
    assert transition.term.shape == (NUM_ENVS,)
    assert transition.trun.shape == (NUM_ENVS,)
    assert transition.nobs.shape == (NUM_ENVS, OBS_DIM)


def test_vecmc_sample_updates_state():
    """VecMc.sample advances episode index and accumulates reward."""
    env = _make_env()
    key = jrd.PRNGKey(2)
    vec_mc, mc_state = _init_vec(key, env)

    actions = jrd.randint(key, (NUM_ENVS,), 0, 2)
    _, new_state = vec_mc.sample(actions, mc_state)

    assert jnp.all(new_state.eps_idx >= 0)
    assert jnp.all(new_state.eps_rew >= 0)


def test_vecmc_consecutive_observations_match():
    """Transition nobs matches next transition obs when episode continues."""
    env = _make_env()
    key = jrd.PRNGKey(3)
    vec_mc, mc_state = _init_vec(key, env)

    actions = jrd.randint(key, (NUM_ENVS,), 0, 2)
    t1, mc_state = vec_mc.sample(actions, mc_state)

    actions = jrd.randint(jrd.fold_in(key, 1), (NUM_ENVS,), 0, 2)
    t2, _ = vec_mc.sample(actions, mc_state)

    done = jnp.logical_or(t1.term, t1.trun)
    continuing = ~done
    if jnp.any(continuing):
        assert jnp.allclose(t1.nobs[continuing], t2.obs[continuing])


# =============================================================================
# Auto-reset Tests
# =============================================================================


def test_auto_reset_triggers_episode_stats():
    """Running many steps triggers auto-resets and populates episode queues."""
    env = _make_env()
    key = jrd.PRNGKey(4)
    vec_mc, mc_state = _init_vec(key, env)

    for i in range(200):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = vec_mc.sample(actions, mc_state)

    assert not jnp.all(jnp.isnan(mc_state.eps_rew_queue))
    assert not jnp.all(jnp.isnan(mc_state.eps_len_queue))


def test_auto_reset_obs_continuity():
    """After auto-reset, last_obs should be a valid initial observation."""
    env = _make_env()
    key = jrd.PRNGKey(5)
    vec_mc, mc_state = _init_vec(key, env)

    for i in range(200):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = vec_mc.sample(actions, mc_state)

    assert jnp.all(jnp.isfinite(mc_state.last_obs))
    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)


# =============================================================================
# JIT Tests
# =============================================================================


def test_jit_vecmc_sample():
    """VecMc.sample works under jax.jit."""
    env = _make_env()
    key = jrd.PRNGKey(6)
    vec_mc, mc_state = _init_vec(key, env)

    jit_sample = jax.jit(vec_mc.sample)
    actions = jrd.randint(key, (NUM_ENVS,), 0, 2)
    transition, new_state = jit_sample(actions, mc_state)

    assert transition.obs.shape == (NUM_ENVS, OBS_DIM)
    assert transition.rew.shape == (NUM_ENVS,)


def test_jit_multiple_steps():
    """Multiple JIT-compiled steps work without recompilation issues."""
    env = _make_env()
    key = jrd.PRNGKey(7)
    vec_mc, mc_state = _init_vec(key, env)

    jit_sample = jax.jit(vec_mc.sample)
    for i in range(20):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = jit_sample(actions, mc_state)

    assert jnp.all(jnp.isfinite(mc_state.last_obs))


# =============================================================================
# Full Roll Chain Tests
# =============================================================================


def test_roll_eager():
    """Roll.sample collects trajectories with correct shapes."""
    env = _make_env()
    key = jrd.PRNGKey(8)
    vec_mc, mc_state = _init_vec(key, env)

    agent = RandomAgent()
    imc = Imc(agent=agent, mc=vec_mc)
    k1, k2 = jrd.split(key)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    seqlen = 32
    roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
    transitions, new_state = roll.sample(imc_state)

    assert transitions.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)
    assert transitions.rew.shape == (NUM_ENVS, seqlen)
    assert transitions.act.shape == (NUM_ENVS, seqlen)
    assert transitions.term.shape == (NUM_ENVS, seqlen)
    assert transitions.nobs.shape == (NUM_ENVS, seqlen, OBS_DIM)


def test_roll_jit():
    """JIT-compiled Roll.sample works end-to-end."""
    env = _make_env()
    key = jrd.PRNGKey(9)
    vec_mc, mc_state = _init_vec(key, env)

    agent = RandomAgent()
    imc = Imc(agent=agent, mc=vec_mc)
    k1, k2 = jrd.split(key)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    seqlen = 16
    roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
    jit_roll = jax.jit(roll.sample)

    transitions, imc_state = jit_roll(imc_state)
    assert transitions.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)

    transitions, imc_state = jit_roll(imc_state)
    assert transitions.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)


def test_roll_multiple_iterations_with_metrics():
    """Multiple Roll iterations followed by metrics computation."""
    env = _make_env()
    key = jrd.PRNGKey(10)
    vec_mc, mc_state = _init_vec(key, env)

    agent = RandomAgent()
    imc = Imc(agent=agent, mc=vec_mc)
    k1, k2 = jrd.split(key)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    roll = Roll(imc=imc, seqlen=64, seq_axis=1)
    jit_roll = jax.jit(roll.sample)

    for _ in range(10):
        _, imc_state = jit_roll(imc_state)

    metrics, _ = vec_mc.metrics(imc_state.mc)

    assert jnp.isfinite(metrics.avg_eps_rew)
    assert jnp.isfinite(metrics.avg_eps_len)
    assert metrics.avg_eps_len > 0


# =============================================================================
# Scalar Path Tests (num_envs=None, plain Mc)
# =============================================================================


def test_scalar_path_single_env():
    """Default num_envs with plain Mc works without vmap."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, queue_size=5, env=env)
    key = jrd.PRNGKey(11)

    env_state = env.init(key)
    mc_state = mc.init(key, env_state)

    assert mc_state.last_obs.shape == (OBS_DIM,)

    action = jnp.array(1)
    transition, mc_state = mc.sample(action, mc_state)

    assert transition.obs.shape == (OBS_DIM,)
    assert transition.rew.shape == ()
    assert transition.term.shape == ()


def test_scalar_path_multiple_steps():
    """Scalar path runs many steps with auto-reset."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, queue_size=5, env=env)
    key = jrd.PRNGKey(12)

    env_state = env.init(key)
    mc_state = mc.init(key, env_state)

    for i in range(100):
        action = jrd.randint(jrd.fold_in(key, i), (), 0, 2)
        _, mc_state = mc.sample(action, mc_state)

    assert jnp.all(jnp.isfinite(mc_state.last_obs))
    assert not jnp.all(jnp.isnan(mc_state.eps_rew_queue))


# =============================================================================
# Metrics Tests
# =============================================================================


def test_metrics_reasonable_values():
    """Metrics produce reasonable values for CartPole random policy."""
    env = _make_env()
    key = jrd.PRNGKey(13)
    vec_mc, mc_state = _init_vec(key, env)

    for i in range(500):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = vec_mc.sample(actions, mc_state)

    metrics, _ = vec_mc.metrics(mc_state)

    assert metrics.avg_eps_len > 5
    assert metrics.avg_eps_len < 100
    assert metrics.avg_eps_rew > 0


def test_metrics_refresh_clears_queues():
    """Metrics call refreshes episode queues."""
    env = _make_env()
    key = jrd.PRNGKey(14)
    vec_mc, mc_state = _init_vec(key, env)

    for i in range(200):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = vec_mc.sample(actions, mc_state)

    _, refreshed_state = vec_mc.metrics(mc_state)
    assert jnp.all(jnp.isnan(refreshed_state.eps_rew_queue))
    assert jnp.all(jnp.isnan(refreshed_state.eps_len_queue))


# =============================================================================
# Single-env Roll Tests
# =============================================================================


def test_roll_single_env():
    """Imc + single-env Mc + Roll collects trajectories."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, queue_size=5, env=env)
    agent = ScalarRandomAgent()
    imc = Imc(agent=agent, mc=mc)

    key = jrd.PRNGKey(30)
    k1, k2, k3 = jrd.split(key, 3)
    env_state = env.init(k1)
    mc_state = mc.init(k2, env_state)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k3))

    seqlen = 10
    roll = Roll(imc=imc, seqlen=seqlen)
    transitions, new_state = roll.sample(imc_state)

    assert transitions.obs.shape == (seqlen, OBS_DIM)
    assert transitions.nobs.shape == (seqlen, OBS_DIM)
    assert transitions.rew.shape == (seqlen,)
    assert transitions.act.shape == (seqlen,)


# =============================================================================
# Trajectory Reproducibility Tests
# =============================================================================


def test_trajectory_reproducible_with_same_key():
    """Same key on fresh envs produces identical trajectories."""
    seed = jrd.PRNGKey(50)
    seqlen = 16

    def _collect(seed):
        env = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
        mc = _make_mc(env)
        vec_mc = VecMc(mc=mc)
        agent = RandomAgent()
        imc = Imc(agent=agent, mc=vec_mc)

        k1, k2, k3 = jrd.split(seed, 3)
        env_state = env.init(k1)
        keys = jrd.split(k2, NUM_ENVS)
        mc_state = vec_mc.init(keys, env_state)
        imc_state = imc.init(mc=mc_state, agent=AgentState(key=k3))

        roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
        transitions, _ = roll.sample(imc_state)
        env._vec_env.close()
        return transitions

    t1 = _collect(seed)
    t2 = _collect(seed)

    assert jnp.array_equal(t1.obs, t2.obs)
    assert jnp.array_equal(t1.act, t2.act)
    assert jnp.array_equal(t1.rew, t2.rew)
    assert jnp.array_equal(t1.nobs, t2.nobs)
    assert jnp.array_equal(t1.term, t2.term)
    assert jnp.array_equal(t1.trun, t2.trun)


def test_trajectory_differs_with_different_key():
    """Different keys on fresh envs produce different trajectories."""
    seqlen = 16

    def _collect(seed):
        env = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
        mc = _make_mc(env)
        vec_mc = VecMc(mc=mc)
        agent = RandomAgent()
        imc = Imc(agent=agent, mc=vec_mc)

        k1, k2, k3 = jrd.split(seed, 3)
        env_state = env.init(k1)
        keys = jrd.split(k2, NUM_ENVS)
        mc_state = vec_mc.init(keys, env_state)
        imc_state = imc.init(mc=mc_state, agent=AgentState(key=k3))

        roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)
        transitions, _ = roll.sample(imc_state)
        env._vec_env.close()
        return transitions

    t1 = _collect(jrd.PRNGKey(0))
    t2 = _collect(jrd.PRNGKey(999))

    assert not jnp.array_equal(t1.obs, t2.obs)
