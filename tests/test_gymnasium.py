"""Tests for Gymnasium environment adapter with custom_vmap."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import pytest
from chex import dataclass
from jaxtor.env import gymnasium
from jaxtor.eval.mc import Eval as McEval
from jaxtor.sampler.stats import EpisodeStats
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.rollout import Roll


# =============================================================================
# Fixtures
# =============================================================================

NUM_ENVS = 4
OBS_DIM = 4  # CartPole-v1 obs dimension
MAX_EPISODE_LEN = 500
COMPONENT_NUM_ENVS = 3
COMPONENT_ROLLOUT_LEN = 4

ENV_CASES = [
    pytest.param("FrozenLake-v1", (), (), False, id="frozen-lake"),
    pytest.param("Taxi-v3", (), (), False, id="taxi"),
    pytest.param("MountainCar-v0", (2,), (), False, id="mountain-car"),
    pytest.param("Acrobot-v1", (6,), (), False, id="acrobot"),
    pytest.param("Pendulum-v1", (3,), (1,), True, id="pendulum"),
    pytest.param(
        "MountainCarContinuous-v0",
        (2,),
        (1,),
        True,
        id="continuous-mountain-car",
    ),
    pytest.param("Hopper-v5", (11,), (3,), True, id="hopper"),
    pytest.param("Walker2d-v5", (17,), (6,), True, id="walker"),
    pytest.param("HalfCheetah-v5", (17,), (6,), True, id="half-cheetah"),
    pytest.param("Swimmer-v5", (8,), (2,), True, id="swimmer"),
]


@pytest.fixture(autouse=True)
def _close_backend_runtimes():
    """Keep the process-local runtime registry isolated across tests."""
    yield
    gymnasium.close_all_runtimes()


def _make_env(num_envs=NUM_ENVS):
    return gymnasium.make("CartPole-v1", num_envs=num_envs)


def _make_mc(env):
    return Mc(max_episode_len=MAX_EPISODE_LEN, env=env)


def _init_vec(key, env):
    """Init env + mc + vec_mc, return (vec_mc, mc_state)."""
    mc = _make_mc(env)
    vec_mc = VecMc(mc=mc)
    env_state = env.init(key)
    keys = jrd.split(key, env.num_envs)
    mc_state = vec_mc.init(keys, env_state)
    return vec_mc, mc_state


@dataclass
class AgentState:
    key: chex.PRNGKey


class RandomAgent:
    """Agent that samples random discrete actions (vectorized path)."""

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        key, subkey = jrd.split(state.key)
        actions = jrd.randint(subkey, (obs.shape[0],), 0, 2)
        return self.Decision(act=actions), AgentState(key=key)


class ScalarRandomAgent:
    """Agent that samples a single random discrete action (scalar path)."""

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        key, subkey = jrd.split(state.key)
        action = jrd.randint(subkey, (), 0, 2)
        return self.Decision(act=action), AgentState(key=key)


@dataclass
class ZeroAgent:
    """Agent producing valid zero actions for array-shaped spaces."""

    obs_shape: tuple[int, ...]
    act_shape: tuple[int, ...]
    continuous: bool

    State = AgentState

    @dataclass
    class Decision:
        act: chex.Array

    def decide(self, obs, state):
        """Return one zero action for every leading observation batch index."""
        obs_ndim = len(self.obs_shape)
        batch_shape = obs.shape[: obs.ndim - obs_ndim]
        dtype = jnp.float32 if self.continuous else jnp.int32
        action = jnp.zeros((*batch_shape, *self.act_shape), dtype=dtype)
        return self.Decision(act=action), state


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
    assert state.runtime.shape == (2,)
    assert state.runtime_id.shape == ()
    assert state.token.shape == ()


def test_state_pytree_round_trip():
    """State contains only array leaves and survives pytree reconstruction."""
    env = _make_env()
    state = env.init(jrd.PRNGKey(0))

    leaves, structure = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(structure, leaves)

    assert len(leaves) == 3
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert isinstance(restored, gymnasium.GymEnv.State)
    chex.assert_trees_all_equal(restored, state)


def test_live_runtime_is_addressed_by_state():
    """The component owns config; State owns the opaque runtime capability."""
    env = _make_env()

    assert not hasattr(env, "_vec_env")
    state = env.init(jrd.PRNGKey(0))
    runtime_id = int(state.runtime_id)
    runtime = gymnasium.lookup_runtime(runtime_id)
    assert not runtime.closed

    env.close(state)
    with pytest.raises(RuntimeError, match="closed or unknown"):
        gymnasium.lookup_runtime(runtime_id)


def test_vecmc_init_shapes():
    """VecMc.init produces per-env Mc states with correct shapes."""
    env = _make_env()
    key = jrd.PRNGKey(0)
    vec_mc, mc_state = _init_vec(key, env)

    assert mc_state.key.shape == (NUM_ENVS, 2)
    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.env.obs.shape == (NUM_ENVS, OBS_DIM)
    assert mc_state.eps_idx.shape == (NUM_ENVS,)


def test_vecmc_init_per_env_obs():
    """VecMc.init gives each env its own initial observation."""
    env = _make_env()
    key = jrd.PRNGKey(0)
    vec_mc, mc_state = _init_vec(key, env)

    assert mc_state.last_obs.shape == (NUM_ENVS, OBS_DIM)
    assert jnp.all(jnp.isfinite(mc_state.last_obs))


def test_default_uses_scalar_runtime():
    """Omitting num_envs creates a scalar Gymnasium runtime."""
    env = gymnasium.make("CartPole-v1")
    assert env.num_envs is None

    key = jrd.PRNGKey(0)
    state = env.init(key)
    runtime = gymnasium.lookup_runtime(state.runtime_id)

    assert state.obs.shape == (OBS_DIM,)
    assert not hasattr(runtime.env, "num_envs")


def test_num_envs_one_uses_vector_runtime():
    """An explicit num_envs=1 creates a one-lane vector runtime."""
    env = gymnasium.make("CartPole-v1", num_envs=1)
    key = jrd.PRNGKey(1)
    state = env.init(key)
    runtime = gymnasium.lookup_runtime(state.runtime_id)
    vec_mc = VecMc(mc=_make_mc(env))
    mc_state = vec_mc.init(jrd.split(key, 1), state)

    transition, _ = vec_mc.sample(jnp.zeros(1, dtype=jnp.int32), mc_state)

    assert env.num_envs == 1
    assert getattr(runtime.env, "num_envs") == 1
    assert transition.obs.shape == (1, OBS_DIM)


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
    keys = jrd.split(key, NUM_ENVS)
    template = env.init(key)
    _, batched_state = jax.vmap(env.reset, in_axes=(0, None))(keys, template)
    acts = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    step_fn = jax.vmap(env.step)
    step_results, new_states = step_fn(keys, acts, batched_state)

    assert step_results.nobs.shape == (NUM_ENVS, OBS_DIM)
    assert step_results.rew.shape == (NUM_ENVS,)
    assert new_states.obs.shape == (NUM_ENVS, OBS_DIM)


def test_scalar_runtime_rejects_vmap():
    """A scalar runtime rejects batching and points to num_envs."""
    env = gymnasium.make("CartPole-v1")
    key = jrd.PRNGKey(23)
    state = env.init(key)

    with pytest.raises(ValueError, match="cannot vmap a scalar Gymnasium runtime"):
        jax.vmap(env.reset, in_axes=(0, None))(jrd.split(key, 1), state)


def test_async_runtime_requires_num_envs():
    """Async vectorization requires an explicit vector-pool size."""
    with pytest.raises(ValueError, match="async_envs requires an integer num_envs"):
        gymnasium.make("CartPole-v1", async_envs=True)


def test_key_seeding_different_keys():
    """Different keys produce different initial observations."""
    env1 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    state1 = env1.init(jrd.PRNGKey(0))
    obs1 = state1.obs.copy()

    env2 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    state2 = env2.init(jrd.PRNGKey(999))
    obs2 = state2.obs.copy()

    assert not jnp.array_equal(obs1, obs2)


def test_key_seeding_reproducible():
    """Same key produces identical initial observations."""
    seed = jrd.PRNGKey(42)

    env1 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    obs1 = env1.init(seed).obs.copy()

    env2 = gymnasium.make("CartPole-v1", num_envs=NUM_ENVS)
    obs2 = env2.init(seed).obs.copy()

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
    """VecMc.sample advances each lane's episode index."""
    env = _make_env()
    key = jrd.PRNGKey(2)
    vec_mc, mc_state = _init_vec(key, env)

    actions = jrd.randint(key, (NUM_ENVS,), 0, 2)
    _, new_state = vec_mc.sample(actions, mc_state)

    assert jnp.all(new_state.eps_idx >= 0)


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


def test_auto_reset_keeps_valid_episode_state():
    """Running across resets preserves valid observations and episode indices."""
    env = _make_env()
    key = jrd.PRNGKey(4)
    vec_mc, mc_state = _init_vec(key, env)

    for i in range(200):
        actions = jrd.randint(jrd.fold_in(key, i), (NUM_ENVS,), 0, 2)
        _, mc_state = vec_mc.sample(actions, mc_state)

    assert jnp.all(jnp.isfinite(mc_state.last_obs))
    assert jnp.all(mc_state.eps_idx >= 0)
    assert jnp.all(mc_state.eps_idx < MAX_EPISODE_LEN)


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


def test_async_vector_runtime():
    """The state-routed bridge also owns an AsyncVectorEnv runtime."""
    num_envs = 2
    env = gymnasium.make("CartPole-v1", num_envs=num_envs, async_envs=True)
    vec_mc = VecMc(mc=_make_mc(env))
    key = jrd.PRNGKey(72)
    state = vec_mc.init(jrd.split(key, num_envs), env.init(key))

    transition, state = jax.jit(vec_mc.sample)(
        jnp.zeros(num_envs, dtype=jnp.int32), state
    )

    assert transition.obs.shape == (num_envs, OBS_DIM)
    assert jnp.all(state.env.token == 1)
    env.close(state.env)


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


def test_scan_threads_runtime_token():
    """A callback-produced token orders every external step through scan."""
    env = _make_env()
    key = jrd.PRNGKey(70)
    vec_mc, mc_state = _init_vec(key, env)
    actions = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    seqlen = 24

    @jax.jit
    def sample_n(state):
        def body(carry, _):
            _, carry = vec_mc.sample(actions, carry)
            return carry, None

        return jax.lax.scan(body, state, None, length=seqlen)[0]

    mc_state = sample_n(mc_state)

    assert jnp.all(mc_state.env.token == seqlen)
    runtime = gymnasium.lookup_runtime(mc_state.env.runtime_id)
    assert runtime.token == seqlen


def test_one_compiled_sampler_routes_multiple_runtimes():
    """runtime_id remains dynamic data rather than a captured Python object."""
    env = _make_env()
    vec_mc = VecMc(mc=_make_mc(env))
    key_a, key_b = jrd.split(jrd.PRNGKey(71))
    state_a = vec_mc.init(jrd.split(key_a, NUM_ENVS), env.init(key_a))
    state_b = vec_mc.init(jrd.split(key_b, NUM_ENVS), env.init(key_b))
    runtime_a = state_a.env.runtime_id[0]
    runtime_b = state_b.env.runtime_id[0]
    assert runtime_a != runtime_b

    sample = jax.jit(vec_mc.sample)
    actions = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    _, state_a = sample(actions, state_a)
    _, state_b = sample(actions, state_b)

    assert jnp.all(state_a.env.token == 1)
    assert jnp.all(state_b.env.token == 1)


def test_nested_vmaps_route_independent_vector_runtimes():
    """Outer vmaps may batch sessions that each own one vector pool."""
    env = _make_env()
    vec_mc = VecMc(mc=_make_mc(env))
    session_shape = (2, 2)
    keys = jrd.split(jrd.PRNGKey(73), 4)
    flat_states = tuple(
        vec_mc.init(jrd.split(key, NUM_ENVS), env.init(key)) for key in keys
    )
    states = jax.tree.map(
        lambda *values: jnp.stack(values).reshape(*session_shape, *values[0].shape),
        *flat_states,
    )

    sample_sessions = jax.jit(jax.vmap(jax.vmap(vec_mc.sample)))
    transitions, states = sample_sessions(
        jnp.zeros((*session_shape, NUM_ENVS), dtype=jnp.int32), states
    )

    assert transitions.obs.shape == (*session_shape, NUM_ENVS, OBS_DIM)
    assert jnp.all(states.env.token == 1)


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

    assert transitions.mc.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)
    assert transitions.mc.rew.shape == (NUM_ENVS, seqlen)
    assert transitions.dec.act.shape == (NUM_ENVS, seqlen)
    assert transitions.mc.term.shape == (NUM_ENVS, seqlen)
    assert transitions.mc.nobs.shape == (NUM_ENVS, seqlen, OBS_DIM)


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
    assert transitions.mc.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)

    transitions, imc_state = jit_roll(imc_state)
    assert transitions.mc.obs.shape == (NUM_ENVS, seqlen, OBS_DIM)


def test_roll_multiple_iterations_with_metrics():
    """EpisodeStats accumulates completed episodes across vector rollouts."""
    env = _make_env()
    key = jrd.PRNGKey(10)
    vec_mc, mc_state = _init_vec(key, env)

    agent = RandomAgent()
    imc = Imc(agent=agent, mc=vec_mc)
    k1, k2 = jrd.split(key)
    imc_state = imc.init(mc=mc_state, agent=AgentState(key=k2))

    roll = Roll(imc=imc, seqlen=64, seq_axis=1)
    stats = EpisodeStats(seq_axis=1)
    stats_state = stats.init((NUM_ENVS,))
    jit_roll = jax.jit(roll.sample)

    for _ in range(10):
        trajectory, imc_state = jit_roll(imc_state)
        stats_state = stats.update(trajectory.mc, stats_state)

    metrics, _ = stats.drain(stats_state)

    assert jnp.isfinite(metrics.avg_eps_rew)
    assert jnp.isfinite(metrics.avg_eps_len)
    assert metrics.avg_eps_len > 0


def test_eval_returns_live_runtime_state():
    """Evaluation returns the runtime token needed by subsequent sampling."""
    env = _make_env()
    key = jrd.PRNGKey(74)
    vec_mc, mc_state = _init_vec(key, env)
    imc = Imc(agent=RandomAgent(), mc=vec_mc)
    state = imc.init(mc=mc_state, agent=AgentState(key=key))
    evaluator = McEval(imc=imc, episode_len=4)

    _, state = jax.jit(evaluator.evaluate)(state)
    _, state = jax.jit(imc.sample)(state)

    assert jnp.all(state.mc.env.token == 5)
    runtime = gymnasium.lookup_runtime(state.mc.env.runtime_id)
    assert runtime.token == 5
    env.close(state.mc.env)


# =============================================================================
# Scalar Path Tests (num_envs=None, plain Mc)
# =============================================================================


def test_scalar_path_single_env():
    """Default num_envs with plain Mc works under jit without vmap."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, env=env)
    key = jrd.PRNGKey(11)

    env_state = env.init(key)
    mc_state = mc.init(key, env_state)

    assert mc_state.last_obs.shape == (OBS_DIM,)

    action = jnp.array(1)
    transition, mc_state = jax.jit(mc.sample)(action, mc_state)

    assert transition.obs.shape == (OBS_DIM,)
    assert transition.rew.shape == ()
    assert transition.term.shape == ()


def test_scalar_path_multiple_steps():
    """Scalar path runs many steps with auto-reset."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, env=env)
    key = jrd.PRNGKey(12)

    env_state = env.init(key)
    mc_state = mc.init(key, env_state)

    for i in range(100):
        action = jrd.randint(jrd.fold_in(key, i), (), 0, 2)
        _, mc_state = mc.sample(action, mc_state)

    assert jnp.all(jnp.isfinite(mc_state.last_obs))
    assert 0 <= mc_state.eps_idx < 500


# =============================================================================
# Single-env Roll Tests
# =============================================================================


def test_roll_single_env():
    """Imc + single-env Mc + Roll collects trajectories."""
    env = gymnasium.make("CartPole-v1")
    mc = Mc(max_episode_len=500, env=env)
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

    assert transitions.mc.obs.shape == (seqlen, OBS_DIM)
    assert transitions.mc.nobs.shape == (seqlen, OBS_DIM)
    assert transitions.mc.rew.shape == (seqlen,)
    assert transitions.dec.act.shape == (seqlen,)


@pytest.mark.parametrize("name, obs_shape, act_shape, continuous", ENV_CASES)
def test_scalar_component_rollout_across_envs(
    name,
    obs_shape,
    act_shape,
    continuous,
):
    """Mc, Imc, and Roll compose under jit across scalar Gymnasium envs."""
    env = gymnasium.make(name)
    mc = Mc(max_episode_len=1000, env=env)
    imc = Imc(
        agent=ZeroAgent(
            obs_shape=obs_shape,
            act_shape=act_shape,
            continuous=continuous,
        ),
        mc=mc,
    )
    roll = Roll(imc=imc, seqlen=COMPONENT_ROLLOUT_LEN)

    env_key, mc_key, agent_key = jrd.split(jrd.PRNGKey(0), 3)
    mc_state = mc.init(mc_key, env.init(env_key))
    state = imc.init(mc=mc_state, agent=AgentState(key=agent_key))

    transitions, state = jax.jit(roll.sample)(state)
    env.close(state.mc.env)

    assert transitions.mc.obs.shape == (COMPONENT_ROLLOUT_LEN, *obs_shape)
    assert transitions.dec.act.shape == (COMPONENT_ROLLOUT_LEN, *act_shape)
    assert transitions.mc.rew.shape == (COMPONENT_ROLLOUT_LEN,)
    assert transitions.mc.term.shape == (COMPONENT_ROLLOUT_LEN,)
    assert transitions.mc.trun.shape == (COMPONENT_ROLLOUT_LEN,)
    assert transitions.mc.nobs.shape == (COMPONENT_ROLLOUT_LEN, *obs_shape)
    assert jnp.all(jnp.isfinite(transitions.mc.obs))
    assert jnp.all(jnp.isfinite(transitions.mc.rew))


@pytest.mark.parametrize("name, obs_shape, act_shape, continuous", ENV_CASES)
def test_vector_component_rollout_across_envs(
    name,
    obs_shape,
    act_shape,
    continuous,
):
    """VecMc, Imc, and Roll compose under jit across vector Gymnasium envs."""
    env = gymnasium.make(name, num_envs=COMPONENT_NUM_ENVS)
    mc = Mc(max_episode_len=1000, env=env)
    vec_mc = VecMc(mc=mc)
    imc = Imc(
        agent=ZeroAgent(
            obs_shape=obs_shape,
            act_shape=act_shape,
            continuous=continuous,
        ),
        mc=vec_mc,
    )
    roll = Roll(
        imc=imc,
        seqlen=COMPONENT_ROLLOUT_LEN,
        seq_axis=1,
    )

    env_key, mc_key, agent_key = jrd.split(jrd.PRNGKey(0), 3)
    mc_state = vec_mc.init(
        jrd.split(mc_key, COMPONENT_NUM_ENVS),
        env.init(env_key),
    )
    state = imc.init(mc=mc_state, agent=AgentState(key=agent_key))

    transitions, state = jax.jit(roll.sample)(state)
    env.close(state.mc.env)

    prefix = (COMPONENT_NUM_ENVS, COMPONENT_ROLLOUT_LEN)
    assert transitions.mc.obs.shape == (*prefix, *obs_shape)
    assert transitions.dec.act.shape == (*prefix, *act_shape)
    assert transitions.mc.rew.shape == prefix
    assert transitions.mc.term.shape == prefix
    assert transitions.mc.trun.shape == prefix
    assert transitions.mc.nobs.shape == (*prefix, *obs_shape)
    assert jnp.all(jnp.isfinite(transitions.mc.obs))
    assert jnp.all(jnp.isfinite(transitions.mc.rew))


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
        transitions, imc_state = roll.sample(imc_state)
        env.close(imc_state.mc.env)
        return transitions

    t1 = _collect(seed)
    t2 = _collect(seed)

    assert jnp.array_equal(t1.mc.obs, t2.mc.obs)
    assert jnp.array_equal(t1.dec.act, t2.dec.act)
    assert jnp.array_equal(t1.mc.rew, t2.mc.rew)
    assert jnp.array_equal(t1.mc.nobs, t2.mc.nobs)
    assert jnp.array_equal(t1.mc.term, t2.mc.term)
    assert jnp.array_equal(t1.mc.trun, t2.mc.trun)


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
        transitions, imc_state = roll.sample(imc_state)
        env.close(imc_state.mc.env)
        return transitions

    t1 = _collect(jrd.PRNGKey(0))
    t2 = _collect(jrd.PRNGKey(999))

    assert not jnp.array_equal(t1.mc.obs, t2.mc.obs)


# =============================================================================
# vmap Axis Guard Tests
# =============================================================================


def test_vmap_axis_larger_than_pool_raises():
    """A vmap wider than the pool is rejected."""
    env = _make_env(num_envs=4)
    env_state = env.init(jrd.PRNGKey(0))
    vec_mc = VecMc(mc=_make_mc(env))

    with pytest.raises(ValueError, match=r"axis_size \(8\) > num_envs \(4\)"):
        vec_mc.init(jrd.split(jrd.PRNGKey(0), 8), env_state)


def test_vmap_axis_smaller_than_pool_raises():
    """A vmap narrower than the pool is rejected for this backend.

    gymnasium's ``make_vec`` steps every env it owns, so it cannot serve a
    subset the way EnvPool can (see ``from_factory(allow_subset=...)``).
    Without this guard the mismatch surfaces as an opaque ``zip()`` error from
    inside ``SyncVectorEnv.step``.
    """
    env = _make_env(num_envs=8)
    env_state = env.init(jrd.PRNGKey(0))
    vec_mc = VecMc(mc=_make_mc(env))

    with pytest.raises(ValueError, match=r"axis_size \(4\) != num_envs \(8\)"):
        vec_mc.init(jrd.split(jrd.PRNGKey(0), 4), env_state)
