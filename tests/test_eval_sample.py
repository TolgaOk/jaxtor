"""Tests for sampling-based evaluator."""

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from chex import dataclass
from jaxtor.env import tabular
from jaxtor.eval.sample import Eval as SampleEval
from jaxtor.sampler.mc import Mc
from jaxtor.sampler.imc import Imc


# =============================================================================
# Fake agents
# =============================================================================


class GoRightAgent:
    """Agent that always takes action 1 (right)."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        return jnp.array(1), state


class GoLeftAgent:
    """Agent that always takes action 3 (left) — hits walls in corridors."""

    @dataclass
    class State:
        key: chex.PRNGKey

    def act(self, obs, state):
        return jnp.array(3), state


# =============================================================================
# Helpers
# =============================================================================


def _make_eval_imc_state(key, mc, agent_state, env_state):
    """Build an Imc.State for evaluation."""
    mc_state = mc.init(key, env_state)
    return Imc.State(mc=mc_state, agent=agent_state)


# =============================================================================
# Sample Eval Tests
# =============================================================================


def test_sample_deterministic_one_step_goal():
    """Agent one step from goal completes episodes with length 1 and reward 1."""
    key = jax.random.PRNGKey(11)

    # 2-state gridworld: P@ — action 1 reaches goal immediately
    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=10, queue_size=20, env=env)
    agent = GoRightAgent()
    imc = Imc(agent=agent, mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=50)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert jnp.allclose(metrics.avg_eps_len, 1.0, atol=0.1)
    assert jnp.allclose(metrics.avg_eps_rew, 1.0, atol=0.1)
    assert metrics.n_episodes >= 5


def test_sample_truncation_rate_dead_end():
    """Agent stuck hitting walls truncates every episode."""
    key = jax.random.PRNGKey(13)

    # 3-state corridor: P _ @ — GoLeftAgent always goes left, stays at P
    config = tabular.gridworld.Config(
        board=["#####", "#P @#", "#####"],
        p_slip=0.0,
        max_episode_len=5,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=5, queue_size=20, env=env)
    imc = Imc(agent=GoLeftAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=20)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoLeftAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert jnp.allclose(metrics.trun_rate, 1.0, atol=0.01)


def test_sample_more_episodes_gives_more_data():
    """More n_episodes collects more completed episodes."""
    key = jax.random.PRNGKey(14)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=10, queue_size=20, env=env)
    agent = GoRightAgent()
    imc = Imc(agent=agent, mc=mc)

    eval_short = SampleEval(imc=imc, episode_len=30)
    eval_long = SampleEval(imc=imc, episode_len=100)

    env_key, mc_key1, mc_key2 = jrd.split(key, 3)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)

    imc_state_1 = _make_eval_imc_state(mc_key1, mc, agent_state, env_state)
    metrics_short = eval_short.metric(imc_state_1)

    imc_state_2 = _make_eval_imc_state(mc_key2, mc, agent_state, env_state)
    metrics_long = eval_long.metric(imc_state_2)

    assert metrics_long.n_episodes >= metrics_short.n_episodes


def test_sample_jit_compilation():
    """Verify metric() can be JIT compiled."""
    key = jax.random.PRNGKey(15)

    config = tabular.garnet.Config(
        state_size=5, action_size=2, max_episode_len=10
    )
    env = tabular.garnet.make(config)

    mc = Mc(max_episode_len=10, queue_size=5, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=20)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    jit_metric = jax.jit(evaluator.metric)
    metrics = jit_metric(imc_state)

    assert metrics.avg_eps_rew.shape == ()


def test_sample_key_determinism():
    """Same imc_state produces identical metrics."""
    key = jax.random.PRNGKey(16)

    config = tabular.garnet.Config(
        state_size=5, action_size=2, max_episode_len=10
    )
    env = tabular.garnet.make(config)

    mc = Mc(max_episode_len=10, queue_size=5, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=20)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics_a = evaluator.metric(imc_state)
    metrics_b = evaluator.metric(imc_state)

    assert jnp.array_equal(metrics_a.avg_eps_rew, metrics_b.avg_eps_rew)
    assert jnp.array_equal(metrics_a.avg_eps_len, metrics_b.avg_eps_len)


# =============================================================================
# Level 1: Edge Cases
# =============================================================================


def test_sample_truncation_rate_zero_when_all_terminate():
    """Agent that always reaches goal has trun_rate=0."""
    key = jax.random.PRNGKey(30)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=10, queue_size=20, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=50)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert jnp.allclose(metrics.trun_rate, 0.0, atol=0.01)


def test_sample_dead_end_episode_length_equals_max():
    """Truncated episodes have length equal to max_episode_len."""
    key = jax.random.PRNGKey(31)

    max_len = 5
    config = tabular.gridworld.Config(
        board=["#####", "#P @#", "#####"],
        p_slip=0.0,
        max_episode_len=max_len,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=max_len, queue_size=20, env=env)
    imc = Imc(agent=GoLeftAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=max_len * 3)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoLeftAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert jnp.allclose(metrics.avg_eps_len, max_len, atol=0.1)


def test_sample_std_zero_for_identical_episodes():
    """Deterministic 1-step episodes all have same reward, so std=0."""
    key = jax.random.PRNGKey(32)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=10, queue_size=20, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=50)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert jnp.allclose(metrics.std_eps_rew, 0.0, atol=1e-5)
    assert jnp.allclose(metrics.min_eps_rew, metrics.max_eps_rew, atol=1e-5)


def test_sample_queue_overflow_keeps_recent():
    """When queue_size < completed episodes, metrics reflect recent episodes only."""
    key = jax.random.PRNGKey(33)

    # 1-step episodes, queue_size=3, n_episodes=10 — queue fills and overwrites
    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)

    mc = Mc(max_episode_len=10, queue_size=3, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=100)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    # Queue is full (3 entries), no NaN — all episodes have reward 1
    assert metrics.n_episodes == 3
    assert jnp.allclose(metrics.avg_eps_rew, 1.0, atol=0.1)


# =============================================================================
# Level 2: Intermediate Dynamics
# =============================================================================


def test_sample_vmap_init_produces_batched_queues():
    """vmap over sampler.init produces (n_envs, queue_size) shaped queues."""
    key = jax.random.PRNGKey(40)

    config = tabular.garnet.Config(
        state_size=5, action_size=2, max_episode_len=10
    )
    env = tabular.garnet.make(config)
    mc = Mc(max_episode_len=10, queue_size=7, env=env)

    n_envs = 4
    init_key, env_key = jrd.split(key)
    env_state = env.init(env_key)
    env_keys = jrd.split(init_key, n_envs)
    states = jax.vmap(mc.init, in_axes=(0, None))(env_keys, env_state)

    assert states.eps_rew_queue.shape == (n_envs, 7)
    assert states.eps_len_queue.shape == (n_envs, 7)
    assert states.last_obs.shape == (n_envs,)
    assert jnp.all(jnp.isnan(states.eps_rew_queue))


def test_sample_queues_populated_after_rollout():
    """After rollout with completed episodes, queues contain non-NaN entries."""
    key = jax.random.PRNGKey(41)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)
    mc = Mc(max_episode_len=10, queue_size=10, env=env)
    agent = GoRightAgent()

    # Manually run init + a few steps to inspect queue
    init_key, env_key, agent_key = jrd.split(key, 3)
    env_state = env.init(env_key)
    mc_state = mc.init(init_key, env_state)
    agent_state = GoRightAgent.State(key=agent_key)

    # Before any steps — all NaN
    assert jnp.all(jnp.isnan(mc_state.eps_rew_queue))

    # Take one step (goal reached — episode done — queue updated)
    act, agent_state = agent.act(mc_state.last_obs, agent_state)
    _, mc_state = mc.sample(act, mc_state)

    # First entry should now be populated
    assert not jnp.isnan(mc_state.eps_rew_queue[0])
    assert jnp.all(jnp.isnan(mc_state.eps_rew_queue[1:]))


def test_sample_done_count_matches_expected():
    """In a 1-step deterministic env, done_count equals rollout_len."""
    key = jax.random.PRNGKey(42)

    config = tabular.gridworld.Config(
        board=["####", "#P@#", "####"],
        p_slip=0.0,
        max_episode_len=10,
    )
    env = tabular.gridworld.make(config)
    mc = Mc(max_episode_len=10, queue_size=20, env=env)
    agent = GoRightAgent()
    imc = Imc(agent=agent, mc=mc)

    evaluator = SampleEval(imc=imc, episode_len=50)

    # The rollout runs for episode_len = 50 steps
    # Each episode is 1 step, so done_count should be 50
    init_key, env_key, agent_key = jrd.split(key, 3)
    env_state = env.init(env_key)
    mc_state = mc.init(init_key, env_state)
    agent_state = GoRightAgent.State(key=agent_key)
    imc_state = Imc.State(mc=mc_state, agent=agent_state)

    _, done_count, trun_count = evaluator._rollout(imc_state)

    assert done_count == 50.0
    assert trun_count == 0.0


def test_sample_metrics_all_scalar():
    """Every field in Metrics is a scalar array."""
    key = jax.random.PRNGKey(43)

    config = tabular.garnet.Config(
        state_size=5, action_size=2, max_episode_len=10
    )
    env = tabular.garnet.make(config)

    mc = Mc(max_episode_len=10, queue_size=5, env=env)
    imc = Imc(agent=GoRightAgent(), mc=mc)
    evaluator = SampleEval(imc=imc, episode_len=20)

    env_key, mc_key = jrd.split(key)
    env_state = env.init(env_key)
    agent_state = GoRightAgent.State(key=key)
    imc_state = _make_eval_imc_state(mc_key, mc, agent_state, env_state)

    metrics = evaluator.metric(imc_state)

    assert metrics.avg_eps_rew.shape == ()
    assert metrics.avg_eps_len.shape == ()
    assert metrics.std_eps_rew.shape == ()
    assert metrics.min_eps_rew.shape == ()
    assert metrics.max_eps_rew.shape == ()
    assert metrics.n_episodes.shape == ()
    assert metrics.trun_rate.shape == ()
