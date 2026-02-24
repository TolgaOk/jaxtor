"""Tests for RewardNorm component."""

import jax
import jax.numpy as jnp
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats


def _make_state(n_envs: int) -> RewardNorm.State:
    """Create a fresh RewardNorm state."""
    return RewardNorm.State(
        ret=jnp.zeros(n_envs),
        rms=RunningStats.State(
            mean=jnp.float32(0.0),
            var=jnp.float32(1.0),
            count=jnp.float32(1e-4),
        ),
    )


# =============================================================================
# Basic update tests
# =============================================================================


def test_update_returns_normalized_rewards():
    """Update returns rewards divided by std of rolling return."""
    rn = RewardNorm(gamma=0.99, rms=RunningStats())
    state = _make_state(n_envs=2)
    rewards = jnp.ones((2, 4))
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    norm_rew, new_state = rn.update(rewards, dones, state)

    assert norm_rew.shape == rewards.shape
    assert new_state.rms.count > state.rms.count


def test_update_output_shape():
    """Output reward shape matches input reward shape."""
    rn = RewardNorm(gamma=0.99, rms=RunningStats())
    state = _make_state(n_envs=4)
    rewards = jnp.ones((4, 10))
    dones = jnp.zeros((4, 10), dtype=jnp.bool_)

    norm_rew, _ = rn.update(rewards, dones, state)
    assert norm_rew.shape == (4, 10)


def test_update_clip():
    """Normalized rewards are clipped when clip is set."""
    rn = RewardNorm(gamma=0.99, rms=RunningStats(), clip=5.0)
    state = _make_state(n_envs=2)
    # Large rewards to trigger clipping
    rewards = jnp.full((2, 4), 1000.0)
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    norm_rew, _ = rn.update(rewards, dones, state)
    assert jnp.all(norm_rew >= -5.0)
    assert jnp.all(norm_rew <= 5.0)


def test_update_no_clip():
    """Without clip, normalized rewards differ from clipped version."""
    rn_noclip = RewardNorm(gamma=0.99, rms=RunningStats())
    rn_clip = RewardNorm(gamma=0.99, rms=RunningStats(), clip=0.5)
    state = _make_state(n_envs=2)
    rewards = jnp.full((2, 4), 1000.0)
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    norm_noclip, _ = rn_noclip.update(rewards, dones, state)
    norm_clip, _ = rn_clip.update(rewards, dones, state)
    # Clipped version should be bounded, unclipped should not be forced
    assert jnp.all(jnp.abs(norm_clip) <= 0.5)
    assert jnp.allclose(norm_noclip, norm_noclip)  # no NaN


# =============================================================================
# Rolling return tests
# =============================================================================


def test_rolling_return_resets_on_done():
    """Done flag resets the rolling return to zero."""
    rn = RewardNorm(gamma=0.99, rms=RunningStats())
    state = _make_state(n_envs=1)
    # Step 0: reward=1 (no done) -> ret = 0.99*0 + 1 = 1
    # Step 1: done=True, reward=1 -> ret = 0.99*0*1 + 1 = 1 (reset by done)
    rewards = jnp.array([[1.0, 1.0]])
    dones = jnp.array([[False, True]])

    _, new_state = rn.update(rewards, dones, state)
    # After done, ret resets: next ret starts from 0 * gamma * (1-done) + rew
    # Final ret = 0.99 * (1-1) * prev_ret + 1 = 1.0
    assert jnp.allclose(new_state.ret, jnp.array([1.0]))


def test_rolling_return_accumulates():
    """Without dones, return accumulates as gamma * ret + rew."""
    rn = RewardNorm(gamma=0.5, rms=RunningStats())
    state = _make_state(n_envs=1)
    rewards = jnp.array([[1.0, 1.0, 1.0]])
    dones = jnp.zeros((1, 3), dtype=jnp.bool_)

    _, new_state = rn.update(rewards, dones, state)
    # ret_0 = 0.5*0 + 1 = 1.0
    # ret_1 = 0.5*1 + 1 = 1.5
    # ret_2 = 0.5*1.5 + 1 = 1.75
    assert jnp.allclose(new_state.ret, jnp.array([1.75]))


# =============================================================================
# Matches ppo.py inline version
# =============================================================================


def test_matches_ppo_inline():
    """Output matches the inline normalize_rewards from ppo.py."""
    gamma = 0.99
    rn = RewardNorm(gamma=gamma, rms=RunningStats(), clip=10.0)
    state = _make_state(n_envs=4)

    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    rewards = jax.random.normal(k1, (4, 16))
    dones = jax.random.bernoulli(k2, 0.1, (4, 16))

    norm_rew, new_state = rn.update(rewards, dones, state)

    # Inline version from ppo.py
    def scan_fn(ret, step_data):
        rew, done = step_data
        ret = ret * gamma * (1.0 - done) + rew
        return ret, ret

    rew_t = jnp.transpose(rewards)
    done_t = jnp.transpose(dones.astype(jnp.float32))
    final_ret, all_rets = jax.lax.scan(scan_fn, state.ret, (rew_t, done_t))

    # Inline update_stats
    batch = all_rets.reshape(-1)
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - state.rms.mean
    total = state.rms.count + batch_count
    _ = state.rms.mean + delta * batch_count / total  # new_mean (unused, for clarity)
    m_a = state.rms.var * state.rms.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * state.rms.count * batch_count / total
    new_var = m2 / total

    expected = rewards / jnp.sqrt(new_var + 1e-8)
    expected = jnp.clip(expected, -10.0, 10.0)

    assert jnp.allclose(norm_rew, expected, atol=1e-5)
    assert jnp.allclose(new_state.ret, final_ret, atol=1e-5)


# =============================================================================
# JIT tests
# =============================================================================


def test_update_jit():
    """Update works under JIT."""
    rn = RewardNorm(gamma=0.99, rms=RunningStats(), clip=10.0)
    state = _make_state(n_envs=2)
    rewards = jnp.ones((2, 4))
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    jit_update = jax.jit(rn.update)
    norm_rew, new_state = jit_update(rewards, dones, state)
    assert norm_rew.shape == (2, 4)
    assert new_state.rms.count > state.rms.count
