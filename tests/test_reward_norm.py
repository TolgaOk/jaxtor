"""Tests for RewardNorm component."""

import jax
import jax.numpy as jnp
import pytest

from jaxtor.util import RewardNorm, RunningStats


def _make_state(n_envs: int) -> RewardNorm.State:
    """Create a fresh RewardNorm state."""
    return RewardNorm.State(
        ret=jnp.zeros(n_envs),
        stats=RunningStats.State(
            mean=jnp.float32(0.0),
            var=jnp.float32(1.0),
            count=jnp.float32(1e-4),
        ),
    )


# =============================================================================
# Basic update tests
# =============================================================================


def test_init_creates_per_environment_returns():
    """Initialization owns both return and running-statistics state."""
    state = RewardNorm(gamma=0.99, stats=RunningStats()).init(batch_shape=(3,))

    assert state.ret.shape == (3,)
    assert state.stats.mean.shape == ()
    assert state.stats.var.shape == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gamma": -0.1}, "gamma must be between zero and one"),
        ({"gamma": 1.1}, "gamma must be between zero and one"),
        ({"gamma": 0.9, "clip": -1.0}, "clip must be nonnegative"),
    ],
)
def test_invalid_static_configuration_is_rejected(kwargs, message):
    """Invalid discounting and clipping fail at component construction."""
    with pytest.raises(ValueError, match=message):
        RewardNorm(stats=RunningStats(), **kwargs)


def test_disabled_reward_norm_is_a_static_noop():
    """Disabled normalization preserves rewards and state under JIT."""
    norm = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1, enabled=False)
    state = norm.init(batch_shape=(2,))
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    dones = jnp.zeros_like(rewards, dtype=jnp.bool_)

    output, updated = jax.jit(norm.update)(rewards, dones, state)

    assert jnp.array_equal(output, rewards)
    assert jnp.array_equal(updated.ret, state.ret)
    assert updated.stats.count == state.stats.count


def test_update_returns_normalized_rewards():
    """Update returns rewards divided by std of rolling return."""
    rn = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1)
    state = _make_state(n_envs=2)
    rewards = jnp.ones((2, 4))
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    norm_rew, new_state = rn.update(rewards, dones, state)

    assert norm_rew.shape == rewards.shape
    assert new_state.stats.count > state.stats.count


def test_default_axis_supports_scalar_rollouts():
    """The default sequence-first convention accepts rewards shaped ``(T,)``."""
    norm = RewardNorm(gamma=0.99, stats=RunningStats())
    rewards = jnp.ones(4)

    output, state = norm.update(
        rewards,
        jnp.zeros_like(rewards, dtype=jnp.bool_),
        norm.init(),
    )

    assert output.shape == (4,)
    assert state.ret.shape == ()


def test_update_output_shape():
    """Output reward shape matches input reward shape."""
    rn = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1)
    state = _make_state(n_envs=4)
    rewards = jnp.ones((4, 10))
    dones = jnp.zeros((4, 10), dtype=jnp.bool_)

    norm_rew, _ = rn.update(rewards, dones, state)
    assert norm_rew.shape == (4, 10)


def test_update_rejects_misaligned_environment_state():
    """The return carry must have one entry for every non-sequence lane."""
    norm = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1)

    with pytest.raises(AssertionError):
        norm.update(
            jnp.ones((2, 4)),
            jnp.zeros((2, 4), dtype=jnp.bool_),
            norm.init(batch_shape=(3,)),
        )


def test_update_clip():
    """Normalized rewards are clipped when clip is set."""
    rn = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1, clip=5.0)
    state = _make_state(n_envs=2)
    # Large rewards to trigger clipping
    rewards = jnp.full((2, 4), 1000.0)
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    norm_rew, _ = rn.update(rewards, dones, state)
    assert jnp.all(norm_rew >= -5.0)
    assert jnp.all(norm_rew <= 5.0)


def test_update_no_clip():
    """Without clip, normalized rewards differ from clipped version."""
    rn_noclip = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1)
    rn_clip = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1, clip=0.5)
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
    """Rewards ``[1, 1]`` record return ``1.99``, then clear the carry."""
    rn = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1)
    state = _make_state(n_envs=1)
    rewards = jnp.array([[1.0, 1.0]])
    dones = jnp.array([[False, True]])

    _, new_state = rn.update(rewards, dones, state)
    assert jnp.allclose(new_state.ret, jnp.array([0.0]))
    assert jnp.allclose(new_state.stats.mean, jnp.array(1.495), atol=1e-3)


def test_rolling_return_accumulates():
    """Without dones, return accumulates as gamma * ret + rew."""
    rn = RewardNorm(gamma=0.5, stats=RunningStats(), seq_axis=1)
    state = _make_state(n_envs=1)
    rewards = jnp.array([[1.0, 1.0, 1.0]])
    dones = jnp.zeros((1, 3), dtype=jnp.bool_)

    _, new_state = rn.update(rewards, dones, state)
    # ret_0 = 0.5*0 + 1 = 1.0
    # ret_1 = 0.5*1 + 1 = 1.5
    # ret_2 = 0.5*1.5 + 1 = 1.75
    assert jnp.allclose(new_state.ret, jnp.array([1.75]))


# =============================================================================
# Reference implementation
# =============================================================================


def test_matches_reference_implementation():
    """Output matches a direct rolling-return implementation."""
    gamma = 0.99
    rn = RewardNorm(gamma=gamma, stats=RunningStats(), seq_axis=1, clip=10.0)
    state = _make_state(n_envs=4)

    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    rewards = jax.random.normal(k1, (4, 16))
    dones = jax.random.bernoulli(k2, 0.1, (4, 16))

    norm_rew, new_state = rn.update(rewards, dones, state)

    # Direct rolling-return calculation.
    def scan_fn(
        ret: jax.Array,
        step_data: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        rew, done = step_data
        ret = ret * gamma + rew
        return jnp.where(done, 0, ret), ret

    rew_t = jnp.transpose(rewards)
    done_t = jnp.transpose(dones.astype(jnp.bool_))
    final_ret, all_rets = jax.lax.scan(scan_fn, state.ret, (rew_t, done_t))

    # Direct parallel Welford merge.
    batch = all_rets.reshape(-1)
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - state.stats.mean
    total = state.stats.count + batch_count
    _ = state.stats.mean + delta * batch_count / total
    m_a = state.stats.var * state.stats.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * state.stats.count * batch_count / total
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
    rn = RewardNorm(gamma=0.99, stats=RunningStats(), seq_axis=1, clip=10.0)
    state = _make_state(n_envs=2)
    rewards = jnp.ones((2, 4))
    dones = jnp.zeros((2, 4), dtype=jnp.bool_)

    jit_update = jax.jit(rn.update)
    norm_rew, new_state = jit_update(rewards, dones, state)
    assert norm_rew.shape == (2, 4)
    assert new_state.stats.count > state.stats.count
