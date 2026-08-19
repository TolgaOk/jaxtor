"""Tests for RunningStats component."""

import jax
import jax.numpy as jnp
import pytest

from jaxtor.util import RunningStats


# =============================================================================
# Update tests
# =============================================================================


def test_init_creates_unit_statistics():
    """Initialization follows the requested feature shape."""
    state = RunningStats().init((3,))

    assert state.mean.shape == (3,)
    assert jnp.array_equal(state.mean, jnp.zeros(3))
    assert jnp.array_equal(state.var, jnp.ones(3))
    assert state.count > 0


def test_update_single_batch_mean():
    """Update with one batch recovers batch mean."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(3), var=jnp.ones(3), count=jnp.float32(0.0)
    )
    batch = jnp.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    state = rs.update(batch, state)
    assert jnp.allclose(state.mean, jnp.array([2.0, 3.0, 4.0]))


def test_update_single_batch_var():
    """Update with one batch recovers batch variance."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(2), var=jnp.ones(2), count=jnp.float32(0.0)
    )
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    state = rs.update(batch, state)
    expected_var = jnp.var(batch, axis=0)
    assert jnp.allclose(state.var, expected_var)


def test_update_count_accumulates():
    """Count increases by batch size on each update."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(0.0)
    )
    state = rs.update(jnp.ones((10,)), state)
    assert state.count == 10.0
    state = rs.update(jnp.ones((5,)), state)
    assert state.count == 15.0


def test_update_two_batches_matches_combined():
    """Two sequential updates match single update on combined data."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(2), var=jnp.zeros(2), count=jnp.float32(0.0)
    )
    batch1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    batch2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    state_seq = rs.update(batch1, state)
    state_seq = rs.update(batch2, state_seq)

    combined = jnp.concatenate([batch1, batch2], axis=0)
    state_all = rs.update(combined, state)

    assert jnp.allclose(state_seq.mean, state_all.mean, atol=1e-5)
    assert jnp.allclose(state_seq.var, state_all.var, atol=1e-5)


def test_update_scalar_stats():
    """Update works with scalar (0-dim feature) statistics."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(1e-4)
    )
    batch = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    state = rs.update(batch, state)
    assert state.mean.shape == ()
    assert state.var.shape == ()


def test_update_rejects_empty_batches_before_corrupting_state():
    """An empty update fails instead of replacing statistics with NaNs."""
    stats = RunningStats()

    with pytest.raises(ValueError, match="batch must not be empty"):
        stats.update(jnp.empty((0, 2)), stats.init((2,)))


def test_update_rejects_a_mismatched_feature_shape():
    """Batch feature axes must match the state initialized for the component."""
    stats = RunningStats()

    with pytest.raises(ValueError, match="expected batch feature shape"):
        stats.update(jnp.ones((4, 3)), stats.init((2,)))


def test_negative_clip_is_rejected():
    """A negative symmetric clipping bound is not meaningful."""
    with pytest.raises(ValueError, match="clip must be nonnegative"):
        RunningStats(clip=-1.0)


def test_update_matches_parallel_welford_formula():
    """Update output matches a direct parallel Welford calculation."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(3), var=jnp.ones(3), count=jnp.float32(1e-4)
    )
    key = jax.random.PRNGKey(42)
    batch = jax.random.normal(key, (100, 3))

    # Parallel Welford merge.
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]
    delta = batch_mean - state.mean
    total = state.count + batch_count
    expected_mean = state.mean + delta * batch_count / total
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * state.count * batch_count / total
    expected_var = m2 / total

    new_state = rs.update(batch, state)
    assert jnp.allclose(new_state.mean, expected_mean)
    assert jnp.allclose(new_state.var, expected_var)


# =============================================================================
# Normalize tests
# =============================================================================


def test_normalize_zero_mean_unit_var():
    """Normalize with mean=0 var=1 returns input unchanged."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(3), var=jnp.ones(3), count=jnp.float32(100.0)
    )
    x = jnp.array([1.0, 2.0, 3.0])
    normed = rs.normalize(x, state)
    assert jnp.allclose(normed, x, atol=1e-4)


def test_normalize_with_clip():
    """Normalize clips output to [-clip, clip]."""
    rs = RunningStats(clip=2.0)
    state = RunningStats.State(
        mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(100.0)
    )
    x = jnp.array([-5.0, 0.0, 5.0])
    normed = rs.normalize(x, state)
    assert jnp.all(normed >= -2.0)
    assert jnp.all(normed <= 2.0)
    assert jnp.allclose(normed[1], 0.0)


def test_normalize_without_clip():
    """Normalize without clip does not clip large values."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.float32(0.0), var=jnp.float32(1.0), count=jnp.float32(100.0)
    )
    x = jnp.array([100.0])
    normed = rs.normalize(x, state)
    assert normed[0] > 10.0


def test_normalize_matches_reference_formula():
    """Normalization matches a direct calculation with clipping."""
    rs = RunningStats(clip=10.0)
    state = RunningStats.State(
        mean=jnp.array([1.0, 2.0]),
        var=jnp.array([0.5, 2.0]),
        count=jnp.float32(100.0),
    )
    x = jnp.array([3.0, 4.0])

    # Direct normalization.
    expected = (x - state.mean) / jnp.sqrt(state.var + 1e-8)
    expected = jnp.clip(expected, -10.0, 10.0)

    normed = rs.normalize(x, state)
    assert jnp.allclose(normed, expected)


# =============================================================================
# JIT tests
# =============================================================================


def test_update_jit():
    """Update works under JIT."""
    rs = RunningStats()
    state = RunningStats.State(
        mean=jnp.zeros(2), var=jnp.ones(2), count=jnp.float32(0.0)
    )
    batch = jnp.ones((5, 2))
    jit_update = jax.jit(rs.update)
    new_state = jit_update(batch, state)
    assert jnp.allclose(new_state.mean, jnp.ones(2))


def test_normalize_jit():
    """Normalize works under JIT."""
    rs = RunningStats(clip=10.0)
    state = RunningStats.State(
        mean=jnp.zeros(2), var=jnp.ones(2), count=jnp.float32(100.0)
    )
    x = jnp.array([1.0, 2.0])
    jit_normalize = jax.jit(rs.normalize)
    normed = jit_normalize(x, state)
    assert jnp.allclose(normed, x, atol=1e-4)
