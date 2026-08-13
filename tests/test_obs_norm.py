"""Tests for observation normalization as a feature transform."""

import jax
import jax.numpy as jnp

from jaxtor.util import ObsNorm, RunningStats


def test_obs_norm_updates_explicitly():
    """Applying reads statistics while updating changes them explicitly."""
    norm = ObsNorm(stats=RunningStats())
    state = norm.init((2,))
    observations = jnp.array([[1.0, 3.0], [3.0, 5.0]])

    before, unchanged = norm.apply(observations, state)
    updated = norm.update(observations, state)
    after, _ = norm.apply(observations, updated)

    assert jnp.allclose(before, observations)
    assert jnp.array_equal(unchanged.stats.mean, state.stats.mean)
    assert jnp.allclose(jnp.mean(after, axis=0), 0.0, atol=2e-4)


def test_disabled_obs_norm_is_a_static_noop():
    """The disabled strategy preserves observations and statistics."""
    norm = ObsNorm(stats=RunningStats(clip=1.0), enabled=False)
    state = norm.init((2,))
    observations = jnp.array([[20.0, -20.0]])

    output, applied = jax.jit(norm.apply)(observations, state)
    updated = jax.jit(norm.update)(observations, state)

    assert jnp.array_equal(output, observations)
    assert jnp.array_equal(applied.stats.mean, state.stats.mean)
    assert jnp.array_equal(updated.stats.mean, state.stats.mean)
    assert updated.stats.count == state.stats.count
