"""Tests for completed-episode trajectory statistics."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass

from jaxtor.sampler import EpisodeStats


@dataclass
class Trajectory:
    """Minimal trajectory consumed by ``EpisodeStats``."""

    rew: jax.Array
    term: jax.Array
    trun: jax.Array


def test_scalar_partial_episode_survives_drain():
    """Draining clears completed sums while preserving an unfinished episode."""
    stats = EpisodeStats()
    state = stats.init()
    state = stats.update(
        Trajectory(
            rew=jnp.array([1.0, 2.0]),
            term=jnp.array([False, False]),
            trun=jnp.array([False, False]),
        ),
        state,
    )

    empty, state = stats.drain(state)

    assert jnp.isnan(empty.avg_eps_rew)
    assert empty.n_episodes == 0
    assert state.eps_rew == 3
    assert state.eps_len == 2

    state = stats.update(
        Trajectory(
            rew=jnp.array([3.0, 4.0]),
            term=jnp.array([True, False]),
            trun=jnp.array([False, False]),
        ),
        state,
    )
    metrics, state = stats.drain(state)

    assert metrics.avg_eps_rew == 6
    assert metrics.avg_eps_len == 3
    assert metrics.n_episodes == 1
    assert state.eps_rew == 4
    assert state.eps_len == 1
    assert state.sum_eps_rew == 0
    assert state.sum_eps_len == 0
    assert state.n_episodes == 0


def test_vector_trajectories_are_weighted_by_episode():
    """Vector lanes contribute completed episodes rather than lane averages."""
    stats = EpisodeStats(seq_axis=1)
    state = stats.init((2,))
    trajectory = Trajectory(
        rew=jnp.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
        term=jnp.array(
            [
                [False, True, False, False],
                [False, False, True, False],
            ]
        ),
        trun=jnp.array(
            [
                [False, False, False, True],
                [False, False, False, False],
            ]
        ),
    )

    state = stats.update(trajectory, state)
    metrics, state = stats.drain(state)

    assert jnp.allclose(metrics.avg_eps_rew, 10 / 3)
    assert jnp.allclose(metrics.avg_eps_len, 7 / 3)
    assert metrics.n_episodes == 3
    assert jnp.array_equal(state.eps_rew, jnp.array([0.0, 2.0]))
    assert jnp.array_equal(state.eps_len, jnp.array([0, 1]))


def test_update_and_drain_are_jittable_pytrees():
    """Statistics state and metrics remain ordinary pytrees under JIT."""
    stats = EpisodeStats()
    state = stats.init()
    trajectory = Trajectory(
        rew=jnp.array([2.0]),
        term=jnp.array([True]),
        trun=jnp.array([False]),
    )

    metrics, state = jax.jit(
        lambda trajectory, state: stats.drain(stats.update(trajectory, state))
    )(trajectory, state)

    assert metrics.avg_eps_rew == 2
    assert metrics.avg_eps_len == 1
    assert jax.tree.structure(state) == jax.tree.structure(stats.init())


def test_nested_vmap_preserves_scalar_statistics_structure():
    """A scalar statistics component supports multiple mapped batch axes."""
    stats = EpisodeStats()
    state = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 3, *leaf.shape)),
        stats.init(),
    )
    trajectory = Trajectory(
        rew=jnp.ones((2, 3, 2)),
        term=jnp.broadcast_to(jnp.array([False, True]), (2, 3, 2)),
        trun=jnp.zeros((2, 3, 2), dtype=jnp.bool_),
    )

    update = jax.jit(jax.vmap(jax.vmap(stats.update)))
    state = update(trajectory, state)

    chex.assert_shape(state.n_episodes, (2, 3))
    assert jnp.all(state.n_episodes == 1)
    assert jnp.all(state.eps_rew == 0)
    assert jnp.all(state.eps_len == 0)


def test_update_checks_trajectory_and_batch_shapes():
    """Trajectory fields and initialized environment lanes must align."""
    stats = EpisodeStats()
    state = stats.init()

    with pytest.raises(AssertionError):
        stats.update(
            Trajectory(
                rew=jnp.ones(2),
                term=jnp.zeros(3, dtype=jnp.bool_),
                trun=jnp.zeros(2, dtype=jnp.bool_),
            ),
            state,
        )

    vector_stats = EpisodeStats(seq_axis=1)
    with pytest.raises(AssertionError):
        vector_stats.update(
            Trajectory(
                rew=jnp.ones((3, 2)),
                term=jnp.zeros((3, 2), dtype=jnp.bool_),
                trun=jnp.zeros((3, 2), dtype=jnp.bool_),
            ),
            vector_stats.init((2,)),
        )
