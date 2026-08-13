"""Tests for shuffled minibatches."""

import jax
import jax.numpy as jnp
import pytest

from jaxtor.util import Minibatches


def test_shuffle_preserves_aligned_pytree_rows():
    """Every row occurs once and aligned leaves use the same permutation."""
    batch = {
        "x": jnp.arange(12).reshape(6, 2),
        "y": jnp.arange(6) + 100,
    }

    shuffled = Minibatches(count=3).shuffle(jax.random.PRNGKey(0), batch)

    assert shuffled["x"].shape == (3, 2, 2)
    assert shuffled["y"].shape == (3, 2)
    assert jnp.array_equal(shuffled["x"][..., 0] // 2, shuffled["y"] - 100)
    assert jnp.array_equal(jnp.sort(shuffled["y"].reshape(-1)), batch["y"])


def test_shuffle_collapses_leading_sample_axes():
    """Multiple sample axes are collapsed before aligned shuffling."""
    batch = {
        "id": jnp.arange(6).reshape(2, 3),
        "x": jnp.arange(12).reshape(2, 3, 2),
    }

    shuffled = Minibatches(count=3, sample_ndim=2).shuffle(
        jax.random.PRNGKey(0),
        batch,
    )

    assert shuffled["id"].shape == (3, 2)
    assert shuffled["x"].shape == (3, 2, 2)
    assert jnp.array_equal(shuffled["x"][..., 0], 2 * shuffled["id"])


def test_shuffle_is_jittable():
    """Shuffling an array pytree works under JIT."""
    batch = (jnp.arange(8), jnp.arange(8) * 2)
    shuffled = jax.jit(Minibatches(count=2).shuffle)(jax.random.PRNGKey(1), batch)

    assert shuffled[0].shape == (2, 4)
    assert jnp.array_equal(shuffled[1], shuffled[0] * 2)


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_nonpositive_count(count: int):
    """The number of minibatches must be positive."""
    with pytest.raises(ValueError, match="count must be positive"):
        Minibatches(count=count)


@pytest.mark.parametrize("sample_ndim", [0, -1])
def test_rejects_nonpositive_sample_ndim(sample_ndim: int):
    """The number of leading sample axes must be positive."""
    with pytest.raises(ValueError, match="sample_ndim must be positive"):
        Minibatches(count=1, sample_ndim=sample_ndim)


def test_rejects_uneven_batch_size():
    """Every minibatch must have the same static size."""
    with pytest.raises(ValueError, match="batch size must be divisible"):
        Minibatches(count=2).shuffle(jax.random.PRNGKey(0), jnp.arange(5))


def test_rejects_scalar_leaves():
    """Every leaf needs the batch axis consumed by the component."""
    with pytest.raises(ValueError, match="fewer axes than sample_ndim"):
        Minibatches(count=1).shuffle(jax.random.PRNGKey(0), jnp.array(1.0))


def test_rejects_misaligned_sample_axes():
    """Every leaf must share all leading sample dimensions."""
    batch = {
        "x": jnp.ones((2, 3)),
        "y": jnp.ones((2, 4)),
    }
    with pytest.raises(ValueError, match="share their leading sample axes"):
        Minibatches(count=2, sample_ndim=2).shuffle(
            jax.random.PRNGKey(0),
            batch,
        )


def test_rejects_empty_batch():
    """The component does not produce zero-size optimizer steps."""
    with pytest.raises(ValueError, match="batch must not be empty"):
        Minibatches(count=1).shuffle(jax.random.PRNGKey(0), jnp.empty(0))
