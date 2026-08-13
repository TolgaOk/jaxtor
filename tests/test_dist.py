"""Tests for pure action-distribution values."""

import jax
import jax.numpy as jnp

from jaxtor.agent import Categorical, DiagNormal


def test_diag_normal_sample_matches_evaluation():
    """A sampled action carries the same log-probability as evaluation."""
    dist = DiagNormal(
        loc=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
        log_scale=jnp.log(jnp.array([0.5, 2.0])),
    )

    sample = jax.jit(lambda d, key: d.sample(key))(dist, jax.random.key(0))
    evaluation = dist.evaluate(sample.act)

    assert sample.act.shape == (2, 2)
    assert sample.logp.shape == (2,)
    assert jnp.allclose(sample.logp, evaluation.logp)
    assert evaluation.entropy.shape == (2,)


def test_diag_normal_mode_and_known_statistics():
    """A standard two-dimensional Normal has its analytic mode and entropy."""
    dist = DiagNormal(loc=jnp.array([1.0, -1.0]), log_scale=jnp.zeros(2))

    mode = dist.mode()
    evaluation = dist.evaluate(mode)

    assert jnp.array_equal(mode, dist.loc)
    assert jnp.allclose(evaluation.logp, -jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(evaluation.entropy, jnp.log(2.0 * jnp.pi) + 1.0)


def test_diag_normal_supports_vmap_and_distribution_pytrees():
    """Distribution values map normally over independent random keys."""
    dist = DiagNormal(
        loc=jnp.zeros((3, 2)),
        log_scale=jnp.zeros((3, 2)),
    )

    sample = jax.jit(jax.vmap(lambda d, key: d.sample(key)))(
        dist,
        jax.random.split(jax.random.key(0), 3),
    )

    assert sample.act.shape == (3, 2)
    assert sample.logp.shape == (3,)
    assert jax.tree.structure(dist) == jax.tree.structure(
        jax.tree.map(lambda x: x, dist)
    )


def test_categorical_sample_matches_evaluation():
    """A sampled category carries the same log-probability as evaluation."""
    dist = Categorical(logits=jnp.array([[0.0, 1.0, -1.0], [2.0, 0.0, 1.0]]))

    sample = jax.jit(lambda d, key: d.sample(key))(dist, jax.random.key(1))
    evaluation = dist.evaluate(sample.act)

    assert sample.act.shape == (2,)
    assert sample.logp.shape == (2,)
    assert jnp.allclose(sample.logp, evaluation.logp)
    assert evaluation.entropy.shape == (2,)


def test_categorical_mode_and_known_statistics():
    """Mode and uniform-distribution statistics match analytic values."""
    dist = Categorical(logits=jnp.zeros((2, 4)))

    mode = dist.mode()
    evaluation = dist.evaluate(mode)

    assert jnp.array_equal(mode, jnp.zeros(2, dtype=jnp.int32))
    assert jnp.allclose(evaluation.logp, -jnp.log(4.0))
    assert jnp.allclose(evaluation.entropy, jnp.log(4.0))


def test_categorical_masked_actions_have_finite_entropy():
    """Zero-probability actions keep entropy and its gradient finite."""
    logits = jnp.array([0.0, -jnp.inf])

    def entropy(logits: jax.Array) -> jax.Array:
        """Evaluate categorical entropy for gradient inspection."""
        return Categorical(logits=logits).evaluate(jnp.array(0)).entropy

    evaluation = Categorical(logits=logits).evaluate(jnp.array(0))
    gradient = jax.grad(entropy)(logits)

    assert evaluation.logp == 0
    assert evaluation.entropy == 0
    assert jnp.isfinite(evaluation.entropy)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.array_equal(gradient, jnp.zeros(2))


def test_categorical_supports_vmap_and_distribution_pytrees():
    """Categorical distributions and outputs remain ordinary mapped pytrees."""
    dist = Categorical(logits=jnp.zeros((3, 4)))

    sample = jax.jit(jax.vmap(lambda d, key: d.sample(key)))(
        dist,
        jax.random.split(jax.random.key(2), 3),
    )

    assert sample.act.shape == (3,)
    assert sample.logp.shape == (3,)
    assert jax.tree.structure(sample) == jax.tree.structure(
        jax.tree.map(lambda x: x, sample)
    )
