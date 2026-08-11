"""Tests for stateless action-distribution components."""

import jax
import jax.numpy as jnp

from jaxtor.dist import Categorical, DiagNormal


def test_diag_normal_sample_matches_evaluation():
    """A sampled action carries the same log-probability as evaluation."""
    dist = DiagNormal()
    params = dist.Params(
        loc=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
        log_scale=jnp.log(jnp.array([0.5, 2.0])),
    )

    sample = jax.jit(dist.sample)(jax.random.key(0), params)
    evaluation = dist.evaluate(params, sample.act)

    assert sample.act.shape == (2, 2)
    assert sample.logp.shape == (2,)
    assert jnp.allclose(sample.logp, evaluation.logp)
    assert evaluation.entropy.shape == (2,)


def test_diag_normal_mode_and_known_statistics():
    """A standard two-dimensional Normal has its analytic mode and entropy."""
    dist = DiagNormal()
    params = dist.Params(loc=jnp.array([1.0, -1.0]), log_scale=jnp.zeros(2))

    mode = dist.mode(params)
    evaluation = dist.evaluate(params, mode)

    assert jnp.array_equal(mode, params.loc)
    assert jnp.allclose(evaluation.logp, -jnp.log(2.0 * jnp.pi))
    assert jnp.allclose(evaluation.entropy, jnp.log(2.0 * jnp.pi) + 1.0)


def test_diag_normal_supports_vmap_and_parameter_pytrees():
    """Parameter dataclasses map normally over independent random keys."""
    dist = DiagNormal()
    params = dist.Params(
        loc=jnp.zeros((3, 2)),
        log_scale=jnp.zeros((3, 2)),
    )

    sample = jax.jit(jax.vmap(dist.sample))(
        jax.random.split(jax.random.key(0), 3),
        params,
    )

    assert sample.act.shape == (3, 2)
    assert sample.logp.shape == (3,)
    assert jax.tree.structure(params) == jax.tree.structure(
        jax.tree.map(lambda x: x, params)
    )


def test_categorical_sample_matches_evaluation():
    """A sampled category carries the same log-probability as evaluation."""
    dist = Categorical()
    params = dist.Params(logits=jnp.array([[0.0, 1.0, -1.0], [2.0, 0.0, 1.0]]))

    sample = jax.jit(dist.sample)(jax.random.key(1), params)
    evaluation = dist.evaluate(params, sample.act)

    assert sample.act.shape == (2,)
    assert sample.logp.shape == (2,)
    assert jnp.allclose(sample.logp, evaluation.logp)
    assert evaluation.entropy.shape == (2,)


def test_categorical_mode_and_known_statistics():
    """Mode and uniform-distribution statistics match analytic values."""
    dist = Categorical()
    params = dist.Params(logits=jnp.zeros((2, 4)))

    mode = dist.mode(params)
    evaluation = dist.evaluate(params, mode)

    assert jnp.array_equal(mode, jnp.zeros(2, dtype=jnp.int32))
    assert jnp.allclose(evaluation.logp, -jnp.log(4.0))
    assert jnp.allclose(evaluation.entropy, jnp.log(4.0))


def test_categorical_supports_vmap_and_parameter_pytrees():
    """Categorical parameters and outputs remain ordinary mapped pytrees."""
    dist = Categorical()
    params = dist.Params(logits=jnp.zeros((3, 4)))

    sample = jax.jit(jax.vmap(dist.sample))(
        jax.random.split(jax.random.key(2), 3),
        params,
    )

    assert sample.act.shape == (3,)
    assert sample.logp.shape == (3,)
    assert jax.tree.structure(sample) == jax.tree.structure(
        jax.tree.map(lambda x: x, sample)
    )
