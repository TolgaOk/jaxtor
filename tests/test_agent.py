"""Tests for component-state composition and trainable partitions."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.agent import (
    CategoricalHead,
    Draw,
    Module,
    NormModel,
    Param,
    VHead,
    VPi,
    combine,
    partition,
)
from jaxtor.util import ObsNorm, RunningStats


@dataclass
class Scale:
    """Small callable pytree used to exercise the module adapter."""

    value: Any

    def __call__(self, x: jax.Array) -> jax.Array:
        """Scale the input by the stored parameter."""
        return x * self.value


@dataclass
class Dense:
    """Small dense callable with independently partitioned parameters."""

    weight: Any
    bias: Any

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply one affine transformation."""
        return x @ self.weight + self.bias


@dataclass
class State:
    """Nested test state containing trainable and frozen array leaves."""

    module: Module.State[jax.Array]
    statistic: jax.Array
    key: jax.Array


def test_partition_selects_only_marked_parameters():
    """Partitioning preserves names while excluding statistics and keys."""
    module = Module(static=Scale(value=None), in_ndim=0)
    state = State(
        module=module.init(Scale(value=jnp.array(2.0))),
        statistic=jnp.array(3.0),
        key=jax.random.key(0),
    )

    split = partition(state)
    restored = combine(split.params, split.frozen)

    assert isinstance(split.params.module.params, Param)
    assert split.params.statistic is None
    assert split.params.key is None
    assert split.frozen.module.params is None
    assert split.frozen.statistic == state.statistic
    assert jax.tree.structure(restored) == jax.tree.structure(state)


def test_partition_supports_jit_and_grad():
    """Gradients traverse marked values without touching frozen state."""
    module = Module(static=Scale(value=None), in_ndim=0)
    state = State(
        module=module.init(Scale(value=jnp.array(2.0))),
        statistic=jnp.array(3.0),
        key=jax.random.key(0),
    )
    split = partition(state)

    def loss(params: Any) -> jax.Array:
        restored = combine(params, split.frozen)
        output, _ = module.apply(jnp.array(4.0), restored.module)
        return output + jax.lax.stop_gradient(restored.statistic)

    value, grads = jax.jit(jax.value_and_grad(loss))(split.params)

    assert value == 11.0
    assert grads.module.params.value.value == 4.0
    assert grads.statistic is None
    assert grads.key is None


def test_module_maps_arbitrary_leading_axes():
    """The adapter preserves every axis preceding one input vector."""
    module = Module(static=Dense(weight=None, bias=None))
    state = module.init(
        Dense(
            weight=jnp.arange(12, dtype=jnp.float32).reshape((3, 4)),
            bias=jnp.arange(4, dtype=jnp.float32),
        )
    )
    x = jnp.ones((2, 5, 3))

    output, next_state = jax.jit(module.apply)(x, state)

    assert output.shape == (2, 5, 4)
    assert jax.tree.structure(next_state) == jax.tree.structure(state)


def test_norm_model_owns_explicit_normalization_updates():
    """Normalization updates avoid exposing the composed state layout."""
    norm = ObsNorm(stats=RunningStats())
    model = Module(static=Scale(value=None), in_ndim=0)
    component = NormModel(norm=norm, model=model)
    state = component.init(
        norm=norm.init((2,)),
        model=model.init(Scale(value=jnp.array(2.0))),
    )
    observations = jnp.array([[1.0, 3.0], [3.0, 5.0]])

    before, applied = jax.jit(component.apply)(observations, state)
    updated = jax.jit(component.update)(observations, state)
    after, _ = jax.jit(component.apply)(observations, updated)

    assert jnp.allclose(before, observations * 2.0)
    assert jnp.array_equal(applied.norm.stats.mean, state.norm.stats.mean)
    chex.assert_trees_all_equal(updated.model, state.model)
    assert jnp.allclose(jnp.mean(after, axis=0), 0.0, atol=4e-4)


def test_vpi_composes_modules_and_preserves_leading_axes():
    """A composed value-policy agent needs only apply and act in PPO."""
    body = Module(static=Dense(weight=None, bias=None))
    value_net = Module(static=Dense(weight=None, bias=None))
    logits_net = Module(static=Dense(weight=None, bias=None))
    value = VHead(net=value_net)
    policy = CategoricalHead(n_actions=4, logits=logits_net)
    agent = VPi(body=body, v=value, pi=policy, select=Draw())
    state = agent.init(
        jax.random.key(0),
        body=body.init(
            Dense(weight=jnp.ones((3, 2)), bias=jnp.zeros(2)),
        ),
        v=value.init(
            value_net.init(
                Dense(weight=jnp.ones((2, 1)), bias=jnp.zeros(1)),
            )
        ),
        pi=policy.init(
            logits_net.init(
                Dense(weight=jnp.ones((2, 4)), bias=jnp.zeros(4)),
            )
        ),
    )
    obs = jnp.ones((2, 5, 3))

    pred, applied_state = jax.jit(agent.apply)(obs, state)
    act, acted_state = jax.jit(agent.act)(obs, state)

    assert pred.v.shape == (2, 5)
    assert pred.pi.logits.shape == (2, 5, 4)
    assert act.shape == (2, 5)
    assert jnp.array_equal(applied_state.select, state.select)
    assert not jnp.array_equal(acted_state.select, state.select)
