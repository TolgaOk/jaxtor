"""Tests for component-state composition and trainable partitions."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.agent import (
    Categorical,
    CategoricalHead,
    DiagNormalHead,
    Model,
    Module,
    NormModel,
    Param,
    QHead,
    QsaHead,
    VHead,
    VPi,
    VQPi,
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
class ScalarDense:
    """Small dense callable producing one scalar instead of a singleton axis."""

    weight: Any
    bias: Any

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply one scalar affine transformation."""
        return x @ self.weight + self.bias


@dataclass
class State:
    """Nested test state containing trainable and frozen array leaves."""

    module: Module.State[jax.Array]
    statistic: jax.Array
    key: jax.Array


@dataclass
class Calls:
    """Number of times one stateful transform was applied."""

    count: jax.Array


@dataclass
class Body:
    """Identity body exposing whether acting evaluated shared features."""

    def apply(self, x: jax.Array, state: Calls) -> tuple[jax.Array, Calls]:
        """Return the input and advance the application count."""
        return x, Calls(count=state.count + 1)


@dataclass
class Value:
    """Value transform exposing whether acting evaluated an unused head."""

    def apply(self, x: jax.Array, state: Calls) -> tuple[jax.Array, Calls]:
        """Return one scalar per feature vector and advance the count."""
        return x[..., 0], Calls(count=state.count + 1)


@dataclass
class ActionValues:
    """Action-value transform exposing unnecessary acting computations."""

    def apply(self, x: jax.Array, state: Calls) -> tuple[jax.Array, Calls]:
        """Return two action values per feature vector and advance the count."""
        return jnp.stack((x[..., 0], -x[..., 0]), axis=-1), Calls(count=state.count + 1)


@dataclass
class Policy:
    """Categorical policy transform with an observable application count."""

    def apply(self, x: jax.Array, state: Calls) -> tuple[Categorical, Calls]:
        """Return a two-action policy and advance the count."""
        return Categorical(logits=jnp.stack((x[..., 0], -x[..., 0]), axis=-1)), Calls(
            count=state.count + 1
        )


class StructuredBody:
    """Map a dictionary observation to tuple-valued shared features."""

    def apply(
        self,
        obs: dict[str, jax.Array],
        state: None,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        """Produce two feature leaves without assuming one input array."""
        return (obs["left"] + obs["right"], obs["left"] - obs["right"]), state


class StructuredValue:
    """Produce values from tuple-valued features."""

    def apply(
        self,
        features: tuple[jax.Array, jax.Array],
        state: None,
    ) -> tuple[jax.Array, None]:
        """Use the first feature leaf as the value prediction."""
        return features[0], state


class StructuredPolicy:
    """Produce categorical policies from tuple-valued features."""

    def apply(
        self,
        features: tuple[jax.Array, jax.Array],
        state: None,
    ) -> tuple[Categorical, None]:
        """Use both feature leaves as categorical logits."""
        return Categorical(logits=jnp.stack(features, axis=-1)), state


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
    agent = VPi(body=body, v=value, pi=policy)
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
    assert jnp.array_equal(applied_state.key, state.key)
    assert not jnp.array_equal(acted_state.key, state.key)


def test_q_heads_preserve_action_semantics_over_leading_axes():
    """Q(s, .) and Q(s, a) keep their distinct output-axis contracts."""
    features = jnp.arange(30, dtype=jnp.float32).reshape((2, 5, 3))
    actions = jnp.ones((2, 5, 2))
    q_net = Module(static=Dense(weight=None, bias=None))
    q = QHead(n_actions=4, net=q_net)
    q_state = q.init(
        q_net.init(
            Dense(weight=jnp.ones((3, 4)), bias=jnp.arange(4.0)),
        )
    )
    qsa_net = Module(static=ScalarDense(weight=None, bias=None))
    qsa = QsaHead(value=qsa_net)
    qsa_state = qsa.init(
        qsa_net.init(
            ScalarDense(weight=jnp.arange(5.0), bias=jnp.array(1.0)),
        )
    )

    all_actions, _ = jax.jit(q.apply)(features, q_state)
    selected, _ = jax.jit(qsa.apply)(features, actions, qsa_state)

    chex.assert_shape(all_actions, (2, 5, 4))
    chex.assert_shape(selected, (2, 5))
    expected = jnp.concatenate((features, actions), axis=-1) @ jnp.arange(5.0) + 1
    assert jnp.allclose(selected, expected)


def test_diag_normal_head_and_model_thread_their_child_states():
    """Independent policy parameters and sequential model children both advance."""
    features = jnp.ones((3, 2))
    loc_net = Module(static=Dense(weight=None, bias=None))
    scale_net = Module(static=Dense(weight=None, bias=None))
    policy = DiagNormalHead(act_size=2, loc=loc_net, log_scale=scale_net)
    policy_state = policy.init(
        loc=loc_net.init(Dense(weight=jnp.eye(2), bias=jnp.zeros(2))),
        log_scale=scale_net.init(
            Dense(weight=jnp.zeros((2, 2)), bias=jnp.full(2, -0.5))
        ),
    )
    zero = Calls(count=jnp.array(0, dtype=jnp.int32))
    model = Model(body=Body(), head=ActionValues())

    pi, _ = jax.jit(policy.apply)(features, policy_state)
    output, state = jax.jit(model.apply)(features, model.init(zero, zero))

    chex.assert_shape(pi.loc, (3, 2))
    chex.assert_shape(pi.log_scale, (3, 2))
    chex.assert_shape(output, (3, 2))
    assert state.body.count == 1
    assert state.head.count == 1


def test_vpi_act_evaluates_only_action_dependencies():
    """Acting advances the shared body and policy, but not the value head."""
    zero = Calls(count=jnp.array(0, dtype=jnp.int32))
    agent = VPi(body=Body(), v=Value(), pi=Policy(), deterministic=True)
    state = agent.init(jax.random.key(0), body=zero, v=zero, pi=zero)

    _, applied = jax.jit(agent.apply)(jnp.ones((3, 2)), state)
    act, acted = jax.jit(agent.act)(jnp.ones((3, 2)), state)

    assert jnp.array_equal(act, jnp.zeros(3, dtype=jnp.int32))
    assert applied.body.count == 1
    assert applied.v.count == 1
    assert applied.pi.count == 1
    assert acted.body.count == 1
    assert acted.v.count == 0
    assert acted.pi.count == 1
    assert jnp.array_equal(acted.key, state.key)


def test_vpi_forwards_structured_observations_and_features():
    """Agent composition preserves pytree inputs and intermediate features."""
    agent = VPi(
        body=StructuredBody(),
        v=StructuredValue(),
        pi=StructuredPolicy(),
        deterministic=True,
    )
    state = agent.init(jax.random.key(0), body=None, v=None, pi=None)
    obs = {
        "left": jnp.array([2.0, 1.0]),
        "right": jnp.array([1.0, 3.0]),
    }

    pred, _ = jax.jit(agent.apply)(obs, state)
    act, _ = jax.jit(agent.act)(obs, state)

    assert jnp.array_equal(pred.v, jnp.array([3.0, 4.0]))
    assert jnp.array_equal(act, jnp.array([0, 0]))


def test_vqpi_act_skips_both_unused_value_heads():
    """Acting through VQPi does not evaluate either V or Q predictions."""
    zero = Calls(count=jnp.array(0, dtype=jnp.int32))
    agent = VQPi(
        body=Body(),
        v=Value(),
        q=ActionValues(),
        pi=Policy(),
        deterministic=True,
    )
    state = agent.init(
        jax.random.key(0),
        body=zero,
        v=zero,
        q=zero,
        pi=zero,
    )

    pred, applied = jax.jit(agent.apply)(jnp.ones((3, 2)), state)
    _, acted = jax.jit(agent.act)(jnp.ones((3, 2)), state)

    chex.assert_shape(pred.q, (3, 2))
    assert applied.v.count == 1
    assert applied.q.count == 1
    assert acted.body.count == 1
    assert acted.v.count == 0
    assert acted.q.count == 0
    assert acted.pi.count == 1
