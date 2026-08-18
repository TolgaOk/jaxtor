"""Tests for agent inference aligned with transition sequences."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass

from jaxtor.agent import (
    Categorical,
    QNextVInference,
    VPiNextVInference,
)


@dataclass
class ValuePolicy:
    """Joint value and policy result used by ``VPiNextVInference``."""

    v: jax.Array
    pi: Categorical


@dataclass
class State:
    """One scalar parameter for the replay test agents."""

    scale: jax.Array


@dataclass
class VPiAgent:
    """Agent providing value-only and joint value-policy endpoints."""

    def v(self, obs: jax.Array, state: State) -> tuple[jax.Array, State]:
        """Return scaled values without constructing policies."""
        return obs[..., 0] * state.scale, state

    def vpi(self, obs: jax.Array, state: State) -> tuple[ValuePolicy, State]:
        """Return scaled values and categorical policies."""
        value = obs[..., 0] * state.scale
        return ValuePolicy(
            v=value,
            pi=Categorical(logits=jnp.stack((value, -value), axis=-1)),
        ), state


@dataclass
class QVAgent:
    """Agent exposing independent action-value and value computations."""

    def q(
        self,
        obs: jax.Array,
        act: jax.Array,
        state: State,
    ) -> tuple[jax.Array, State]:
        """Evaluate one Q-value for each observation-action pair."""
        return obs[..., 0] * state.scale + act[..., 0], state

    def v(self, obs: jax.Array, state: State) -> tuple[jax.Array, State]:
        """Evaluate one value for each observation."""
        return obs[..., 0] * state.scale, state


@dataclass
class Sequence:
    """Minimal transition sequence consumed by inference components."""

    obs: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


@dataclass
class ActionSequence:
    """Transition sequence including actions consumed by Q-function inference."""

    obs: jax.Array
    act: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


def test_vpi_next_v_evaluates_every_stored_successor():
    """Successor values remain raw across every boundary combination."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        nobs=jnp.array([[2.0], [-999.0], [30.0], [-999.0]]),
        term=jnp.array([False, True, False, True]),
        trun=jnp.array([False, False, True, True]),
    )

    infer = jax.jit(component.apply)(
        seq,
        State(scale=jnp.array(1.0)),
    )

    assert jnp.array_equal(infer.v_tm1, jnp.array([1.0, 2.0, 3.0, 4.0]))
    assert jnp.array_equal(infer.v_t, jnp.array([2.0, -999.0, 30.0, -999.0]))
    assert infer.pi_tm1.evaluate(jnp.zeros(4, dtype=jnp.int32)).logp.shape == (4,)


def test_vpi_next_v_does_not_substitute_continuation_values():
    """Dense inference evaluates stored successors rather than shifted values."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        nobs=jnp.array([[-999.0], [-999.0], [-999.0], [5.0]]),
        term=jnp.zeros(4, dtype=jnp.bool_),
        trun=jnp.zeros(4, dtype=jnp.bool_),
    )

    infer = jax.jit(component.apply)(
        seq,
        State(scale=jnp.array(1.0)),
    )

    assert jnp.array_equal(infer.v_t, jnp.array([-999.0, -999.0, -999.0, 5.0]))


def test_q_next_v_evaluates_q_and_raw_successor_values():
    """Q excludes the first source while V evaluates every true successor."""
    component = QNextVInference(agent=QVAgent(), seq_axis=1)
    seq = ActionSequence(
        obs=jnp.array(
            [
                [[-999.0], [2.0], [3.0], [4.0]],
                [[-999.0], [6.0], [7.0], [8.0]],
            ]
        ),
        act=jnp.array(
            [
                [[10.0], [20.0], [30.0], [40.0]],
                [[50.0], [60.0], [70.0], [80.0]],
            ]
        ),
        nobs=jnp.array(
            [
                [[2.0], [-999.0], [30.0], [-999.0]],
                [[6.0], [60.0], [-999.0], [9.0]],
            ]
        ),
        term=jnp.array(
            [
                [False, True, False, True],
                [False, False, True, False],
            ]
        ),
        trun=jnp.array(
            [
                [False, False, True, True],
                [False, True, False, False],
            ]
        ),
    )

    infer = jax.jit(component.apply)(
        seq,
        State(scale=jnp.array(1.0)),
    )

    assert jnp.array_equal(
        infer.q_t,
        jnp.array([[22.0, 33.0, 44.0], [66.0, 77.0, 88.0]]),
    )
    assert jnp.array_equal(
        infer.v_t,
        jnp.array([[2.0, -999.0, 30.0, -999.0], [6.0, 60.0, -999.0, 9.0]]),
    )


def test_vpi_next_v_supports_a_nonleading_sequence_axis():
    """Value-policy inference preserves batch axes when time is axis one."""
    component = VPiNextVInference(agent=VPiAgent(), seq_axis=1)
    obs = jnp.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    seq = Sequence(
        obs=obs,
        nobs=jnp.array([[[2.0], [20.0], [4.0]], [[5.0], [50.0], [7.0]]]),
        term=jnp.zeros((2, 3), dtype=jnp.bool_),
        trun=jnp.array([[False, True, False], [False, True, False]]),
    )

    infer = jax.jit(component.apply)(seq, State(scale=jnp.array(1.0)))

    chex.assert_shape(infer.v_tm1, (2, 3))
    chex.assert_shape(infer.v_t, (2, 3))
    chex.assert_shape(infer.pi_tm1.logits, (2, 3, 2))
    assert jnp.array_equal(infer.v_t, jnp.array([[2.0, 20.0, 4.0], [5.0, 50.0, 7.0]]))


def test_vpi_next_v_supports_nested_vmap_jit_and_tree_structure():
    """Independent mapped axes compose around scalar sequence inference."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0]]),
        nobs=jnp.array([[2.0], [3.0]]),
        term=jnp.zeros(2, dtype=jnp.bool_),
        trun=jnp.zeros(2, dtype=jnp.bool_),
    )
    seq = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 3, *leaf.shape)),
        seq,
    )
    state = State(scale=jnp.ones((2, 3)))

    infer = jax.jit(jax.vmap(jax.vmap(component.apply)))(seq, state)

    chex.assert_shape(infer.v_tm1, (2, 3, 2))
    chex.assert_shape(infer.v_t, (2, 3, 2))
    chex.assert_shape(infer.pi_tm1.logits, (2, 3, 2, 2))
    assert jax.tree.structure(infer) == jax.tree.structure(
        jax.tree.map(lambda leaf: leaf, infer)
    )


def test_vpi_next_v_remains_differentiable_through_replayed_values():
    """Inference leaves current and stored-successor values differentiable."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0]]),
        nobs=jnp.array([[2.0], [3.0]]),
        term=jnp.zeros(2, dtype=jnp.bool_),
        trun=jnp.zeros(2, dtype=jnp.bool_),
    )

    def loss(scale: jax.Array) -> jax.Array:
        """Sum all replayed values."""
        infer = component.apply(seq, State(scale=scale))
        return jnp.sum(infer.v_tm1) + jnp.sum(infer.v_t)

    assert jax.grad(loss)(jnp.array(1.0)) == 8


@pytest.mark.parametrize("seq_axis", [-2, 1])
def test_vpi_next_v_rejects_an_axis_outside_scalar_sequence_samples(seq_axis):
    """The configured sequence axis must index a transition sample axis."""
    component = VPiNextVInference(agent=VPiAgent(), seq_axis=seq_axis)
    seq = Sequence(
        obs=jnp.ones((2, 1)),
        nobs=jnp.ones((2, 1)),
        term=jnp.zeros(2, dtype=jnp.bool_),
        trun=jnp.zeros(2, dtype=jnp.bool_),
    )

    with pytest.raises(ValueError, match="seq_axis is outside"):
        component.apply(seq, State(scale=jnp.array(1.0)))


def test_vpi_next_v_rejects_a_sequence_without_a_sample_axis():
    """Scalar boundary flags cannot describe a transition sequence."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.ones(1),
        nobs=jnp.ones(1),
        term=jnp.array(False),
        trun=jnp.array(False),
    )

    with pytest.raises(ValueError, match="must contain a sequence axis"):
        component.apply(seq, State(scale=jnp.array(1.0)))


def test_vpi_next_v_rejects_an_empty_sequence():
    """Dense inference requires at least one transition."""
    component = VPiNextVInference(agent=VPiAgent())
    seq = Sequence(
        obs=jnp.empty((0, 1)),
        nobs=jnp.empty((0, 1)),
        term=jnp.empty(0, dtype=jnp.bool_),
        trun=jnp.empty(0, dtype=jnp.bool_),
    )

    with pytest.raises(ValueError, match="sequence must not be empty"):
        component.apply(seq, State(scale=jnp.array(1.0)))
