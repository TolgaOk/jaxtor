"""Tests for sequence-replayed temporal-difference estimates."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass
from jax.experimental import checkify

from jaxtor.dist import Categorical
from jaxtor.estimate import TDEst


@dataclass
class Pred:
    """Value-policy prediction used to verify preserved agent output."""

    v: jax.Array
    pi: Categorical


@dataclass
class State:
    """One scalar parameter for the replay test agent."""

    scale: jax.Array


@dataclass
class Agent:
    """Pointwise value-policy agent rejecting invalid observations."""

    def apply(self, obs: jax.Array, state: State) -> tuple[Pred, State]:
        """Predict values and policies without changing state."""
        checkify.check(jnp.all(obs >= 0), "invalid successor was evaluated")
        value = obs[..., 0] * state.scale
        return Pred(
            v=value,
            pi=Categorical(logits=jnp.stack((value, -value), axis=-1)),
        ), state


@dataclass
class PureAgent:
    """Differentiable value-policy agent used to inspect gradient boundaries."""

    def apply(self, obs: jax.Array, state: State) -> tuple[Pred, State]:
        """Predict values and policies without runtime checks."""
        value = obs[..., 0] * state.scale
        return Pred(
            v=value,
            pi=Categorical(logits=jnp.stack((value, -value), axis=-1)),
        ), state


@dataclass
class Sequence:
    """Minimal sequence consumed by ``TDEst``."""

    obs: jax.Array
    nobs: jax.Array
    rew: jax.Array
    term: jax.Array
    trun: jax.Array


def test_td_est_matches_hand_computed_mixed_boundary_oracle():
    """Continuation, termination, truncation, and both flags align exactly."""
    expected_adv = jnp.array([0.75, -1.0, 13.0, -3.0])
    expected_ret = jnp.array([1.75, 1.0, 16.0, 1.0])
    component = TDEst(agent=Agent(), gamma=0.5, lam=0.5)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        nobs=jnp.array([[2.0], [-999.0], [30.0], [-999.0]]),
        rew=jnp.ones(4),
        term=jnp.array([False, True, False, True]),
        trun=jnp.array([False, False, True, True]),
    )

    error, est = checkify.checkify(jax.jit(component.estimate))(
        seq,
        State(scale=jnp.array(1.0)),
    )
    error.throw()

    assert jnp.allclose(est.adv, expected_adv)
    assert jnp.allclose(est.ret, expected_ret)
    assert est.pred.pi.evaluate(jnp.zeros(4, dtype=jnp.int32)).logp.shape == (4,)


def test_td_est_reuses_continuations_and_infers_only_the_open_tail():
    """Intermediate next observations are unused; an open tail bootstraps."""
    expected_adv = jnp.array([1.1171875, 0.46875, -0.125, -0.5])
    expected_ret = jnp.array([2.1171875, 2.46875, 2.875, 3.5])
    component = TDEst(agent=Agent(), gamma=0.5, lam=0.5)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        nobs=jnp.array([[-999.0], [-999.0], [-999.0], [5.0]]),
        rew=jnp.ones(4),
        term=jnp.zeros(4, dtype=jnp.bool_),
        trun=jnp.zeros(4, dtype=jnp.bool_),
    )

    error, est = checkify.checkify(jax.jit(component.estimate))(
        seq,
        State(scale=jnp.array(1.0)),
    )
    error.throw()

    assert jnp.allclose(est.adv, expected_adv)
    assert jnp.allclose(est.ret, expected_ret)


@pytest.mark.parametrize(
    ("lam", "expected_adv", "expected_ret"),
    [
        (
            0.0,
            jnp.array([1.0, 1.5, 2.0]),
            jnp.array([2.0, 3.5, 5.0]),
        ),
        (
            1.0,
            jnp.array([2.25, 2.5, 2.0]),
            jnp.array([3.25, 4.5, 5.0]),
        ),
    ],
)
def test_td_est_lambda_endpoints_match_their_analytic_returns(
    lam,
    expected_adv,
    expected_ret,
):
    """Lambda zero is one-step TD; lambda one is a bootstrapped return."""
    component = TDEst(agent=PureAgent(), gamma=0.5, lam=lam)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0]]),
        nobs=jnp.array([[2.0], [3.0], [4.0]]),
        rew=jnp.array([1.0, 2.0, 3.0]),
        term=jnp.zeros(3, dtype=jnp.bool_),
        trun=jnp.zeros(3, dtype=jnp.bool_),
    )

    eager = component.estimate(seq, State(scale=jnp.array(1.0)))
    compiled = jax.jit(component.estimate)(seq, State(scale=jnp.array(1.0)))

    assert jnp.allclose(eager.adv, expected_adv)
    assert jnp.allclose(eager.ret, expected_ret)
    chex.assert_trees_all_close(compiled, eager)


def test_td_est_supports_a_nonleading_sequence_axis_under_jit():
    """Batch axes remain aligned when time is not the leading axis."""
    component = TDEst(agent=Agent(), gamma=0.9, lam=0.8, seq_axis=1)
    obs = jnp.array([[[1.0], [2.0], [3.0]], [[1.0], [2.0], [3.0]]])
    seq = Sequence(
        obs=obs,
        nobs=jnp.array([[[2.0], [0.0], [4.0]], [[2.0], [0.0], [4.0]]]),
        rew=jnp.ones((2, 3)),
        term=jnp.array([[False, True, False], [False, True, False]]),
        trun=jnp.array([[False, False, True], [False, False, True]]),
    )

    error, est = checkify.checkify(jax.jit(component.estimate))(
        seq,
        State(scale=jnp.array(1.0)),
    )
    error.throw()

    chex.assert_shape(est.adv, (2, 3))
    chex.assert_shape(est.ret, (2, 3))
    assert jnp.allclose(est.adv[0], est.adv[1])
    assert jax.tree.structure(est.pred) == jax.tree.structure(
        jax.tree.map(lambda x: x, est.pred)
    )


def test_td_est_supports_nested_vmap_and_tree_structure():
    """Independent mapped axes compose outside a scalar sequence estimator."""
    component = TDEst(agent=PureAgent(), gamma=0.9, lam=0.8)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0]]),
        nobs=jnp.array([[2.0], [3.0]]),
        rew=jnp.ones(2),
        term=jnp.zeros(2, dtype=jnp.bool_),
        trun=jnp.zeros(2, dtype=jnp.bool_),
    )
    seq = jax.tree.map(
        lambda leaf: jnp.broadcast_to(leaf, (2, 3, *leaf.shape)),
        seq,
    )
    state = State(scale=jnp.ones((2, 3)))

    est = jax.jit(jax.vmap(jax.vmap(component.estimate)))(seq, state)

    chex.assert_shape(est.adv, (2, 3, 2))
    chex.assert_shape(est.ret, (2, 3, 2))
    chex.assert_shape(est.pred.pi.logits, (2, 3, 2, 2))
    assert jax.tree.structure(est) == jax.tree.structure(
        jax.tree.map(lambda leaf: leaf, est)
    )


def test_td_est_stops_target_gradients_but_preserves_prediction_gradients():
    """Fixed targets do not backpropagate while replayed predictions still do."""
    component = TDEst(agent=PureAgent(), gamma=0.9, lam=0.8)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0]]),
        nobs=jnp.array([[2.0], [3.0]]),
        rew=jnp.ones(2),
        term=jnp.zeros(2, dtype=jnp.bool_),
        trun=jnp.zeros(2, dtype=jnp.bool_),
    )

    def target_loss(scale: jax.Array) -> jax.Array:
        """Sum fixed value-estimation targets."""
        est = component.estimate(seq, State(scale=scale))
        return jnp.sum(est.adv) + jnp.sum(est.ret)

    def prediction_loss(scale: jax.Array) -> jax.Array:
        """Sum the differentiable replayed prediction."""
        return jnp.sum(component.estimate(seq, State(scale=scale)).pred.v)

    assert jax.grad(target_loss)(jnp.array(1.0)) == 0
    assert jax.grad(prediction_loss)(jnp.array(1.0)) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gamma": -0.1, "lam": 0.5}, "gamma must be between zero and one"),
        ({"gamma": 1.1, "lam": 0.5}, "gamma must be between zero and one"),
        ({"gamma": 0.9, "lam": -0.1}, "lam must be between zero and one"),
        ({"gamma": 0.9, "lam": 1.1}, "lam must be between zero and one"),
    ],
)
def test_td_est_rejects_invalid_static_configuration(kwargs, message):
    """Discount and trace parameters are probabilities at construction."""
    with pytest.raises(ValueError, match=message):
        TDEst(agent=PureAgent(), **kwargs)
