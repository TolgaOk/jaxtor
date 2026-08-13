"""Tests for sequence-replayed temporal-difference estimates."""

import chex
import jax
import jax.numpy as jnp
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
class Sequence:
    """Minimal sequence consumed by ``TDEst``."""

    obs: jax.Array
    nobs: jax.Array
    rew: jax.Array
    term: jax.Array
    trun: jax.Array


def test_td_est_skips_terminal_successors_and_preserves_prediction():
    """Termination suppresses inference while truncation still bootstraps."""
    component = TDEst(agent=Agent(), gamma=0.9, lam=0.8)
    seq = Sequence(
        obs=jnp.array([[1.0], [2.0], [3.0]]),
        nobs=jnp.array([[2.0], [-999.0], [4.0]]),
        rew=jnp.ones(3),
        term=jnp.array([False, True, False]),
        trun=jnp.array([False, False, True]),
    )

    error, est = checkify.checkify(jax.jit(component.estimate))(
        seq,
        State(scale=jnp.array(1.0)),
    )
    error.throw()

    assert jnp.allclose(est.adv, jnp.array([1.08, -1.0, 1.6]))
    assert jnp.allclose(est.ret, jnp.array([2.08, 1.0, 4.6]))
    assert est.pred.pi.evaluate(jnp.zeros(3, dtype=jnp.int32)).logp.shape == (3,)


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
