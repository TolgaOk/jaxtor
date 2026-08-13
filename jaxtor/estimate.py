"""Value estimates formed by replaying compatible agents over sequences."""

from __future__ import annotations

from functools import partial
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Pred(Protocol):
    """Agent prediction required by temporal-difference estimation."""

    v: jax.Array


class Agent[PredT: Pred, StateT](Protocol):
    """Read-only replay capability required by :class:`TDEst`."""

    def apply(
        self,
        obs: jax.Array,
        state: StateT,
    ) -> tuple[PredT, StateT]: ...


class Sequence(Protocol):
    """Sequential fields required by :class:`TDEst`."""

    obs: jax.Array
    nobs: jax.Array
    rew: jax.Array
    term: jax.Array
    trun: jax.Array


@dataclass
class TDEst[PredT: Pred, AgentStateT]:
    """Replay an agent and form TD(lambda) advantages and value targets.

    Current observations are evaluated together. Ordinary successor values are
    reused from the next current prediction. True successor observations are
    evaluated only at truncations and at a nonterminal sequence tail; natural
    termination instead contributes a zero bootstrap value.

    ``agent.apply`` is observational here: returned agent states are discarded.
    Agents whose application carries sequence-dependent state require a
    sequence-aware estimator instead.

    Attributes:
        agent: Agent whose predictions contain ``V(s)``.
        gamma: Reward discount in ``[0, 1]``.
        lam: TD trace parameter in ``[0, 1]``.
        seq_axis: Sequence axis containing consecutive transitions.

    Public dataclasses:
        Estimate: Replayed predictions, advantages, and value targets.

    Public methods:
        estimate: Replay the agent and compute one estimate per transition.
    """

    agent: Agent[PredT, AgentStateT]
    gamma: float
    lam: float
    seq_axis: int = 0

    @dataclass
    class Estimate[PredDataT]:
        """Predictions and fixed TD(lambda) estimates.

        Attributes:
            pred: Agent predictions aligned with current observations.
            adv: TD(lambda) errors used as advantages.
            ret: Bootstrapped value targets, equal to ``pred.v + adv``.
        """

        pred: PredDataT
        adv: jax.Array
        ret: jax.Array

    @dataclass
    class _Endpoint[StateT]:
        """One possible explicit successor evaluation."""

        obs: jax.Array
        fallback: jax.Array
        state: StateT

    @dataclass
    class _Step:
        """One reverse TD(lambda) recursion step."""

        delta: jax.Array
        trace: jax.Array

    def __post_init__(self) -> None:
        """Validate the static estimator configuration."""
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be between zero and one")

    def _infer(self, endpoint: TDEst._Endpoint[AgentStateT]) -> jax.Array:
        """Evaluate one successor observation that requires bootstrapping."""
        pred, _ = self.agent.apply(endpoint.obs, endpoint.state)
        chex.assert_equal_shape([pred.v, endpoint.fallback])
        return pred.v

    @staticmethod
    def _reuse(endpoint: TDEst._Endpoint[AgentStateT]) -> jax.Array:
        """Reuse an already aligned successor value."""
        return endpoint.fallback

    def _bootstrap(
        self,
        item: tuple[jax.Array, jax.Array, jax.Array],
        state: AgentStateT,
    ) -> jax.Array:
        """Evaluate or reuse one successor value under a scalar condition."""
        obs, fallback, infer = item
        return jax.lax.cond(
            infer,
            self._infer,
            self._reuse,
            self._Endpoint(obs=obs, fallback=fallback, state=state),
        )

    def _next_v(
        self,
        seq: Sequence,
        pred: PredT,
        state: AgentStateT,
    ) -> jax.Array:
        """Align successor values without evaluating terminal observations."""
        term = seq.term
        trun = seq.trun
        if term.ndim < 1:
            raise ValueError("sequence fields must contain a sequence axis")
        if not -term.ndim <= self.seq_axis < term.ndim:
            raise ValueError("seq_axis is outside the sequence sample axes")
        if term.shape[self.seq_axis] < 1:
            raise ValueError("sequence must not be empty")

        chex.assert_equal_shape([seq.obs, seq.nobs])
        chex.assert_equal_shape([pred.v, term, trun])
        chex.assert_equal_shape_prefix([seq.nobs, term], term.ndim)

        v = jnp.moveaxis(pred.v, self.seq_axis, 0)
        nobs = jnp.moveaxis(seq.nobs, self.seq_axis, 0)
        term = jnp.moveaxis(term, self.seq_axis, 0)
        trun = jnp.moveaxis(trun, self.seq_axis, 0)

        shifted = jnp.concatenate((v[1:], jnp.zeros_like(v[:1])))
        fallback = jnp.where(term, 0, shifted)
        tail = jnp.zeros_like(term).at[-1].set(True)
        infer = (~term) & (trun | tail)

        obs_shape = nobs.shape[term.ndim :]
        next_v = jax.lax.map(
            partial(self._bootstrap, state=state),
            (
                nobs.reshape((-1, *obs_shape)),
                fallback.reshape(-1),
                infer.reshape(-1),
            ),
        ).reshape(term.shape)
        return jnp.moveaxis(next_v, 0, self.seq_axis)

    @staticmethod
    def _backward(
        next_adv: jax.Array,
        step: TDEst._Step,
    ) -> tuple[jax.Array, jax.Array]:
        """Accumulate one TD(lambda) error in reverse sequence order."""
        adv = step.delta + step.trace * next_adv
        return adv, adv

    def estimate(
        self,
        seq: Sequence,
        state: AgentStateT,
    ) -> TDEst.Estimate[PredT]:
        """Return replayed predictions, advantages, and value targets."""
        pred, _ = self.agent.apply(seq.obs, state)
        next_v = self._next_v(seq, pred, state)
        chex.assert_equal_shape([pred.v, seq.rew, next_v])

        term = jnp.moveaxis(seq.term, self.seq_axis, 0)
        done = jnp.moveaxis(seq.term | seq.trun, self.seq_axis, 0)
        rew = jnp.moveaxis(seq.rew, self.seq_axis, 0)
        v = jnp.moveaxis(pred.v, self.seq_axis, 0)
        next_v = jnp.moveaxis(next_v, self.seq_axis, 0)
        delta = rew + self.gamma * (~term).astype(rew.dtype) * next_v - v
        trace = self.gamma * self.lam * (~done).astype(rew.dtype)

        _, adv = jax.lax.scan(
            self._backward,
            jnp.zeros_like(delta[0]),
            self._Step(delta=delta, trace=trace),
            reverse=True,
        )
        adv = jax.lax.stop_gradient(jnp.moveaxis(adv, 0, self.seq_axis))
        ret = jax.lax.stop_gradient(adv + pred.v)
        return self.Estimate(pred=pred, adv=adv, ret=ret)
