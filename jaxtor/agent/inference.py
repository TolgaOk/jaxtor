"""Agent inference aligned with sampled transition sequences.

Inference replays an agent and aligns values with true successors::

    inference = VPiNextVInference(agent=agent, seq_axis=1)
    infer = inference.apply(seq, agent_state)
    values, next_values = infer.v_tm1, infer.v_t
    policies = infer.pi_tm1

An agent may instead expose a state-bound Q-function. The corresponding
inference evaluates it at sampled actions and produces the Q and V sequences
expected by RLax off-policy returns::

    inference = QfnVnextInference(agent=agent, seq_axis=1)
    infer = inference.apply(seq, agent_state)
    q_t, v_t = infer.q_t, infer.v_t

Ordinary continuations reuse the next value. Sequence tails and truncations
evaluate their true successor observations.
"""

from __future__ import annotations

from functools import partial
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class VPred(Protocol):
    """Agent prediction containing one value per observation."""

    v: jax.Array


class VPiPred[Pi](Protocol):
    """Agent prediction containing a value and policy per observation."""

    v: jax.Array
    pi: Pi


class Qfn(Protocol):
    """State-bound action-value function consumed during sequence replay."""

    def evaluate(self, act: jax.Array) -> jax.Array: ...


class VQfnPred[Q](Protocol):
    """Agent prediction containing a value and state-bound Q-function."""

    v: jax.Array
    qfn: Q


class Agent[Pred, S](Protocol):
    """Read-only application capability consumed during sequence replay."""

    def apply(self, obs: jax.Array, state: S) -> tuple[Pred, S]: ...


class Sequence(Protocol):
    """Transition fields needed to align current and successor inference."""

    obs: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


class ActionSequence(Protocol):
    """Transition fields needed for action-value sequence inference."""

    obs: jax.Array
    act: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


@dataclass
class _BootstrapInput:
    """One successor value that may require explicit inference."""

    obs: jax.Array
    fallback: jax.Array
    infer: jax.Array


@dataclass
class _Endpoint[S]:
    """Inputs shared by the two conditional bootstrap branches."""

    obs: jax.Array
    fallback: jax.Array
    state: S


def _apply_endpoint[Pred: VPred, S](
    endpoint: _Endpoint[S],
    *,
    agent: Agent[Pred, S],
) -> jax.Array:
    """Evaluate one true successor observation."""
    pred, _ = agent.apply(endpoint.obs, endpoint.state)
    chex.assert_equal_shape([pred.v, endpoint.fallback])
    return pred.v


def _reuse_endpoint[S](endpoint: _Endpoint[S]) -> jax.Array:
    """Reuse the next current value for an ordinary continuation."""
    return endpoint.fallback


def _bootstrap[Pred: VPred, S](
    item: _BootstrapInput,
    *,
    agent: Agent[Pred, S],
    state: S,
) -> jax.Array:
    """Infer or reuse one successor value under a scalar condition."""
    return jax.lax.cond(
        item.infer,
        partial(_apply_endpoint, agent=agent),
        _reuse_endpoint,
        _Endpoint(obs=item.obs, fallback=item.fallback, state=state),
    )


def _validate_sequence(seq: Sequence, seq_axis: int) -> None:
    """Validate transition sample axes shared by inference components."""
    term = seq.term
    if term.ndim < 1:
        raise ValueError("sequence fields must contain a sequence axis")
    if not -term.ndim <= seq_axis < term.ndim:
        raise ValueError("seq_axis is outside the sequence sample axes")
    if term.shape[seq_axis] < 1:
        raise ValueError("sequence must not be empty")

    chex.assert_equal_shape([seq.obs, seq.nobs])
    chex.assert_equal_shape([term, seq.trun])
    chex.assert_equal_shape_prefix([seq.obs, term], term.ndim)


def _after_start(x: jax.Array, seq_axis: int) -> jax.Array:
    """Remove the starting entry along a validated sequence axis."""
    return jax.lax.slice_in_dim(
        x,
        start_index=1,
        limit_index=x.shape[seq_axis],
        axis=seq_axis,
    )


def _align_vnext[Pred: VPred, S](
    agent: Agent[Pred, S],
    seq: Sequence,
    v_next: jax.Array,
    state: S,
    seq_axis: int,
) -> jax.Array:
    """Align true-successor values without evaluating terminal observations."""
    chex.assert_equal_shape([v_next, _after_start(seq.term, seq_axis)])

    v_next = jnp.moveaxis(v_next, seq_axis, 0)
    nobs = jnp.moveaxis(seq.nobs, seq_axis, 0)
    term = jnp.moveaxis(seq.term, seq_axis, 0)
    trun = jnp.moveaxis(seq.trun, seq_axis, 0)

    fallback = jnp.concatenate((v_next, jnp.zeros_like(term[:1], dtype=v_next.dtype)))
    fallback = jnp.where(term, 0, fallback)
    tail = jnp.zeros_like(term).at[-1].set(True)
    infer = (~term) & (trun | tail)
    obs_shape = nobs.shape[term.ndim :]

    v_t = jax.lax.map(
        partial(_bootstrap, agent=agent, state=state),
        _BootstrapInput(
            obs=nobs.reshape((-1, *obs_shape)),
            fallback=fallback.reshape(-1),
            infer=infer.reshape(-1),
        ),
    ).reshape(term.shape)
    return jnp.moveaxis(v_t, 0, seq_axis)


@dataclass
class VNextVInference[AgentS]:
    """Replay values and align them with true transition successors.

    Ordinary continuations reuse the next current value. Truncations and an
    open sequence tail evaluate their true ``nobs``; natural terminations use
    zero without evaluating an invalid terminal successor.

    ``agent.apply`` is observational here: its returned state is discarded.
    Sequence-dependent agents require a sequence-aware inference component.

    Attributes:
        agent: Agent whose prediction contains ``V(s)``.
        seq_axis: Axis containing consecutive transitions.

    Public dataclasses:
        Inference: Current and true-successor values.

    Public methods:
        apply: Replay the agent and align values for every transition.
    """

    agent: Agent[VPred, AgentS]
    seq_axis: int = 0

    @dataclass
    class Inference:
        """Values aligned with ``T`` sampled transitions.

        Attributes:
            v_tm1: Current values ``V(s_t)``, shaped ``[..., T, ...]``.
            v_t: True-successor values ``V(s_{t+1})``, with the same shape.
        """

        v_tm1: jax.Array
        v_t: jax.Array

    def apply(
        self,
        seq: Sequence,
        state: AgentS,
    ) -> VNextVInference.Inference:
        """Return current and true-successor values for one sequence."""
        _validate_sequence(seq, self.seq_axis)
        pred, _ = self.agent.apply(seq.obs, state)
        chex.assert_equal_shape([pred.v, seq.term])
        return self.Inference(
            v_tm1=pred.v,
            v_t=_align_vnext(
                self.agent,
                seq,
                _after_start(pred.v, self.seq_axis),
                state,
                self.seq_axis,
            ),
        )


@dataclass
class VPiNextVInference[Pi, AgentS]:
    """Replay current values and policies, then align successor values.

    Ordinary continuations reuse the next current value. Truncations and an
    open sequence tail evaluate their true ``nobs``; natural terminations use
    zero without evaluating an invalid terminal successor.

    ``agent.apply`` is observational here: its returned state is discarded.
    Sequence-dependent agents require a sequence-aware inference component.

    Attributes:
        agent: Agent whose prediction contains ``V(s)`` and ``pi(.|s)``.
        seq_axis: Axis containing consecutive transitions.

    Public dataclasses:
        Inference: Current policy and current and true-successor values.

    Public methods:
        apply: Replay the agent and align inference for every transition.
    """

    agent: Agent[VPiPred[Pi], AgentS]
    seq_axis: int = 0

    @dataclass
    class Inference[PiData]:
        """Policy and values aligned with ``T`` sampled transitions.

        Attributes:
            v_tm1: Current values ``V(s_t)``, shaped ``[..., T, ...]``.
            pi_tm1: Current policies ``pi(.|s_t)``, with ``T`` sample entries.
            v_t: True-successor values ``V(s_{t+1})``, shaped like ``v_tm1``.
        """

        v_tm1: jax.Array
        pi_tm1: PiData
        v_t: jax.Array

    def apply(
        self,
        seq: Sequence,
        state: AgentS,
    ) -> VPiNextVInference.Inference[Pi]:
        """Return current policy and current and successor values."""
        _validate_sequence(seq, self.seq_axis)
        pred, _ = self.agent.apply(seq.obs, state)
        chex.assert_equal_shape([pred.v, seq.term])
        return self.Inference(
            v_tm1=pred.v,
            pi_tm1=pred.pi,
            v_t=_align_vnext(
                self.agent,
                seq,
                _after_start(pred.v, self.seq_axis),
                state,
                self.seq_axis,
            ),
        )


@dataclass
class QfnVnextInference[Q: Qfn, AgentS]:
    """Evaluate state-bound Q-functions and align successor values.

    A sequence begins at ``s_0``::

        s_0, a_0, r_1, s_1, a_1, ..., a_{T-1}, r_T, s_T

    The returned arrays are::

        q_t = [Q(s_1, a_1), ..., Q(s_{T-1}, a_{T-1})]
        v_t = [V(s_1), ..., V(s_T)]

    The starting prediction ``Q(s_0, a_0)`` is excluded. Learning evaluates it
    separately as ``q_tm1``. Ordinary continuations reuse inference at the next
    sampled observation. Truncations and the sequence tail evaluate their true
    ``nobs``; natural terminations use zero.

    ``agent.apply`` is observational here: its returned state is discarded.
    Sequence-dependent agents require a sequence-aware inference component.

    Attributes:
        agent: Agent whose prediction contains ``V(s)`` and ``Q(s, .)``.
        seq_axis: Axis containing consecutive transitions.

    Public dataclasses:
        Inference: Action-evaluated Q-values and true-successor values.

    Public methods:
        apply: Replay the agent and align inference for off-policy returns.
    """

    agent: Agent[VQfnPred[Q], AgentS]
    seq_axis: int = 0

    @dataclass
    class Inference:
        """Q-values and successor values aligned for RLax.

        Attributes:
            q_t: Values ``Q(s_t, a_t)`` for times ``1`` through ``T - 1``,
                shaped ``[..., T - 1]``.
            v_t: Values ``V(s_t)`` for times ``1`` through ``T``, shaped
                ``[..., T]``.
        """

        q_t: jax.Array
        v_t: jax.Array

    def apply(
        self,
        seq: ActionSequence,
        state: AgentS,
    ) -> QfnVnextInference.Inference:
        """Return action-evaluated Q-values and true-successor values."""
        _validate_sequence(seq, self.seq_axis)
        chex.assert_equal_shape_prefix([seq.act, seq.term], seq.term.ndim)
        pred, _ = self.agent.apply(_after_start(seq.obs, self.seq_axis), state)
        q_t = pred.qfn.evaluate(_after_start(seq.act, self.seq_axis))
        chex.assert_equal_shape([pred.v, q_t, _after_start(seq.term, self.seq_axis)])
        return self.Inference(
            q_t=q_t,
            v_t=_align_vnext(
                self.agent,
                seq,
                pred.v,
                state,
                self.seq_axis,
            ),
        )
