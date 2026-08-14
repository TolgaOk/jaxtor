"""Agent inference aligned with sampled transition sequences.

Inference replays an agent and aligns values with true successors::

    inference = VPiNextVInference(agent=agent, seq_axis=1)
    infer = inference.apply(seq, agent_state)
    values, next_values = infer.v_tm1, infer.v_t
    policies = infer.pi_tm1

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


class Agent[Pred, S](Protocol):
    """Read-only application capability consumed during sequence replay."""

    def apply(self, obs: jax.Array, state: S) -> tuple[Pred, S]: ...


class Sequence(Protocol):
    """Transition fields needed to align current and successor inference."""

    obs: jax.Array
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


def _next_v[Pred: VPred, S](
    agent: Agent[Pred, S],
    seq: Sequence,
    v_tm1: jax.Array,
    state: S,
    seq_axis: int,
) -> jax.Array:
    """Align true-successor values without evaluating terminal observations."""
    term = seq.term
    trun = seq.trun
    if term.ndim < 1:
        raise ValueError("sequence fields must contain a sequence axis")
    if not -term.ndim <= seq_axis < term.ndim:
        raise ValueError("seq_axis is outside the sequence sample axes")
    if term.shape[seq_axis] < 1:
        raise ValueError("sequence must not be empty")

    chex.assert_equal_shape([seq.obs, seq.nobs])
    chex.assert_equal_shape([v_tm1, term, trun])
    chex.assert_equal_shape_prefix([seq.nobs, term], term.ndim)

    v_tm1 = jnp.moveaxis(v_tm1, seq_axis, 0)
    nobs = jnp.moveaxis(seq.nobs, seq_axis, 0)
    term = jnp.moveaxis(term, seq_axis, 0)
    trun = jnp.moveaxis(trun, seq_axis, 0)

    shifted = jnp.concatenate((v_tm1[1:], jnp.zeros_like(v_tm1[:1])))
    fallback = jnp.where(term, 0, shifted)
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
        pred, _ = self.agent.apply(seq.obs, state)
        return self.Inference(
            v_tm1=pred.v,
            v_t=_next_v(self.agent, seq, pred.v, state, self.seq_axis),
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
        pred, _ = self.agent.apply(seq.obs, state)
        return self.Inference(
            v_tm1=pred.v,
            pi_tm1=pred.pi,
            v_t=_next_v(self.agent, seq, pred.v, state, self.seq_axis),
        )
