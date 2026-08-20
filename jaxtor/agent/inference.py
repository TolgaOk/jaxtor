"""Dense agent inference over sampled transition sequences.

Each component combines a structural agent interface with the transition fields
required by one alignment rule::

    inference = VPiNextVInference(agent=agent, seq_axis=1)
    infer = inference.apply(seq, agent_state)
    values, next_values = infer.v_tm1, infer.v_t
    policies = infer.pi_tm1

Action-value inference uses independent ``q`` and ``v`` methods::

    inference = QNextVInference(agent=agent, seq_axis=1)
    infer = inference.apply(seq, agent_state)
    q_t, v_t = infer.q_t, infer.v_t

Every ``nobs`` is evaluated directly. Algorithms remain responsible for
discounting terminal successor values.
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
from chex import dataclass


class ValuePolicy[Pi](Protocol):
    """Joint value-policy result consumed by inference."""

    v: jax.Array
    pi: Pi


class VPiAgent[Pi, S](Protocol):
    """Value-policy agent consumed by inference."""

    def v(self, obs: jax.Array, state: S) -> tuple[jax.Array, S]: ...
    def vpi(self, obs: jax.Array, state: S) -> tuple[ValuePolicy[Pi], S]: ...


class QVAgent[S](Protocol):
    """Action-value and value agent consumed by inference."""

    def q(self, obs: jax.Array, act: jax.Array, state: S) -> tuple[jax.Array, S]: ...
    def v(self, obs: jax.Array, state: S) -> tuple[jax.Array, S]: ...


class Sequence(Protocol):
    """Transition sequence consumed by value-policy inference."""

    obs: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


class ActionSequence(Protocol):
    """Transition sequence consumed by action-value inference."""

    obs: jax.Array
    act: jax.Array
    nobs: jax.Array
    term: jax.Array
    trun: jax.Array


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


@dataclass
class VPiNextVInference[Pi, AgentS]:
    """Evaluate current value-policy results and stored successor values.

    Every ``nobs`` is evaluated directly through the value-only endpoint.
    Terminal values are returned unchanged because the consuming algorithm owns
    discounting.

    The states returned by ``agent.vpi`` and ``agent.v`` are ignored. Both
    methods receive the supplied state. Use a sequence-aware inference
    component when evaluation must advance agent state.

    Required protocols::

        agent.v(obs, agent_state) -> (value: jax.Array, agent_state)
        agent.vpi(obs, agent_state) -> (value_policy, agent_state)
        value_policy.v: jax.Array
        value_policy.pi: Pi
        seq.obs: jax.Array
        seq.nobs: jax.Array
        seq.term: jax.Array
        seq.trun: jax.Array

    Attributes:
        agent: Agent providing joint ``V(s)`` and ``pi(.|s)`` plus value-only
            evaluation.
        seq_axis: Axis containing consecutive transitions.

    Public dataclasses:
        Inference: Current policy and current and true-successor values.

    Public methods:
        apply: Evaluate policy and values for every transition.
    """

    agent: VPiAgent[Pi, AgentS]
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
        value_policy, _ = self.agent.vpi(seq.obs, state)
        v_t, _ = self.agent.v(seq.nobs, state)
        chex.assert_equal_shape([value_policy.v, v_t, seq.term])
        return self.Inference(
            v_tm1=value_policy.v,
            pi_tm1=value_policy.pi,
            v_t=v_t,
        )


@dataclass
class QNextVInference[AgentS]:
    """Evaluate Q-values and values on their dense sequence inputs.

    A sequence begins at ``s_0``::

        s_0, a_0, r_1, s_1, a_1, ..., a_{T-1}, r_T, s_T

    The returned arrays are::

        q_t = [Q(s_1, a_1), ..., Q(s_{T-1}, a_{T-1})]
        v_t = [V(s_1), ..., V(s_T)]

    The starting value ``Q(s_0, a_0)`` is excluded. Learning evaluates current
    Q-values separately with the parameters being optimized. Successor values
    are evaluated directly from every ``nobs``. Terminal values are returned
    unchanged because RLax discounts determine whether they contribute.

    The states returned by ``agent.q`` and ``agent.v`` are ignored. Both methods
    receive the supplied state. Use a sequence-aware inference component when
    evaluation must advance agent state.

    Required protocols::

        agent.q(obs, action, agent_state) -> (q: jax.Array, agent_state)
        agent.v(obs, agent_state) -> (value: jax.Array, agent_state)
        seq.obs: jax.Array
        seq.act: jax.Array
        seq.nobs: jax.Array
        seq.term: jax.Array
        seq.trun: jax.Array

    Attributes:
        agent: Agent providing independent ``Q(s, a)`` and ``V(s)`` methods.
        seq_axis: Axis containing consecutive transitions.

    Public dataclasses:
        Inference: Action-evaluated Q-values and true-successor values.

    Public methods:
        apply: Evaluate action-values and successor values for off-policy returns.
    """

    agent: QVAgent[AgentS]
    seq_axis: int = 0

    @dataclass
    class Inference:
        """Q-values and raw successor values aligned for RLax.

        Attributes:
            q_t: Values ``Q(s_t, a_t)`` for times ``1`` through ``T - 1``,
                shaped ``[..., T - 1]``.
            v_t: Values of every stored ``nobs``, shaped ``[..., T]``.
        """

        q_t: jax.Array
        v_t: jax.Array

    def apply(
        self,
        seq: ActionSequence,
        state: AgentS,
    ) -> QNextVInference.Inference:
        """Return action-evaluated Q-values and true-successor values."""
        _validate_sequence(seq, self.seq_axis)
        chex.assert_equal_shape_prefix([seq.act, seq.term], seq.term.ndim)
        q_t, _ = self.agent.q(
            _after_start(seq.obs, self.seq_axis),
            _after_start(seq.act, self.seq_axis),
            state,
        )
        v_t, _ = self.agent.v(seq.nobs, state)
        chex.assert_equal_shape([q_t, _after_start(seq.term, self.seq_axis)])
        chex.assert_equal_shape([v_t, seq.term])
        return self.Inference(q_t=q_t, v_t=v_t)
