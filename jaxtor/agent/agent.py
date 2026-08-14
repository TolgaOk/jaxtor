"""Composed value-policy agents.

An agent joins a shared body and semantic heads while state remains explicit::

    agent = VPi(body=body, v=v, pi=pi)
    state = agent.init(key, body=body_state, v=v_state, pi=pi_state)
    pred, applied_state = agent.apply(obs, state)
    act, acted_state = agent.act(obs, state)

``apply`` returns all predictions. ``act`` evaluates only action dependencies.
"""

from __future__ import annotations

from typing import Protocol

import jax
from chex import dataclass


class Distribution[Act, Eval](Protocol):
    """Action-distribution capability consumed by composed agents."""

    def sample(self, key: jax.Array) -> Act: ...
    def evaluate(self, act: Act) -> Eval: ...
    def mode(self) -> Act: ...


class Transform[In, Out, S](Protocol):
    """Stateful transformation capability consumed by composed agents."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


@dataclass
class VPi[Act, Eval, BodyS, ValS, PiS]:
    """Compose a body, value head, and policy head into an acting agent.

    Attributes:
        body: Transform mapping observations to common features.
        v: Transform producing ``V(s)``.
        pi: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        Pred: Value and policy prediction.

    Public methods:
        init: Combine initialized children and the selection key.
        apply: Produce value and policy predictions.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[jax.Array, jax.Array, BodyS]
    v: Transform[jax.Array, jax.Array, ValS]
    pi: Transform[jax.Array, Distribution[Act, Eval], PiS]
    deterministic: bool = False

    @dataclass
    class State[BodyData, ValData, PiData]:
        """Value-policy child-state tree.

        Attributes:
            body: Body-transform state.
            v: Value-head state.
            pi: Policy-head state.
            key: Action-sampling key.
        """

        body: BodyData
        v: ValData
        pi: PiData
        key: jax.Array

    @dataclass
    class Pred[ActData, EvalData]:
        """Value and policy predictions aligned by leading axes."""

        v: jax.Array
        pi: Distribution[ActData, EvalData]

    def init(
        self,
        key: jax.Array,
        body: BodyS,
        v: ValS,
        pi: PiS,
    ) -> VPi.State[BodyS, ValS, PiS]:
        """Combine initialized children and the sampling key."""
        return self.State(body=body, v=v, pi=pi, key=key)

    def apply(
        self,
        obs: jax.Array,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[
        VPi.Pred[Act, Eval],
        VPi.State[BodyS, ValS, PiS],
    ]:
        """Produce value and policy predictions without selecting an action."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.v.apply(features, state.v)
        dist, pi = self.pi.apply(features, state.pi)
        return (
            self.Pred(v=value, pi=dist),
            self.State(body=body, v=v, pi=pi, key=state.key),
        )

    def act(
        self,
        obs: jax.Array,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[
        Act,
        VPi.State[BodyS, ValS, PiS],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, self.State(
            body=body,
            v=state.v,
            pi=pi,
            key=key,
        )


@dataclass
class VQPi[
    Act,
    Eval,
    BodyS,
    ValS,
    QS,
    PiS,
]:
    """Compose value, action-value, and policy components into an agent.

    Attributes:
        body: Transform mapping observations to common features.
        v: Transform producing ``V(s)``.
        q: Transform producing ``Q(s, .)``.
        pi: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        Pred: Value, action values, and policy prediction.

    Public methods:
        init: Combine initialized children and the selection key.
        apply: Produce value, action-value, and policy predictions.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[jax.Array, jax.Array, BodyS]
    v: Transform[jax.Array, jax.Array, ValS]
    q: Transform[jax.Array, jax.Array, QS]
    pi: Transform[jax.Array, Distribution[Act, Eval], PiS]
    deterministic: bool = False

    @dataclass
    class State[BodyData, ValData, QData, PiData]:
        """Value-action-values-policy child-state tree."""

        body: BodyData
        v: ValData
        q: QData
        pi: PiData
        key: jax.Array

    @dataclass
    class Pred[ActData, EvalData]:
        """Value, action values, and policy predictions."""

        v: jax.Array
        q: jax.Array
        pi: Distribution[ActData, EvalData]

    def init(
        self,
        key: jax.Array,
        body: BodyS,
        v: ValS,
        q: QS,
        pi: PiS,
    ) -> VQPi.State[BodyS, ValS, QS, PiS]:
        """Combine initialized children and the sampling key."""
        return self.State(body=body, v=v, q=q, pi=pi, key=key)

    def apply(
        self,
        obs: jax.Array,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[
        VQPi.Pred[Act, Eval],
        VQPi.State[BodyS, ValS, QS, PiS],
    ]:
        """Produce value, action-value, and policy predictions."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.v.apply(features, state.v)
        q, q_state = self.q.apply(features, state.q)
        dist, pi = self.pi.apply(features, state.pi)
        return (
            self.Pred(v=value, q=q, pi=dist),
            self.State(
                body=body,
                v=v,
                q=q_state,
                pi=pi,
                key=state.key,
            ),
        )

    def act(
        self,
        obs: jax.Array,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[
        Act,
        VQPi.State[BodyS, ValS, QS, PiS],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, self.State(
            body=body,
            v=state.v,
            q=state.q,
            pi=pi,
            key=key,
        )
