"""Composed value-policy agents."""

from __future__ import annotations

from typing import Protocol

import jax
from chex import dataclass


class Sample(Protocol):
    """Sampled action capability consumed during acting."""

    act: jax.Array


class Distribution(Protocol):
    """Action-distribution capability consumed by composed agents."""

    def sample(self, key: jax.Array) -> Sample: ...

    def mode(self) -> jax.Array: ...


class Transform[In, Out, S](Protocol):
    """Stateful transformation capability consumed by composed agents."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


@dataclass
class VPi[Dist: Distribution, BodyS, ValS, PiS]:
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
    pi: Transform[jax.Array, Dist, PiS]
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
    class Pred[DistData]:
        """Value and policy predictions aligned by leading axes."""

        v: jax.Array
        pi: DistData

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
        VPi.Pred[Dist],
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
        jax.Array,
        VPi.State[BodyS, ValS, PiS],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key).act
        return act, self.State(
            body=body,
            v=state.v,
            pi=pi,
            key=key,
        )


@dataclass
class VQPi[
    Dist: Distribution,
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
    pi: Transform[jax.Array, Dist, PiS]
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
    class Pred[DistData]:
        """Value, action values, and policy predictions."""

        v: jax.Array
        q: jax.Array
        pi: DistData

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
        VQPi.Pred[Dist],
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
        jax.Array,
        VQPi.State[BodyS, ValS, QS, PiS],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key).act
        return act, self.State(
            body=body,
            v=state.v,
            q=state.q,
            pi=pi,
            key=key,
        )
