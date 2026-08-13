"""Composed value-policy agents and their action selectors."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.random as jrd
from chex import dataclass

from jaxtor.agent.dist import Distribution, Sample
from jaxtor.agent.head import (
    _assert_feature_array,
    _assert_value,
    _assert_vector_output,
)


class Selector[PiT: Distribution](Protocol):
    """Action-selection capability consumed by agent compositions."""

    def select(
        self,
        pi: PiT,
        key: jax.Array,
    ) -> tuple[Sample, jax.Array]: ...


@dataclass
class Draw:
    """Draw stochastic actions and advance the supplied random key.

    Public methods:
        select: Sample an action and return the next random key.
    """

    def select(
        self,
        pi: Distribution,
        key: jax.Array,
    ) -> tuple[Sample, jax.Array]:
        """Sample an action and return the next random key."""
        key, sample_key = jrd.split(key)
        return pi.sample(sample_key), key


@dataclass
class Mode:
    """Select modal actions without advancing the supplied random key.

    Public methods:
        select: Select the mode and evaluate its log-probability.
    """

    def select(
        self,
        pi: Distribution,
        key: jax.Array,
    ) -> tuple[Sample, jax.Array]:
        """Select the mode and evaluate its log-probability."""
        act = pi.mode()
        return Sample(act=act, logp=pi.evaluate(act).logp), key


class Transform[InputT, OutputT, StateT](Protocol):
    """Stateful transformation capability consumed by composed agents."""

    def apply(
        self,
        x: InputT,
        state: StateT,
        /,
    ) -> tuple[OutputT, StateT]: ...


@dataclass
class VPi[PiT: Distribution, BodyStateT, ValueStateT, PiStateT]:
    """Compose a body, value head, policy head, and action selector.

    Attributes:
        body: Transform mapping observations to common features.
        v: Transform producing ``V(s)``.
        pi: Transform producing a policy distribution.
        select: Strategy converting the policy into an action.

    Public dataclasses:
        State: Complete child-state tree and selector key.
        Pred: Value and policy prediction.

    Public methods:
        init: Combine initialized children and the selection key.
        apply: Produce value and policy predictions.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[jax.Array, jax.Array, BodyStateT]
    v: Transform[jax.Array, jax.Array, ValueStateT]
    pi: Transform[jax.Array, PiT, PiStateT]
    select: Selector[PiT]

    @dataclass
    class State[BodyDataT, ValueDataT, PiDataT]:
        """Value-policy child-state tree.

        Attributes:
            body: Body-transform state.
            v: Value-head state.
            pi: Policy-head state.
            select: Action-selection key.
        """

        body: BodyDataT
        v: ValueDataT
        pi: PiDataT
        select: jax.Array

    @dataclass
    class Pred[PiDataT: Distribution]:
        """Value and policy predictions aligned by leading axes."""

        v: jax.Array
        pi: PiDataT

    def init(
        self,
        key: jax.Array,
        body: BodyStateT,
        v: ValueStateT,
        pi: PiStateT,
    ) -> VPi.State[BodyStateT, ValueStateT, PiStateT]:
        """Combine initialized children and the selection key."""
        return self.State(body=body, v=v, pi=pi, select=key)

    def apply(
        self,
        obs: jax.Array,
        state: VPi.State[BodyStateT, ValueStateT, PiStateT],
    ) -> tuple[
        VPi.Pred[PiT],
        VPi.State[BodyStateT, ValueStateT, PiStateT],
    ]:
        """Produce value and policy predictions without selecting an action."""
        features, body = self.body.apply(obs, state.body)
        _assert_feature_array(features)
        value, v = self.v.apply(features, state.v)
        pi, pi_state = self.pi.apply(features, state.pi)
        _assert_value(value, features)
        return (
            self.Pred(v=value, pi=pi),
            self.State(body=body, v=v, pi=pi_state, select=state.select),
        )

    def act(
        self,
        obs: jax.Array,
        state: VPi.State[BodyStateT, ValueStateT, PiStateT],
    ) -> tuple[
        jax.Array,
        VPi.State[BodyStateT, ValueStateT, PiStateT],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        _assert_feature_array(features)
        pi, pi_state = self.pi.apply(features, state.pi)
        sample, key = self.select.select(pi, state.select)
        return sample.act, self.State(
            body=body,
            v=state.v,
            pi=pi_state,
            select=key,
        )


@dataclass
class VQPi[
    PiT: Distribution,
    BodyStateT,
    ValueStateT,
    QStateT,
    PiStateT,
]:
    """Compose value, action-value, policy, and selection components.

    Attributes:
        body: Transform mapping observations to common features.
        v: Transform producing ``V(s)``.
        q: Transform producing ``Q(s, .)``.
        pi: Transform producing a policy distribution.
        select: Strategy converting the policy into an action.

    Public dataclasses:
        State: Complete child-state tree and selector key.
        Pred: Value, action values, and policy prediction.

    Public methods:
        init: Combine initialized children and the selection key.
        apply: Produce value, action-value, and policy predictions.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[jax.Array, jax.Array, BodyStateT]
    v: Transform[jax.Array, jax.Array, ValueStateT]
    q: Transform[jax.Array, jax.Array, QStateT]
    pi: Transform[jax.Array, PiT, PiStateT]
    select: Selector[PiT]

    @dataclass
    class State[BodyDataT, ValueDataT, QDataT, PiDataT]:
        """Value-action-values-policy child-state tree."""

        body: BodyDataT
        v: ValueDataT
        q: QDataT
        pi: PiDataT
        select: jax.Array

    @dataclass
    class Pred[PiDataT: Distribution]:
        """Value, action values, and policy predictions."""

        v: jax.Array
        q: jax.Array
        pi: PiDataT

    def init(
        self,
        key: jax.Array,
        body: BodyStateT,
        v: ValueStateT,
        q: QStateT,
        pi: PiStateT,
    ) -> VQPi.State[BodyStateT, ValueStateT, QStateT, PiStateT]:
        """Combine initialized children and the selection key."""
        return self.State(body=body, v=v, q=q, pi=pi, select=key)

    def apply(
        self,
        obs: jax.Array,
        state: VQPi.State[BodyStateT, ValueStateT, QStateT, PiStateT],
    ) -> tuple[
        VQPi.Pred[PiT],
        VQPi.State[BodyStateT, ValueStateT, QStateT, PiStateT],
    ]:
        """Produce value, action-value, and policy predictions."""
        features, body = self.body.apply(obs, state.body)
        _assert_feature_array(features)
        value, v = self.v.apply(features, state.v)
        q, q_state = self.q.apply(features, state.q)
        pi, pi_state = self.pi.apply(features, state.pi)
        _assert_value(value, features)
        _assert_vector_output(q, features)
        return (
            self.Pred(v=value, q=q, pi=pi),
            self.State(
                body=body,
                v=v,
                q=q_state,
                pi=pi_state,
                select=state.select,
            ),
        )

    def act(
        self,
        obs: jax.Array,
        state: VQPi.State[BodyStateT, ValueStateT, QStateT, PiStateT],
    ) -> tuple[
        jax.Array,
        VQPi.State[BodyStateT, ValueStateT, QStateT, PiStateT],
    ]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        _assert_feature_array(features)
        pi, pi_state = self.pi.apply(features, state.pi)
        sample, key = self.select.select(pi, state.select)
        return sample.act, self.State(
            body=body,
            v=state.v,
            q=state.q,
            pi=pi_state,
            select=key,
        )
