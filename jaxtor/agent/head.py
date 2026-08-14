"""Semantic value and policy heads for agent compositions.

Heads attach RL semantics to generic transforms while threading child state::

    v = VHead(net=value_net)
    v_state = v.init(value_state)
    values, v_state = v.apply(features, v_state)

    pi = CategoricalHead(n_actions=4, logits=logit_net)
    pi_state = pi.init(logit_state)
    dist, pi_state = pi.apply(features, pi_state)

The final input axis contains features; arbitrary leading axes are preserved.
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.agent.dist import Categorical, DiagNormal


def _assert_feature_array(x: jax.Array) -> None:
    """Require an array with a final feature axis."""
    if x.ndim < 1:
        raise ValueError("input must have a final feature axis")


def _assert_value(value: jax.Array, features: jax.Array) -> None:
    """Require one scalar value per leading feature index."""
    chex.assert_shape(value, features.shape[:-1])


class Transform[In, Out, S](Protocol):
    """Stateful transformation capability consumed by semantic heads."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


@dataclass
class VHead[NetS]:
    """Produce one scalar state value from each feature vector.

    Attributes:
        net: Transform producing vectors with a final singleton axis.

    Public dataclasses:
        State: Complete child-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce ``V(s)``.
    """

    net: Transform[jax.Array, jax.Array, NetS]

    @dataclass
    class State[NetData]:
        """Value-head state mirroring its child transform."""

        net: NetData

    def init(self, net: NetS) -> VHead.State[NetS]:
        """Store the initialized child state."""
        return self.State(net=net)

    def apply(
        self,
        features: jax.Array,
        state: VHead.State[NetS],
    ) -> tuple[jax.Array, VHead.State[NetS]]:
        """Produce one scalar value per feature vector."""
        _assert_feature_array(features)
        value, net = self.net.apply(features, state.net)
        chex.assert_shape(value, (*features.shape[:-1], 1))
        return value[..., 0], self.State(net=net)


@dataclass
class QHead[NetS]:
    """Produce all finite-action values from each feature vector.

    Attributes:
        n_actions: Size of the final action-value axis.
        net: Transform producing action-value vectors.

    Public dataclasses:
        State: Complete child-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce ``Q(s, .)``.
    """

    n_actions: int
    net: Transform[jax.Array, jax.Array, NetS]

    @dataclass
    class State[NetData]:
        """Action-value-head state mirroring its child transform."""

        net: NetData

    def __post_init__(self) -> None:
        """Validate the action count."""
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def init(self, net: NetS) -> QHead.State[NetS]:
        """Store the initialized child state."""
        return self.State(net=net)

    def apply(
        self,
        features: jax.Array,
        state: QHead.State[NetS],
    ) -> tuple[jax.Array, QHead.State[NetS]]:
        """Produce all action values for every feature vector."""
        _assert_feature_array(features)
        q, net = self.net.apply(features, state.net)
        chex.assert_shape(q, (*features.shape[:-1], self.n_actions))
        return q, self.State(net=net)


@dataclass
class QsaHead[ValS]:
    """Produce ``Q(s, a)`` from state features and one action.

    Attributes:
        value: Transform applied after concatenating features and actions.

    Public dataclasses:
        State: Complete child-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce ``Q(s, a)``.
    """

    value: Transform[jax.Array, jax.Array, ValS]

    @dataclass
    class State[ValData]:
        """State-action-value-head state."""

        value: ValData

    def init(self, value: ValS) -> QsaHead.State[ValS]:
        """Store the initialized child state."""
        return self.State(value=value)

    def apply(
        self,
        features: jax.Array,
        act: jax.Array,
        state: QsaHead.State[ValS],
    ) -> tuple[jax.Array, QsaHead.State[ValS]]:
        """Concatenate state features and actions, then evaluate them."""
        _assert_feature_array(features)
        _assert_feature_array(act)
        chex.assert_rank(act, features.ndim)
        chex.assert_equal_shape_prefix([features, act], features.ndim - 1)
        q, value = self.value.apply(
            jnp.concatenate((features, act), axis=-1),
            state.value,
        )
        _assert_value(q, features)
        return q, self.State(value=value)


@dataclass
class CategoricalHead[LogitS]:
    """Produce categorical policy distributions from feature vectors.

    Attributes:
        n_actions: Number of categorical actions.
        logits: Transform producing unnormalized logits.

    Public dataclasses:
        State: Complete logits-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce :class:`Categorical`.
    """

    n_actions: int
    logits: Transform[jax.Array, jax.Array, LogitS]

    @dataclass
    class State[LogitData]:
        """Categorical-head state."""

        logits: LogitData

    def __post_init__(self) -> None:
        """Validate the action count."""
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def init(self, logits: LogitS) -> CategoricalHead.State[LogitS]:
        """Store the initialized child state."""
        return self.State(logits=logits)

    def apply(
        self,
        features: jax.Array,
        state: CategoricalHead.State[LogitS],
    ) -> tuple[Categorical, CategoricalHead.State[LogitS]]:
        """Produce one categorical distribution per feature vector."""
        _assert_feature_array(features)
        logits, logits_state = self.logits.apply(features, state.logits)
        chex.assert_shape(logits, (*features.shape[:-1], self.n_actions))
        return Categorical(logits=logits), self.State(logits=logits_state)


@dataclass
class DiagNormalHead[LocS, LogScaleS]:
    """Produce diagonal-Normal policy distributions from feature vectors.

    Attributes:
        act_size: Size of the vector-valued action event.
        loc: Transform producing distribution locations.
        log_scale: Transform producing log standard deviations.

    Public dataclasses:
        State: Complete child-state tree.

    Public methods:
        init: Combine initialized child states.
        apply: Produce :class:`DiagNormal`.
    """

    act_size: int
    loc: Transform[jax.Array, jax.Array, LocS]
    log_scale: Transform[jax.Array, jax.Array, LogScaleS]

    @dataclass
    class State[LocData, LogScaleData]:
        """Diagonal-Normal-head child-state tree."""

        loc: LocData
        log_scale: LogScaleData

    def __post_init__(self) -> None:
        """Validate the action-event size."""
        if self.act_size < 1:
            raise ValueError("act_size must be positive")

    def init(
        self,
        loc: LocS,
        log_scale: LogScaleS,
    ) -> DiagNormalHead.State[LocS, LogScaleS]:
        """Combine initialized child states."""
        return self.State(loc=loc, log_scale=log_scale)

    def apply(
        self,
        features: jax.Array,
        state: DiagNormalHead.State[LocS, LogScaleS],
    ) -> tuple[DiagNormal, DiagNormalHead.State[LocS, LogScaleS]]:
        """Produce one diagonal-Normal distribution per feature vector."""
        _assert_feature_array(features)
        loc, loc_state = self.loc.apply(features, state.loc)
        log_scale, log_scale_state = self.log_scale.apply(
            features,
            state.log_scale,
        )
        shape = (*features.shape[:-1], self.act_size)
        chex.assert_shape(loc, shape)
        chex.assert_shape(log_scale, shape)
        return (
            DiagNormal(loc=loc, log_scale=log_scale),
            self.State(loc=loc_state, log_scale=log_scale_state),
        )
