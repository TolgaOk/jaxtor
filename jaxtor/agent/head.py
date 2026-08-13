"""Semantic value and policy heads for agent compositions."""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.agent.dist import Categorical, DiagNormal


class Transform[InputT, OutputT, StateT](Protocol):
    """Stateful transformation capability consumed by semantic heads."""

    def apply(
        self,
        x: InputT,
        state: StateT,
        /,
    ) -> tuple[OutputT, StateT]: ...


def _assert_feature_array(x: jax.Array) -> None:
    """Require an array with a final feature axis."""
    if x.ndim < 1:
        raise ValueError("input must have a final feature axis")


def _assert_value(value: jax.Array, features: jax.Array) -> None:
    """Require one scalar value per leading feature index."""
    chex.assert_shape(value, features.shape[:-1])


def _assert_vector_output(output: jax.Array, features: jax.Array) -> None:
    """Require an output vector for every leading feature index."""
    chex.assert_rank(output, features.ndim)
    chex.assert_equal_shape_prefix([features, output], features.ndim - 1)


@dataclass
class VHead[NetStateT]:
    """Produce one scalar state value from each feature vector.

    Attributes:
        net: Transform producing vectors with a final singleton axis.

    Public dataclasses:
        State: Complete child-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce ``V(s)``.
    """

    net: Transform[jax.Array, jax.Array, NetStateT]

    @dataclass
    class State[NetDataT]:
        """Value-head state mirroring its child transform."""

        net: NetDataT

    def init(self, net: NetStateT) -> VHead.State[NetStateT]:
        """Store the initialized child state."""
        return self.State(net=net)

    def apply(
        self,
        features: jax.Array,
        state: VHead.State[NetStateT],
    ) -> tuple[jax.Array, VHead.State[NetStateT]]:
        """Produce one scalar value per feature vector."""
        _assert_feature_array(features)
        value, net = self.net.apply(features, state.net)
        chex.assert_shape(value, (*features.shape[:-1], 1))
        return value[..., 0], self.State(net=net)


@dataclass
class QHead[NetStateT]:
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
    net: Transform[jax.Array, jax.Array, NetStateT]

    @dataclass
    class State[NetDataT]:
        """Action-value-head state mirroring its child transform."""

        net: NetDataT

    def __post_init__(self) -> None:
        """Validate the action count."""
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def init(self, net: NetStateT) -> QHead.State[NetStateT]:
        """Store the initialized child state."""
        return self.State(net=net)

    def apply(
        self,
        features: jax.Array,
        state: QHead.State[NetStateT],
    ) -> tuple[jax.Array, QHead.State[NetStateT]]:
        """Produce all action values for every feature vector."""
        _assert_feature_array(features)
        q, net = self.net.apply(features, state.net)
        chex.assert_shape(q, (*features.shape[:-1], self.n_actions))
        return q, self.State(net=net)


@dataclass
class QsaHead[ValueStateT]:
    """Produce ``Q(s, a)`` from state features and one action.

    Attributes:
        value: Transform applied after concatenating features and actions.

    Public dataclasses:
        State: Complete child-transform state.

    Public methods:
        init: Store the initialized child state.
        apply: Produce ``Q(s, a)``.
    """

    value: Transform[jax.Array, jax.Array, ValueStateT]

    @dataclass
    class State[ValueDataT]:
        """State-action-value-head state."""

        value: ValueDataT

    def init(self, value: ValueStateT) -> QsaHead.State[ValueStateT]:
        """Store the initialized child state."""
        return self.State(value=value)

    def apply(
        self,
        features: jax.Array,
        act: jax.Array,
        state: QsaHead.State[ValueStateT],
    ) -> tuple[jax.Array, QsaHead.State[ValueStateT]]:
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
class CategoricalHead[LogitsStateT]:
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
    logits: Transform[jax.Array, jax.Array, LogitsStateT]

    @dataclass
    class State[LogitsDataT]:
        """Categorical-head state."""

        logits: LogitsDataT

    def __post_init__(self) -> None:
        """Validate the action count."""
        if self.n_actions < 1:
            raise ValueError("n_actions must be positive")

    def init(self, logits: LogitsStateT) -> CategoricalHead.State[LogitsStateT]:
        """Store the initialized child state."""
        return self.State(logits=logits)

    def apply(
        self,
        features: jax.Array,
        state: CategoricalHead.State[LogitsStateT],
    ) -> tuple[Categorical, CategoricalHead.State[LogitsStateT]]:
        """Produce one categorical distribution per feature vector."""
        _assert_feature_array(features)
        logits, logits_state = self.logits.apply(features, state.logits)
        chex.assert_shape(logits, (*features.shape[:-1], self.n_actions))
        return Categorical(logits=logits), self.State(logits=logits_state)


@dataclass
class DiagNormalHead[LocStateT, LogScaleStateT]:
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
    loc: Transform[jax.Array, jax.Array, LocStateT]
    log_scale: Transform[jax.Array, jax.Array, LogScaleStateT]

    @dataclass
    class State[LocDataT, LogScaleDataT]:
        """Diagonal-Normal-head child-state tree."""

        loc: LocDataT
        log_scale: LogScaleDataT

    def __post_init__(self) -> None:
        """Validate the action-event size."""
        if self.act_size < 1:
            raise ValueError("act_size must be positive")

    def init(
        self,
        loc: LocStateT,
        log_scale: LogScaleStateT,
    ) -> DiagNormalHead.State[LocStateT, LogScaleStateT]:
        """Combine initialized child states."""
        return self.State(loc=loc, log_scale=log_scale)

    def apply(
        self,
        features: jax.Array,
        state: DiagNormalHead.State[LocStateT, LogScaleStateT],
    ) -> tuple[
        DiagNormal,
        DiagNormalHead.State[LocStateT, LogScaleStateT],
    ]:
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
