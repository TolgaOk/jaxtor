"""Composable transforms, semantic heads, and acting compositions.

Configured components are static, while their nested ``State`` dataclasses
mirror the dynamic child-state tree. Trainable leaves are marked by
:class:`Param`; :func:`partition` selects them only at the optimizer boundary.

Neural-library callables are adapted at the leaves. For example, given an
observation normalizer and semantic value and policy heads, an Equinox body is
partitioned before entering the component tree::

    body_eqx = eqx.nn.Sequential(
        [
            eqx.nn.Linear(obs_size, hidden_size, key=body_key),
            eqx.nn.Lambda(jax.nn.tanh),
        ]
    )
    body_params, body_static = eqx.partition(body_eqx, eqx.is_array)

    body_net = Module(static=body_static)
    body_net_state = body_net.init(body_params)
    body = NormModel(norm=obs_norm, model=body_net)
    body_state = body.init(obs_norm_state, body_net_state)

    agent = VPi(
        body=body,
        v=value_head,
        pi=policy_head,
        select=Draw(),
    )

``Module`` keeps the callable's static structure on the configured component
and its array parameters in ``Module.State``. Outer components add semantics
and mirror the same nesting in ``NormModel.State``, head states, and finally
``VPi.State``. Calling ``agent.apply`` follows that tree inward and returns the
updated state with the same structure.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
from chex import dataclass

from jaxtor.dist import Categorical, DiagNormal, Distribution, Sample


@dataclass
class Param[ValueT]:
    """Mark one pytree as trainable without changing its JAX behavior.

    Attributes:
        value: Trainable array or nested array pytree.
    """

    value: ValueT


@dataclass
class Partition[ParamsT, FrozenT]:
    """Trainable and frozen views of one component state.

    Attributes:
        params: State-shaped tree containing :class:`Param` leaves.
        frozen: Complementary state-shaped tree containing all other leaves.
    """

    params: ParamsT
    frozen: FrozenT


def _is_param(value: Any) -> bool:
    """Identify trainable markers while partitioning a state tree."""
    return isinstance(value, Param)


def _is_partition_leaf(value: Any) -> bool:
    """Treat parameters and empty complementary leaves atomically."""
    return value is None or isinstance(value, Param)


def partition[StateT](state: StateT) -> Partition[StateT, StateT]:
    """Split a state into trainable and frozen state-shaped pytrees."""
    params = jax.tree.map(
        lambda value: value if isinstance(value, Param) else None,
        state,
        is_leaf=_is_param,
    )
    frozen = jax.tree.map(
        lambda value: None if isinstance(value, Param) else value,
        state,
        is_leaf=_is_param,
    )
    return Partition(params=params, frozen=frozen)


def combine[StateT](params: StateT, frozen: StateT) -> StateT:
    """Reconstruct a component state from complementary partition trees."""
    return jax.tree.map(
        lambda param, fixed: fixed if param is None else param,
        params,
        frozen,
        is_leaf=_is_partition_leaf,
    )


class Transform[InputT, OutputT, StateT](Protocol):
    """Stateful transformation capability consumed by composed models."""

    def apply(
        self,
        x: InputT,
        state: StateT,
        /,
    ) -> tuple[OutputT, StateT]: ...


class Normalizer[InputT, StateT](Protocol):
    """Normalization capability consumed by :class:`NormModel`."""

    def apply(
        self,
        x: InputT,
        state: StateT,
        /,
    ) -> tuple[InputT, StateT]: ...

    def update(
        self,
        x: InputT,
        state: StateT,
        /,
    ) -> StateT: ...


@runtime_checkable
class Function[InputT, OutputT](Protocol):
    """Callable capability consumed by :class:`Module`."""

    def __call__(self, x: InputT, /) -> OutputT: ...


class Selector[PiT: Distribution](Protocol):
    """Action-selection capability consumed by acting compositions."""

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
class Module[OutputT]:
    """Adapt a partitioned callable pytree to :class:`Transform`.

    The component holds the non-parameter partition. :class:`State` holds the
    complementary parameter partition under one :class:`Param` marker. This
    supports Equinox modules without making Equinox a core dependency.

    Attributes:
        static: Callable pytree with parameter leaves replaced by ``None``.
        in_ndim: Number of trailing axes consumed by one callable invocation.

    Public dataclasses:
        State: Marked parameter partition of the callable pytree.

    Public methods:
        init: Store an initialized parameter partition.
        apply: Apply the reconstructed callable.
    """

    static: Function[jax.Array, OutputT]
    in_ndim: int = 1

    @dataclass
    class State[ModuleOutputT]:
        """Dynamic parameter partition of a callable module."""

        params: Param[Function[jax.Array, ModuleOutputT]]

    def __post_init__(self) -> None:
        """Validate the rank consumed by one callable invocation."""
        if self.in_ndim < 0:
            raise ValueError("in_ndim must be nonnegative")

    def init(
        self,
        params: Function[jax.Array, OutputT],
    ) -> Module.State[OutputT]:
        """Store an initialized parameter partition."""
        return self.State(params=Param(value=params))

    def _combine(
        self,
        params: Function[jax.Array, OutputT],
    ) -> Function[jax.Array, OutputT]:
        """Combine complementary parameter and static partitions."""
        return jax.tree.map(
            lambda array, static: static if array is None else array,
            params,
            self.static,
            is_leaf=lambda leaf: leaf is None,
        )

    def apply(
        self,
        x: jax.Array,
        state: Module.State[OutputT],
    ) -> tuple[OutputT, Module.State[OutputT]]:
        """Apply the callable independently over arbitrary leading axes."""
        if x.ndim < self.in_ndim:
            raise ValueError(f"input rank must be at least {self.in_ndim}")

        if self.in_ndim == 0:
            lead_shape = x.shape
            input_shape: tuple[int, ...] = ()
        else:
            lead_shape = x.shape[: -self.in_ndim]
            input_shape = x.shape[-self.in_ndim :]

        fn = self._combine(state.params.value)
        output = jax.vmap(fn)(x.reshape((-1, *input_shape)))
        output = jax.tree.map(
            lambda leaf: leaf.reshape((*lead_shape, *leaf.shape[1:])),
            output,
        )
        return output, state


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


@dataclass
class Model[InputT, HiddenT, OutputT, BodyStateT, HeadStateT]:
    """Apply one feature transform followed by one prediction head.

    Attributes:
        body: Transform mapping inputs to hidden representations.
        head: Transform mapping hidden representations to predictions.

    Public dataclasses:
        State: Complete child-state tree.

    Public methods:
        init: Combine initialized child states.
        apply: Produce predictions and advance child states.
    """

    body: Transform[InputT, HiddenT, BodyStateT]
    head: Transform[HiddenT, OutputT, HeadStateT]

    @dataclass
    class State[BodyDataT, HeadDataT]:
        """Shared-body model child-state tree."""

        body: BodyDataT
        head: HeadDataT

    def init(
        self,
        body: BodyStateT,
        head: HeadStateT,
    ) -> Model.State[BodyStateT, HeadStateT]:
        """Combine initialized child states."""
        return self.State(body=body, head=head)

    def apply(
        self,
        x: InputT,
        state: Model.State[BodyStateT, HeadStateT],
    ) -> tuple[OutputT, Model.State[BodyStateT, HeadStateT]]:
        """Produce hidden features, then apply the configured head."""
        hidden, body = self.body.apply(x, state.body)
        output, head = self.head.apply(hidden, state.head)
        return output, self.State(body=body, head=head)


@dataclass
class NormModel[InputT, OutputT, NormStateT, ModelStateT]:
    """Normalize inputs before applying a model.

    Applying the component reads normalization statistics without updating
    them. :meth:`update` explicitly adds observations at the algorithm
    boundary while preserving model state.

    Attributes:
        norm: Input normalizer with an explicit update operation.
        model: Transform applied to normalized inputs.

    Public dataclasses:
        State: Normalization and model child states.

    Public methods:
        init: Combine initialized child states.
        apply: Normalize inputs and apply the model.
        update: Update normalization state from new inputs.
    """

    norm: Normalizer[InputT, NormStateT]
    model: Transform[InputT, OutputT, ModelStateT]

    @dataclass
    class State[NormDataT, ModelDataT]:
        """Normalization and model child-state tree."""

        norm: NormDataT
        model: ModelDataT

    def init(
        self,
        norm: NormStateT,
        model: ModelStateT,
    ) -> NormModel.State[NormStateT, ModelStateT]:
        """Combine initialized normalization and model states."""
        return self.State(norm=norm, model=model)

    def apply(
        self,
        x: InputT,
        state: NormModel.State[NormStateT, ModelStateT],
    ) -> tuple[OutputT, NormModel.State[NormStateT, ModelStateT]]:
        """Normalize inputs without updating statistics, then apply the model."""
        x, norm = self.norm.apply(x, state.norm)
        output, model = self.model.apply(x, state.model)
        return output, self.State(norm=norm, model=model)

    def update(
        self,
        x: InputT,
        state: NormModel.State[NormStateT, ModelStateT],
    ) -> NormModel.State[NormStateT, ModelStateT]:
        """Update normalization state while preserving model state."""
        return self.State(
            norm=self.norm.update(x, state.norm),
            model=state.model,
        )
