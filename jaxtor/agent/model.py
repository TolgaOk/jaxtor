"""Generic trainable transforms and state-tree composition."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import jax
from chex import dataclass


@dataclass
class Param[Value]:
    """Mark one pytree as trainable without changing its JAX behavior.

    Attributes:
        value: Trainable array or nested array pytree.
    """

    value: Value


@dataclass
class Partition[Params, Frozen]:
    """Trainable and frozen views of one component state.

    Attributes:
        params: State-shaped tree containing :class:`Param` leaves.
        frozen: Complementary state-shaped tree containing all other leaves.
    """

    params: Params
    frozen: Frozen


def _is_param(value: Any) -> bool:
    """Identify trainable markers while partitioning a state tree."""
    return isinstance(value, Param)


def _is_partition_leaf(value: Any) -> bool:
    """Treat parameters and empty complementary leaves atomically."""
    return value is None or isinstance(value, Param)


def partition[S](state: S) -> Partition[S, S]:
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


def combine[S](params: S, frozen: S) -> S:
    """Reconstruct a component state from complementary partition trees."""
    return jax.tree.map(
        lambda param, fixed: fixed if param is None else param,
        params,
        frozen,
        is_leaf=_is_partition_leaf,
    )


@runtime_checkable
class Function[In, Out](Protocol):
    """Callable capability consumed by :class:`Module`."""

    def __call__(self, x: In, /) -> Out: ...


@dataclass
class Module[Out]:
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

    static: Function[jax.Array, Out]
    in_ndim: int = 1

    @dataclass
    class State[Result]:
        """Dynamic parameter partition of a callable module."""

        params: Param[Function[jax.Array, Result]]

    def __post_init__(self) -> None:
        """Validate the rank consumed by one callable invocation."""
        if self.in_ndim < 0:
            raise ValueError("in_ndim must be nonnegative")

    def init(
        self,
        params: Function[jax.Array, Out],
    ) -> Module.State[Out]:
        """Store an initialized parameter partition."""
        return self.State(params=Param(value=params))

    def _combine(
        self,
        params: Function[jax.Array, Out],
    ) -> Function[jax.Array, Out]:
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
        state: Module.State[Out],
    ) -> tuple[Out, Module.State[Out]]:
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


class Transform[In, Out, S](Protocol):
    """Stateful transformation capability consumed by :class:`Model`."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


@dataclass
class Model[In, Feat, Pred, BodyS, HeadS]:
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

    body: Transform[In, Feat, BodyS]
    head: Transform[Feat, Pred, HeadS]

    @dataclass
    class State[BodyData, HeadData]:
        """Shared-body model child-state tree."""

        body: BodyData
        head: HeadData

    def init(
        self,
        body: BodyS,
        head: HeadS,
    ) -> Model.State[BodyS, HeadS]:
        """Combine initialized child states."""
        return self.State(body=body, head=head)

    def apply(
        self,
        x: In,
        state: Model.State[BodyS, HeadS],
    ) -> tuple[Pred, Model.State[BodyS, HeadS]]:
        """Produce hidden features, then apply the configured head."""
        hidden, body = self.body.apply(x, state.body)
        output, head = self.head.apply(hidden, state.head)
        return output, self.State(body=body, head=head)


class Normalizer[Value, S](Protocol):
    """Normalization capability consumed by :class:`NormModel`."""

    def apply(self, x: Value, state: S, /) -> tuple[Value, S]: ...

    def update(self, x: Value, state: S, /) -> S: ...


@dataclass
class NormModel[In, Pred, NormS, ModelS]:
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

    norm: Normalizer[In, NormS]
    model: Transform[In, Pred, ModelS]

    @dataclass
    class State[NormData, ModelData]:
        """Normalization and model child-state tree."""

        norm: NormData
        model: ModelData

    def init(
        self,
        norm: NormS,
        model: ModelS,
    ) -> NormModel.State[NormS, ModelS]:
        """Combine initialized normalization and model states."""
        return self.State(norm=norm, model=model)

    def apply(
        self,
        x: In,
        state: NormModel.State[NormS, ModelS],
    ) -> tuple[Pred, NormModel.State[NormS, ModelS]]:
        """Normalize inputs without updating statistics, then apply the model."""
        x, norm = self.norm.apply(x, state.norm)
        output, model = self.model.apply(x, state.model)
        return output, self.State(norm=norm, model=model)

    def update(
        self,
        x: In,
        state: NormModel.State[NormS, ModelS],
    ) -> NormModel.State[NormS, ModelS]:
        """Update normalization state while preserving model state."""
        return self.State(
            norm=self.norm.update(x, state.norm),
            model=state.model,
        )
