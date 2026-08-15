"""Composed policy, value-policy, and normalized-advantage agents.

An agent joins a shared body and semantic heads while state remains explicit::

    agent = Pi(body=body, pi=pi)
    state = agent.init(key, body=body_state, pi=pi_state)
    pred, applied_state = agent.apply(obs, state)
    act, acted_state = agent.act(obs, state)

``apply`` returns all predictions. ``act`` evaluates only action dependencies.
For :class:`Naf`, the prediction exposes a state-bound Q-function and behavior
policy::

    pred, state = naf.apply(obs, state)
    q = pred.qfn.evaluate(act)
    logp = pred.pi.evaluate(act).logp
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.agent.dist import Normal


class Distribution[Act, Eval](Protocol):
    """Action-distribution capability consumed by composed agents."""

    def sample(self, key: jax.Array) -> Act: ...
    def evaluate(self, act: Act) -> Eval: ...
    def mode(self) -> Act: ...


class Transform[In, Out, S](Protocol):
    """Stateful transformation capability consumed by composed agents."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


@dataclass
class Pi[Obs, Feat, Act, Eval, BodyS, PiS]:
    """Compose a body and policy head into an acting agent.

    Attributes:
        body: Transform mapping observations to common features.
        pi: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        Pred: Policy prediction.

    Public methods:
        init: Combine initialized children and the sampling key.
        apply: Produce a policy prediction.
        act: Select an action from the policy.
    """

    body: Transform[Obs, Feat, BodyS]
    pi: Transform[Feat, Distribution[Act, Eval], PiS]
    deterministic: bool = False

    @dataclass
    class State[BodyData, PiData]:
        """Policy child-state tree and sampling key."""

        body: BodyData
        pi: PiData
        key: jax.Array

    @dataclass
    class Pred[ActData, EvalData]:
        """Policy prediction aligned with the observation leading axes."""

        pi: Distribution[ActData, EvalData]

    def init(
        self,
        key: jax.Array,
        body: BodyS,
        pi: PiS,
    ) -> Pi.State[BodyS, PiS]:
        """Combine initialized children and the sampling key."""
        return self.State(body=body, pi=pi, key=key)

    def apply(
        self,
        obs: Obs,
        state: Pi.State[BodyS, PiS],
    ) -> tuple[Pi.Pred[Act, Eval], Pi.State[BodyS, PiS]]:
        """Produce a policy prediction without selecting an action."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        return self.Pred(pi=dist), self.State(body=body, pi=pi, key=state.key)

    def act(
        self,
        obs: Obs,
        state: Pi.State[BodyS, PiS],
    ) -> tuple[Act, Pi.State[BodyS, PiS]]:
        """Select an action from the policy distribution."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.pi.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, self.State(body=body, pi=pi, key=key)


@dataclass
class VPi[Obs, Feat, Act, Eval, BodyS, ValS, PiS]:
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

    body: Transform[Obs, Feat, BodyS]
    v: Transform[Feat, jax.Array, ValS]
    pi: Transform[Feat, Distribution[Act, Eval], PiS]
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
        obs: Obs,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[VPi.Pred[Act, Eval], VPi.State[BodyS, ValS, PiS]]:
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
        obs: Obs,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[Act, VPi.State[BodyS, ValS, PiS]]:
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
class VQPi[Obs, Feat, Act, Eval, BodyS, ValS, QS, PiS]:
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

    body: Transform[Obs, Feat, BodyS]
    v: Transform[Feat, jax.Array, ValS]
    q: Transform[Feat, jax.Array, QS]
    pi: Transform[Feat, Distribution[Act, Eval], PiS]
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
        obs: Obs,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[VQPi.Pred[Act, Eval], VQPi.State[BodyS, ValS, QS, PiS]]:
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
        obs: Obs,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[Act, VQPi.State[BodyS, ValS, QS, PiS]]:
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


@dataclass
class Naf[Obs, Feat, BodyS, ValS, LocS, PS]:
    """Compose a diagonal normalized advantage function agent.

    The agent parameterizes

    .. code-block:: text

        mu(s) = tanh(loc(s))
        d(s) = softplus(raw_p(s)) + eps
        P(s) = diag(d(s))
        A(s, a) = -1/2 (a - mu(s))^T P(s) (a - mu(s))
        Q(s, a) = V(s) + A(s, a)

    The behavior policy reuses ``P(s)`` as precision:

    .. code-block:: text

        Sigma(s) = scale^2 P(s)^-1
        pi(. | s) = Normal(mu(s), Sigma(s))

    Attributes:
        act_size: Size of the continuous action vector.
        body: Transform mapping observations to common features.
        v: Transform producing ``V(s)``.
        loc: Transform producing unconstrained action locations.
        p: Transform producing unconstrained diagonal precisions.
        eps: Positive floor applied to transformed precisions.
        scale: Positive exploration standard-deviation scale.
        deterministic: Whether acting returns ``mu(s)``.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        Qfn: State-bound function mapping actions to Q-values.
        Pred: Value, Q-function, and behavior-policy prediction.

    Public methods:
        init: Combine initialized children and the sampling key.
        apply: Produce the complete NAF prediction.
        act: Select an action without evaluating ``V(s)``.
    """

    act_size: int
    body: Transform[Obs, Feat, BodyS]
    v: Transform[Feat, jax.Array, ValS]
    loc: Transform[Feat, jax.Array, LocS]
    p: Transform[Feat, jax.Array, PS]
    eps: float = 1.0
    scale: float = 0.1
    deterministic: bool = False

    @dataclass
    class State[BodyData, ValData, LocData, PData]:
        """Body, value, location, precision, and sampling state."""

        body: BodyData
        v: ValData
        loc: LocData
        p: PData
        key: jax.Array

    @dataclass
    class Qfn:
        """State-bound normalized advantage Q-function.

        Attributes:
            v: State values shaped ``[...]``.
            loc: Actions maximizing Q, shaped ``[..., A]``.
            p: Positive-definite precision matrices shaped ``[..., A, A]``.

        Public methods:
            evaluate: Evaluate ``Q(s, a)`` for supplied actions.
        """

        v: jax.Array
        loc: jax.Array
        p: jax.Array

        def evaluate(self, act: jax.Array) -> jax.Array:
            """Evaluate the state-bound Q-function at ``act``."""
            chex.assert_equal_shape([act, self.loc])
            chex.assert_shape(self.v, self.loc.shape[:-1])
            delta = act - self.loc
            q = self.v - 0.5 * jnp.einsum(
                "...i,...ij,...j->...",
                delta,
                self.p,
                delta,
            )
            chex.assert_equal_shape([self.v, q])
            return q

    @dataclass
    class Pred:
        """Complete NAF prediction aligned by leading axes.

        Attributes:
            v: State values shaped ``[...]``.
            qfn: State-bound function ``a -> Q(s, a)``.
            pi: Normal behavior policy with event shape ``[A]``.
        """

        v: jax.Array
        qfn: Naf.Qfn
        pi: Normal

    def __post_init__(self) -> None:
        """Validate the action size, precision floor, and policy scale."""
        if self.act_size < 1:
            raise ValueError("act_size must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def init(
        self,
        key: jax.Array,
        body: BodyS,
        v: ValS,
        loc: LocS,
        p: PS,
    ) -> Naf.State[BodyS, ValS, LocS, PS]:
        """Combine initialized children and the sampling key."""
        return self.State(body=body, v=v, loc=loc, p=p, key=key)

    def _policy(
        self,
        features: Feat,
        state: Naf.State[BodyS, ValS, LocS, PS],
    ) -> tuple[Normal, jax.Array, Naf.State[BodyS, ValS, LocS, PS]]:
        """Produce the behavior policy, precision, and advanced child state."""
        loc, loc_state = self.loc.apply(features, state.loc)
        raw_p, p_state = self.p.apply(features, state.p)
        shape = (*loc.shape[:-1], self.act_size)
        chex.assert_shape(loc, shape)
        chex.assert_shape(raw_p, shape)
        diagonal = jax.nn.softplus(raw_p) + self.eps
        eye = jnp.eye(self.act_size, dtype=loc.dtype)
        precision = jnp.einsum("...i,ij->...ij", diagonal, eye)
        return (
            Normal(
                loc=jnp.tanh(loc),
                cov=jnp.einsum("...i,ij->...ij", self.scale**2 / diagonal, eye),
            ),
            precision,
            replace(state, loc=loc_state, p=p_state),
        )

    def apply(
        self,
        obs: Obs,
        state: Naf.State[BodyS, ValS, LocS, PS],
    ) -> tuple[Naf.Pred, Naf.State[BodyS, ValS, LocS, PS]]:
        """Produce value, state-bound Q-function, and policy predictions."""
        features, body = self.body.apply(obs, state.body)
        state = replace(state, body=body)
        value, v = self.v.apply(features, state.v)
        policy, precision, state = self._policy(features, state)
        chex.assert_shape(value, policy.loc.shape[:-1])
        return self.Pred(
            v=value,
            qfn=self.Qfn(v=value, loc=policy.loc, p=precision),
            pi=policy,
        ), replace(
            state,
            v=v,
        )

    def act(
        self,
        obs: Obs,
        state: Naf.State[BodyS, ValS, LocS, PS],
    ) -> tuple[jax.Array, Naf.State[BodyS, ValS, LocS, PS]]:
        """Select an action without evaluating the value transform."""
        features, body = self.body.apply(obs, state.body)
        state = replace(state, body=body)
        policy, _, state = self._policy(features, state)
        if self.deterministic:
            return policy.mode(), state
        key, sample_key = jax.random.split(state.key)
        return policy.sample(sample_key), replace(state, key=key)
