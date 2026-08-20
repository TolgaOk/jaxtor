"""Structural policy, value-policy, and quadratic agent components.

Each component stores child transforms as configuration and threads their
dynamic states through a nested ``State`` pytree::

    agent = Pi(body=body, policy=pi)
    state = agent.init(key, body=body_state, pi=pi_state)
    pi, applied_state = agent.pi(obs, state)
    act, acted_state = agent.act(obs, state)

Value-policy agents expose separate value and joint value-policy endpoints::

    value_policy, state = agent.vpi(obs, state)
    value, state = agent.v(obs, state)

Quadratic agents expose value and action-value endpoints::

    q, state = agent.q(obs, act, state)
    value, state = agent.v(obs, state)

``Ou`` adds temporal exploration by wrapping action selection::

    explorer = Ou(agent=deterministic_agent, sigma=0.1)
    state = explorer.init(key, noise, agent_state)
    act, state = explorer.act(obs, state)
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Distribution[Act, Eval](Protocol):
    """Action distribution consumed by policy agents."""

    def sample(self, key: jax.Array) -> Act: ...
    def evaluate(self, act: Act) -> Eval: ...
    def mode(self) -> Act: ...


class Transform[In, Out, S](Protocol):
    """Stateful transform consumed by composed agents."""

    def apply(self, x: In, state: S, /) -> tuple[Out, S]: ...


class Agent[Obs, S](Protocol):
    """Action selector wrapped by :class:`Ou`."""

    def act(self, obs: Obs, state: S) -> tuple[jax.Array, S]: ...


@dataclass
class Ou[Obs, S]:
    """Add bounded Ornstein–Uhlenbeck noise to an agent's actions.

    The component advances one independent process per action entry:

    .. code-block:: text

        noise_t = noise_{t-1} + theta (0 - noise_{t-1}) dt
                  + sigma sqrt(dt) epsilon_t
        action_t = clip(agent_action_t + noise_t, low, high)

    The noise is added to the action returned by ``agent``. A deterministic
    child agent makes this process the sole source of action-selection noise.
    Supplying the initial noise explicitly keeps its batch shape visible and
    supports scalar, vectorized, or nested-vectorized agents. The process does
    not infer episode boundaries and continues until its state is reinitialized.

    Required protocols::

        agent.act(obs, agent_state) -> (action: jax.Array, agent_state)

    Attributes:
        agent: Child agent producing the unperturbed action.
        theta: Rate at which noise returns to zero.
        sigma: Standard deviation of each Gaussian innovation.
        dt: Process time increment per action.
        low: Lower action bound.
        high: Upper action bound.

    Public dataclasses:
        State: Child-agent, random-key, and noise-process states.

    Public methods:
        init: Combine initialized child and noise states.
        act: Select an action and advance the noise process.
    """

    agent: Agent[Obs, S]
    theta: float = 0.15
    sigma: float = 0.1
    dt: float = 1.0
    low: float = -1.0
    high: float = 1.0

    @dataclass
    class State[AgentData]:
        """Child-agent state and Ornstein-Uhlenbeck process state.

        Attributes:
            agent: Child-agent state.
            key: Random key used for the next noise innovation.
            noise: Current action-shaped process value.
        """

        agent: AgentData
        key: jax.Array
        noise: jax.Array

    def __post_init__(self) -> None:
        """Validate process and action-bound parameters."""
        if self.theta < 0:
            raise ValueError("theta must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.low >= self.high:
            raise ValueError("low must be less than high")

    def init(
        self,
        key: jax.Array,
        noise: jax.Array,
        agent: S,
    ) -> Ou.State[S]:
        """Combine initialized child and noise-process states."""
        return self.State(agent=agent, key=key, noise=noise)

    def act(
        self,
        obs: Obs,
        state: Ou.State[S],
    ) -> tuple[jax.Array, Ou.State[S]]:
        """Select an action and advance its correlated perturbation."""
        act, agent = self.agent.act(obs, state.agent)
        chex.assert_equal_shape([act, state.noise])
        key, sample_key = jax.random.split(state.key)
        noise = (
            state.noise
            - self.theta * state.noise * self.dt
            + self.sigma
            * math.sqrt(self.dt)
            * jax.random.normal(sample_key, act.shape, dtype=act.dtype)
        )
        return jnp.clip(act + noise, self.low, self.high), replace(
            state,
            agent=agent,
            key=key,
            noise=noise,
        )


@dataclass
class Pi[Obs, Feat, Act, Eval, BodyS, PiS]:
    """Compose a body and policy head into an acting agent.

    Required protocols::

        body.apply(obs, body_state) -> (features, body_state)
        policy.apply(features, policy_state) -> (distribution, policy_state)
        distribution.sample(key) -> action
        distribution.evaluate(action) -> evaluation
        distribution.mode() -> action

    Attributes:
        body: Transform mapping observations to common features.
        policy: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.

    Public methods:
        init: Combine initialized children and the sampling key.
        pi: Produce a policy distribution.
        act: Select an action from the policy.
    """

    body: Transform[Obs, Feat, BodyS]
    policy: Transform[Feat, Distribution[Act, Eval], PiS]
    deterministic: bool = False

    @dataclass
    class State[BodyData, PiData]:
        """Policy child-state tree and sampling key.

        Attributes:
            body: Body-transform state.
            pi: Policy-head state.
            key: Action-sampling key.
        """

        body: BodyData
        pi: PiData
        key: jax.Array

    def init(
        self,
        key: jax.Array,
        body: BodyS,
        pi: PiS,
    ) -> Pi.State[BodyS, PiS]:
        """Combine initialized children and the sampling key."""
        return self.State(body=body, pi=pi, key=key)

    def pi(
        self,
        obs: Obs,
        state: Pi.State[BodyS, PiS],
    ) -> tuple[Distribution[Act, Eval], Pi.State[BodyS, PiS]]:
        """Produce a policy distribution without selecting an action."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.policy.apply(features, state.pi)
        return dist, replace(state, body=body, pi=pi)

    def act(
        self,
        obs: Obs,
        state: Pi.State[BodyS, PiS],
    ) -> tuple[Act, Pi.State[BodyS, PiS]]:
        """Select an action from the policy distribution."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.policy.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, replace(state, body=body, pi=pi, key=key)


@dataclass
class VPi[Obs, Feat, Act, Eval, BodyS, ValS, PiS]:
    """Compose a body, value head, and policy head into an acting agent.

    Required protocols::

        body.apply(obs, body_state) -> (features, body_state)
        value.apply(features, value_state) -> (value, value_state)
        policy.apply(features, policy_state) -> (distribution, policy_state)
        distribution.sample(key) -> action
        distribution.evaluate(action) -> evaluation
        distribution.mode() -> action

    Attributes:
        body: Transform mapping observations to common features.
        value: Transform producing ``V(s)``.
        policy: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        ValuePolicy: Joint value and policy result.

    Public methods:
        init: Combine initialized children and the selection key.
        v: Produce state values without evaluating the policy.
        vpi: Produce state values and policies from shared features.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[Obs, Feat, BodyS]
    value: Transform[Feat, jax.Array, ValS]
    policy: Transform[Feat, Distribution[Act, Eval], PiS]
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
    class ValuePolicy[ActData, EvalData]:
        """State values and policies aligned by leading axes.

        Attributes:
            v: State values.
            pi: Policy distributions.
        """

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

    def v(
        self,
        obs: Obs,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[jax.Array, VPi.State[BodyS, ValS, PiS]]:
        """Produce state values without evaluating the policy head."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.value.apply(features, state.v)
        return value, replace(state, body=body, v=v)

    def vpi(
        self,
        obs: Obs,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[VPi.ValuePolicy[Act, Eval], VPi.State[BodyS, ValS, PiS]]:
        """Produce state values and policies from one shared body application."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.value.apply(features, state.v)
        dist, pi = self.policy.apply(features, state.pi)
        return (
            self.ValuePolicy(v=value, pi=dist),
            replace(state, body=body, v=v, pi=pi),
        )

    def act(
        self,
        obs: Obs,
        state: VPi.State[BodyS, ValS, PiS],
    ) -> tuple[Act, VPi.State[BodyS, ValS, PiS]]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.policy.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, replace(
            state,
            body=body,
            pi=pi,
            key=key,
        )


@dataclass
class VQPi[Obs, Feat, Act, Eval, BodyS, ValS, QS, PiS]:
    """Compose value, action-value, and policy components into an agent.

    Required protocols::

        body.apply(obs, body_state) -> (features, body_state)
        value.apply(features, value_state) -> (value, value_state)
        q_values.apply(features, q_state) -> (q_values, q_state)
        policy.apply(features, policy_state) -> (distribution, policy_state)
        distribution.sample(key) -> action
        distribution.evaluate(action) -> evaluation
        distribution.mode() -> action

    Attributes:
        body: Transform mapping observations to common features.
        value: Transform producing ``V(s)``.
        q_values: Transform producing ``Q(s, .)``.
        policy: Transform producing a policy distribution.
        deterministic: Whether acting uses the distribution mode instead of a
            stochastic sample.

    Public dataclasses:
        State: Complete child-state tree and sampling key.
        ValueQPolicy: Joint value, action-value, and policy result.

    Public methods:
        init: Combine initialized children and the selection key.
        vqpi: Produce value, action-value, and policy results.
        act: Select only the action required by a minimal sampler.
    """

    body: Transform[Obs, Feat, BodyS]
    value: Transform[Feat, jax.Array, ValS]
    q_values: Transform[Feat, jax.Array, QS]
    policy: Transform[Feat, Distribution[Act, Eval], PiS]
    deterministic: bool = False

    @dataclass
    class State[BodyData, ValData, QData, PiData]:
        """Value-action-values-policy child-state tree.

        Attributes:
            body: Body-transform state.
            v: Value-head state.
            q: Action-value-head state.
            pi: Policy-head state.
            key: Action-sampling key.
        """

        body: BodyData
        v: ValData
        q: QData
        pi: PiData
        key: jax.Array

    @dataclass
    class ValueQPolicy[ActData, EvalData]:
        """State values, action values, and policy distributions.

        Attributes:
            v: State values.
            q: Finite-action value vectors.
            pi: Policy distributions.
        """

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

    def vqpi(
        self,
        obs: Obs,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[
        VQPi.ValueQPolicy[Act, Eval],
        VQPi.State[BodyS, ValS, QS, PiS],
    ]:
        """Produce value, action values, and policy from shared features."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.value.apply(features, state.v)
        q, q_state = self.q_values.apply(features, state.q)
        dist, pi = self.policy.apply(features, state.pi)
        return (
            self.ValueQPolicy(v=value, q=q, pi=dist),
            replace(
                state,
                body=body,
                v=v,
                q=q_state,
                pi=pi,
            ),
        )

    def act(
        self,
        obs: Obs,
        state: VQPi.State[BodyS, ValS, QS, PiS],
    ) -> tuple[Act, VQPi.State[BodyS, ValS, QS, PiS]]:
        """Select only the action required by a minimal sampler."""
        features, body = self.body.apply(obs, state.body)
        dist, pi = self.policy.apply(features, state.pi)
        if self.deterministic:
            act, key = dist.mode(), state.key
        else:
            key, sample_key = jax.random.split(state.key)
            act = dist.sample(sample_key)
        return act, replace(
            state,
            body=body,
            pi=pi,
            key=key,
        )


@dataclass
class Quadratic[Obs, Feat, BodyS, ValS, LocS, PS]:
    """Compose a diagonal quadratic action-value agent.

    The agent parameterizes

    .. code-block:: text

        mu(s) = loc(s)
        d(s) = softplus(raw_p(s)) + eps
        P(s) = diag(d(s))
        A(s, a) = -1/2 (a - mu(s))^T P(s) (a - mu(s))
        Q(s, a) = V(s) + A(s, a)

    Required protocols::

        body.apply(obs, body_state) -> (features, body_state)
        value.apply(features, value_state) -> (value, value_state)
        loc.apply(features, loc_state) -> (location, loc_state)
        p.apply(features, p_state) -> (raw_precision, p_state)

    Attributes:
        act_size: Size of the continuous action vector.
        body: Transform mapping observations to common features.
        value: Transform producing ``V(s)``.
        loc: Transform producing action locations.
        p: Transform producing unconstrained diagonal precisions.
        eps: Positive floor applied to transformed precisions.

    Public dataclasses:
        State: Complete child-state tree.

    Public methods:
        init: Combine initialized children.
        q: Evaluate ``Q(s, a)``.
        v: Evaluate ``V(s)`` without evaluating action-dependent heads.
        act: Select the maximizing action without evaluating ``V(s)`` or ``P(s)``.
    """

    act_size: int
    body: Transform[Obs, Feat, BodyS]
    value: Transform[Feat, jax.Array, ValS]
    loc: Transform[Feat, jax.Array, LocS]
    p: Transform[Feat, jax.Array, PS]
    eps: float = 1.0

    @dataclass
    class State[BodyData, ValData, LocData, PData]:
        """Body, value, location, and precision child states.

        Attributes:
            body: Body-transform state.
            v: Value-head state.
            loc: Location-head state.
            p: Precision-head state.
        """

        body: BodyData
        v: ValData
        loc: LocData
        p: PData

    def __post_init__(self) -> None:
        """Validate the action size and precision floor."""
        if self.act_size < 1:
            raise ValueError("act_size must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def init(
        self,
        body: BodyS,
        v: ValS,
        loc: LocS,
        p: PS,
    ) -> Quadratic.State[BodyS, ValS, LocS, PS]:
        """Combine initialized child states."""
        return self.State(body=body, v=v, loc=loc, p=p)

    def q(
        self,
        obs: Obs,
        act: jax.Array,
        state: Quadratic.State[BodyS, ValS, LocS, PS],
    ) -> tuple[jax.Array, Quadratic.State[BodyS, ValS, LocS, PS]]:
        """Evaluate ``Q(s, a)`` and advance every required child state."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.value.apply(features, state.v)
        loc, loc_state = self.loc.apply(features, state.loc)
        raw_p, p_state = self.p.apply(features, state.p)
        chex.assert_shape(act, (*act.shape[:-1], self.act_size))
        chex.assert_equal_shape([act, loc, raw_p])
        chex.assert_shape(value, loc.shape[:-1])
        diagonal = jax.nn.softplus(raw_p) + self.eps
        q = value - 0.5 * jnp.sum(diagonal * (act - loc) ** 2, axis=-1)
        chex.assert_equal_shape([value, q])
        return q, replace(
            state,
            body=body,
            v=v,
            loc=loc_state,
            p=p_state,
        )

    def v(
        self,
        obs: Obs,
        state: Quadratic.State[BodyS, ValS, LocS, PS],
    ) -> tuple[jax.Array, Quadratic.State[BodyS, ValS, LocS, PS]]:
        """Evaluate ``V(s)`` without evaluating location or precision heads."""
        features, body = self.body.apply(obs, state.body)
        value, v = self.value.apply(features, state.v)
        return value, replace(state, body=body, v=v)

    def act(
        self,
        obs: Obs,
        state: Quadratic.State[BodyS, ValS, LocS, PS],
    ) -> tuple[jax.Array, Quadratic.State[BodyS, ValS, LocS, PS]]:
        """Select the maximizing action without evaluating value or precision."""
        features, body = self.body.apply(obs, state.body)
        loc, loc_state = self.loc.apply(features, state.loc)
        chex.assert_shape(loc, (*loc.shape[:-1], self.act_size))
        return loc, replace(state, body=body, loc=loc_state)
