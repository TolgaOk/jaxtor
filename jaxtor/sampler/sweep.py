"""Stochastic exhaustive sampling for finite Markov decision processes.

``Sweep`` conditions one tabular environment state on every state index, then
samples one designated action from every conditioned state::

    sweep = Sweep(mc=mc)
    transitions, states = sweep.sample(key, env_state)

The flat batch is action-major: ``position = action * state_size + state``.
"""

from __future__ import annotations

from typing import Protocol, Self

import chex
from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as jrd


class Mdp(Protocol):
    """Finite MDP surface consumed by ``Sweep``."""

    @property
    def state_size(self) -> int: ...
    @property
    def action_size(self) -> int: ...
    def replace(self, **changes: object) -> Self: ...


class EnvState(Protocol):
    """Tabular environment-state surface consumed by ``Sweep``."""

    @property
    def mdp(self) -> Mdp: ...
    def replace(self, **changes: object) -> Self: ...


class Mc[EnvS, McS, Transition](Protocol):
    """Markov-chain surface consumed by ``Sweep``."""

    def init(self, key: jax.Array, env: EnvS) -> McS: ...
    def sample(self, act: chex.Array, state: McS) -> tuple[Transition, McS]: ...


@dataclass
class Sweep[EnvS: EnvState, McS, Transition]:
    """Sample one stochastic transition from every state-action pair.

    Required protocols::

        env_state.mdp.state_size: int
        env_state.mdp.action_size: int
        env_state.mdp.replace(initial=...) -> mdp
        env_state.replace(mdp=...) -> env_state
        mc.init(key, env_state) -> mc_state
        mc.sample(action, mc_state) -> (transition, mc_state)

    Attributes:
        mc: Scalar Markov-chain sampler.

    Public methods:
        sample: Sample every state-action pair in one action-major batch.
    """

    mc: Mc[EnvS, McS, Transition]

    def sample(
        self,
        key: jax.Array,
        env: EnvS,
    ) -> tuple[Transition, McS]:
        """Sample every state-action pair in one action-major batch."""
        state_size = env.mdp.state_size
        action_size = env.mdp.action_size
        batch_size = action_size * state_size

        state_indices = jnp.tile(jnp.arange(state_size), action_size)
        initials = jax.nn.one_hot(state_indices, state_size)
        chex.assert_shape(state_indices, (batch_size,))
        chex.assert_shape(initials, (batch_size, state_size))

        def condition(initial: jax.Array) -> EnvS:
            return env.replace(mdp=env.mdp.replace(initial=initial))

        envs = jax.vmap(condition)(initials)
        states = jax.vmap(self.mc.init)(jrd.split(key, batch_size), envs)
        actions = jnp.arange(batch_size) // state_size
        chex.assert_shape(actions, (batch_size,))
        return jax.vmap(self.mc.sample)(actions, states)
