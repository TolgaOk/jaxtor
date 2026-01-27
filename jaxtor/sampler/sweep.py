"""Stochastic sweep sampler for tabular MDPs.

MC-protocol sweep over all (s,a) pairs using stochastic transitions.
Conditions each position's MDP initial distribution to ensure resets
return to the designated initial state.

Example:
    >>> from jaxtor.env import tabular
    >>> from jaxtor.sampler import mc, sweep
    >>> config = tabular.garnet.Config(state_size=10, action_size=4)
    >>> env = tabular.garnet.make(config)
    >>> mc_sampler = mc.Mc(max_episode_len=100, queue_size=10, env=env)
    >>> sweeper = sweep.Sweep(mc=mc_sampler)
    >>> env_state = env.init(key)
    >>> state = sweeper.init(key, env_state)
    >>> transition, state = sweeper.sample(act, state)
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from jaxdp.mdp import MDP

Transition = TypeVar("Transition")


class Env(Protocol):
    """Environment protocol for sweep sampler."""

    class State(Protocol):
        """Environment state with MDP access."""

        mdp: MDP


class MC(Protocol):
    """Markov chain sampler protocol."""

    class State(Protocol):
        """MC state with episode tracking and env access."""

        eps_idx: chex.Array
        env: Env.State

    class Metrics(Protocol):
        """Episode statistics."""

        avg_eps_rew: chex.Numeric
        avg_eps_len: chex.Numeric

    def init(self, key: chex.PRNGKey, env: Env.State) -> MC.State: ...

    def sample(
        self, act: chex.Array, state: MC.State
    ) -> tuple[chex.ArrayTree, MC.State]: ...

    def metrics(self, state: MC.State) -> tuple[Metrics, MC.State]: ...


def _condition_mdp_initial(mdp: MDP, init_dist: chex.Array) -> MDP:
    """Create an MDP with a modified initial distribution.

    Args:
        mdp: Original MDP.
        init_dist: New initial state distribution (typically one-hot).

    Returns:
        MDP with the new initial distribution, sharing other arrays.
    """
    return MDP(
        transition=mdp.transition,
        reward=mdp.reward,
        initial=init_dist,
        terminal=mdp.terminal,
        features=mdp.features,
        name=mdp.name,
        validate=False,
    )


@dataclass
class Sweep:
    """MC-protocol sweep over all (s,a) pairs with stochastic transitions.

    Creates A*S parallel Mc samplers, one for each (state, action) pair.
    Conditions each MDP's initial distribution to ensure resets return
    to the designated initial state.

    Flat batch ordering: position = a * S + s (action-major).

    Attributes:
        mc: MC instance for single-environment sampling.
    """

    mc: MC

    @property
    def env(self):
        """Access underlying environment for reset operations."""
        return self.mc.env

    def init(self, key: chex.PRNGKey, env_state: Env.State) -> MC.State:
        """Initialize A*S parallel Mc states with conditioned MDPs.

        Each position gets an MDP with initial distribution set to one-hot
        for its designated state. This ensures resets naturally return to
        the correct initial state without manual last_state manipulation.

        Args:
            key: Random key for initialization.
            env_state: Base environment state (template).

        Returns:
            Batched MC.State with shape (A*S, ...).
        """
        S, A = env_state.mdp.state_size, env_state.mdp.action_size

        state_indices = jnp.tile(jnp.arange(S), A)
        init_dists = jax.nn.one_hot(state_indices, S)

        def condition_env_state(init_dist: chex.Array) -> Env.State:
            new_mdp = _condition_mdp_initial(env_state.mdp, init_dist)
            return env_state.replace(mdp=new_mdp)  # type: ignore[attr-defined]

        conditioned_env_states = jax.vmap(condition_env_state)(init_dists)

        keys = jrd.split(key, A * S)
        return jax.vmap(self.mc.init)(keys, conditioned_env_states)

    def sample(
        self, act: chex.Array, state: MC.State
    ) -> tuple[Transition, MC.State]:
        """Sample from all (s,a) pairs.

        Uses init_action on first step of each episode. Resets naturally
        return to the correct initial state via conditioned mdp.initial.

        Args:
            act: Batched actions, shape (A*S,).
            state: Batched MC state.

        Returns:
            Batched transitions and updated state.
        """
        S, A = state.env.mdp.state_size, state.env.mdp.action_size
        init_action = jnp.arange(A * S) // S

        is_first = (state.eps_idx == 0).astype(act.dtype)
        actual_act = (1 - is_first) * act + is_first * init_action

        return jax.vmap(self.mc.sample)(actual_act, state)

    def metrics(self, state: MC.State) -> tuple[MC.Metrics, MC.State]:
        """Aggregate metrics from all (s,a) pairs.

        Args:
            state: Batched MC state.

        Returns:
            Aggregated scalar metrics and state with refreshed queues.
        """
        per_pos, new_state = jax.vmap(self.mc.metrics)(state)
        aggregated = jax.tree.map(jnp.nanmean, per_pos)
        return aggregated, new_state
