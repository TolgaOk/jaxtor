"""Stochastic sweep sampler for tabular MDPs.

Sweep over all (s,a) pairs using stochastic transitions.

Example:
    >>> from jaxtor.env import tabular
    >>> from jaxtor.sampler import mc, sweep
    >>> config = tabular.garnet.Config(state_size=10, action_size=4)
    >>> env = tabular.garnet.make(config)
    >>> mc_sampler = mc.Mc(max_episode_len=100, queue_size=10, env=env)
    >>> sweeper = sweep.Sweep(mc=mc_sampler)
    >>> env_state = env.init(key)
    >>> transition, mc_state = sweeper.sample(key, env_state)
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
    class State(Protocol):
        mdp: MDP


class Mc(Protocol):
    class State(Protocol): ...

    def init(self, key: chex.PRNGKey, env: Env.State) -> Mc.State: ...
    def sample(
        self, act: chex.Array, state: Mc.State
    ) -> tuple[Transition, Mc.State]: ...


@dataclass
class Sweep:
    """Sweep over all (s,a) pairs with stochastic transitions.

    Flat batch ordering: position = a * S + s (action-major).

    Attributes:
        mc: Mc instance for single-environment sampling.
    """

    mc: Mc

    def _condition_mdp_initial(self, mdp: MDP, init_dist: chex.Array) -> MDP:
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

    def sample(self, key: chex.PRNGKey, env: Env.State) -> tuple[Transition, Mc.State]:
        """Sample one transition from each (s,a) pair.

        Initializes A*S parallel MC states with conditioned initial distributions,
        then samples with the designated initial action for each position.

        Args:
            key: Random key for initialization and sampling.
            env: Environment state (template).

        Returns:
            Batched transitions and MC states with shape (A*S, ...).
        """
        S, A = env.mdp.state_size, env.mdp.action_size

        state_indices = jnp.tile(jnp.arange(S), A)
        init_dists = jax.nn.one_hot(state_indices, S)
        chex.assert_shape(state_indices, (A * S,))
        chex.assert_shape(init_dists, (A * S, S))

        def condition_env_state(init_dist: chex.Array) -> Env.State:
            new_mdp = self._condition_mdp_initial(env.mdp, init_dist)
            return env.replace(mdp=new_mdp)  # type: ignore[attr-defined]

        conditioned_env_states = jax.vmap(condition_env_state)(init_dists)

        keys = jrd.split(key, A * S)
        mc_state = jax.vmap(self.mc.init)(keys, conditioned_env_states)
        init_action = jnp.arange(A * S) // S
        chex.assert_shape(init_action, (A * S,))
        return jax.vmap(self.mc.sample)(init_action, mc_state)
