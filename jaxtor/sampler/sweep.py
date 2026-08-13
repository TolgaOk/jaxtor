"""Stochastic sweep sampler for tabular MDPs.

Sweep over all (s,a) pairs using stochastic transitions.

Example:
    >>> from jaxtor.env import tabular
    >>> from jaxtor.sampler import mc, sweep
    >>> config = tabular.garnet.Config(state_size=10, action_size=4)
    >>> env = tabular.garnet.make(config)
    >>> mc_sampler = mc.Mc(max_eps_len=100, env=env)
    >>> sweeper = sweep.Sweep(mc=mc_sampler)
    >>> env_state = env.init(key)
    >>> transition, mc_state = sweeper.sample(key, env_state)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass

from jaxtor.env.tabular import Mdp, TabularEnv


class Mc[McStateT, TransitionT](Protocol):
    """Markov-chain capability required by ``Sweep``."""

    def init(self, key: chex.PRNGKey, env: TabularEnv.State) -> McStateT: ...
    def sample(
        self, act: chex.Array, state: McStateT
    ) -> tuple[TransitionT, McStateT]: ...


@dataclass
class Sweep[McStateT, TransitionT]:
    """Sweep over all (s,a) pairs with stochastic transitions.

    Flat batch ordering: position = a * S + s (action-major).

    Attributes:
        mc: Mc instance for single-environment sampling.
    """

    mc: Mc[McStateT, TransitionT]

    def _condition_mdp_initial(self, mdp: Mdp, init_dist: jax.Array) -> Mdp:
        """Create an MDP with a modified initial distribution.

        Args:
            mdp: Original MDP.
            init_dist: New initial state distribution (typically one-hot).

        Returns:
            MDP with the new initial distribution, sharing other arrays.
        """
        return replace(mdp, initial=init_dist)

    def sample(
        self,
        key: chex.PRNGKey,
        env: TabularEnv.State,
    ) -> tuple[TransitionT, McStateT]:
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

        def condition_env_state(init_dist: jax.Array) -> TabularEnv.State:
            new_mdp = self._condition_mdp_initial(env.mdp, init_dist)
            return replace(env, mdp=new_mdp)

        conditioned_env_states = jax.vmap(condition_env_state)(init_dists)

        keys = jrd.split(key, A * S)
        mc_state = jax.vmap(self.mc.init)(keys, conditioned_env_states)
        init_action = jnp.arange(A * S) // S
        chex.assert_shape(init_action, (A * S,))
        return jax.vmap(self.mc.sample)(init_action, mc_state)
