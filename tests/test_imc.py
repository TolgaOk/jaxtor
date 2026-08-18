"""Tests for minimal agent-induced Markov-chain sampling."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.sampler import Imc, Mc, VecMc


@dataclass
class CounterEnv:
    """Deterministic environment terminating after two actions."""

    @dataclass
    class State:
        """Current counter value."""

        count: jax.Array

    @dataclass
    class Step:
        """Environment step result."""

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def init(self) -> State:
        """Return an uninitialized counter state."""
        return self.State(count=jnp.array(0, dtype=jnp.int32))

    def reset(
        self,
        key: jax.Array,
        state: State,
    ) -> tuple[jax.Array, State]:
        """Reset the counter to zero."""
        del key, state
        count = jnp.array(0, dtype=jnp.int32)
        return count, self.State(count=count)

    def step(
        self,
        key: jax.Array,
        act: jax.Array,
        state: State,
    ) -> tuple[Step, State]:
        """Increment the counter and terminate at two."""
        del key
        count = state.count + 1
        return self.Step(
            nobs=count,
            rew=act.astype(jnp.float32),
            term=count == 2,
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class CountingAgent:
    """Select an observation-dependent action and count selections."""

    @dataclass
    class State:
        """Action offset and number of selected actions."""

        offset: jax.Array
        actions: jax.Array

    def act(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[jax.Array, State]:
        """Return one raw action and advance the selection count."""
        act = obs.astype(jnp.int32) + state.offset
        return act, replace(state, actions=state.actions + 1)


def make_imc_state(
    key: jax.Array,
    *,
    offset: int = 1,
) -> tuple[Imc, Imc.State]:
    """Build a scalar minimal IMC and its state."""
    env = CounterEnv()
    mc = Mc(max_eps_len=10, env=env)
    imc = Imc(agent=CountingAgent(), mc=mc)
    return imc, imc.init(
        mc.init(key, env.init()),
        CountingAgent.State(
            offset=jnp.array(offset, dtype=jnp.int32),
            actions=jnp.array(0, dtype=jnp.int32),
        ),
    )


def test_sample_returns_mc_transition_directly():
    """The minimal IMC adds no wrapper around the MC transition."""
    imc, state = make_imc_state(jax.random.key(0))

    transition, state = imc.sample(state)

    assert transition.obs == 0
    assert transition.act == 1
    assert transition.nobs == 1
    assert state.agent.actions == 1


def test_boundary_is_owned_only_by_mc():
    """IMC delegates terminal reset handling without caching agent data."""
    imc, state = make_imc_state(jax.random.key(0))
    _, state = imc.sample(state)

    transition, state = imc.sample(state)

    assert transition.term
    assert transition.nobs == 2
    assert imc.mc.observe(state.mc) == 0
    assert state.agent.actions == 2


def test_changed_agent_state_affects_the_next_action_immediately():
    """Replacing agent state cannot leave a stale action cache."""
    imc, state = make_imc_state(jax.random.key(0))
    state = replace(state, agent=replace(state.agent, offset=jnp.array(7)))

    transition, _ = imc.sample(state)

    assert transition.act == 7


def test_sample_is_jittable_and_tree_compatible():
    """Minimal IMC data remains an ordinary JAX pytree under JIT."""
    imc, state = make_imc_state(jax.random.key(0))

    transition, state = jax.jit(imc.sample)(state)
    copied = jax.tree.map(lambda leaf: leaf, state)

    assert transition.act == 1
    assert jax.tree.structure(copied) == jax.tree.structure(state)
    assert len(jax.tree.leaves(state)) > 0


def test_vec_mc_preserves_action_batch_axes():
    """A batched action agent composes directly with ``VecMc``."""
    n_envs = 3
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_eps_len=10, env=env))
    imc = Imc(agent=CountingAgent(), mc=mc)
    keys = jax.random.split(jax.random.key(0), n_envs)
    state = imc.init(
        mc.init(keys, jax.vmap(lambda _: env.init())(keys)),
        CountingAgent.State(offset=jnp.array(1), actions=jnp.array(0)),
    )

    transition, state = jax.jit(imc.sample)(state)

    chex.assert_shape(transition.act, (n_envs,))
    chex.assert_shape(transition.rew, (n_envs,))
    chex.assert_shape(transition.nobs, (n_envs,))
    assert state.agent.actions == 1


def test_nested_vmap_preserves_minimal_imc_structure():
    """Scalar IMC sampling supports more than one mapped leading axis."""
    imc, state = make_imc_state(jax.random.key(0))
    state = jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf),
            (2, 3, *jnp.shape(leaf)),
        ),
        state,
    )

    transition, state = jax.jit(jax.vmap(jax.vmap(imc.sample)))(state)

    chex.assert_shape(transition.act, (2, 3))
    chex.assert_shape(transition.nobs, (2, 3))
    chex.assert_shape(state.agent.actions, (2, 3))
