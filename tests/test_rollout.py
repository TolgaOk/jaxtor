"""Tests for fixed-length collection from a minimal IMC."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.sampler import Imc, Mc, Roll, VecMc


@dataclass
class CounterEnv:
    """Deterministic environment terminating after three actions."""

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
        """Return the initial environment state."""
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
        """Advance one counter step."""
        del key
        count = state.count + 1
        return self.Step(
            nobs=count,
            rew=act.astype(jnp.float32),
            term=count == 3,
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class CountingAgent:
    """Select zero and count how many actions were requested."""

    @dataclass
    class State:
        """Number of selected actions."""

        actions: jax.Array

    def act(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[jax.Array, State]:
        """Return a zero action matching the observation batch shape."""
        act = jnp.zeros_like(obs, dtype=jnp.int32)
        return act, replace(state, actions=state.actions + 1)


def make_roll(
    key: jax.Array,
    *,
    seq_len: int,
) -> tuple[Roll, Imc.State]:
    """Build a scalar minimal rollout and its state."""
    env = CounterEnv()
    mc = Mc(max_eps_len=10, env=env)
    imc = Imc(agent=CountingAgent(), mc=mc)
    state = imc.init(
        mc.init(key, env.init()),
        CountingAgent.State(actions=jnp.array(0, dtype=jnp.int32)),
    )
    return Roll(imc=imc, seq_len=seq_len), state


def test_roll_stacks_mc_transitions_directly():
    """The MC transition pytree receives the configured sequence length."""
    roll, state = make_roll(jax.random.key(0), seq_len=5)

    seq, _ = roll.sample(state)

    chex.assert_shape(seq.act, (5,))
    chex.assert_shape(seq.obs, (5,))
    chex.assert_shape(seq.rew, (5,))
    chex.assert_shape(seq.nobs, (5,))


def test_roll_exposes_normal_and_boundary_alignment():
    """Normal successors continue while terminal successors precede reset."""
    roll, state = make_roll(jax.random.key(0), seq_len=4)

    seq, state = roll.sample(state)

    assert jnp.array_equal(seq.obs, jnp.array([0, 1, 2, 0]))
    assert jnp.array_equal(seq.nobs, jnp.array([1, 2, 3, 1]))
    assert jnp.array_equal(
        seq.term,
        jnp.array([False, False, True, False]),
    )
    assert state.mc.last_obs == 1


def test_roll_recomputes_from_updated_agent_state_between_calls():
    """Ordinary Roll carries no derived agent output between calls."""
    roll, state = make_roll(jax.random.key(0), seq_len=2)
    _, state = roll.sample(state)
    state = replace(state, agent=replace(state.agent, actions=jnp.array(100)))

    _, state = roll.sample(state)

    assert state.agent.actions == 102


def test_roll_is_jittable():
    """The complete minimal sequence collection compiles under JIT."""
    roll, state = make_roll(jax.random.key(0), seq_len=5)

    seq, state = jax.jit(roll.sample)(state)

    chex.assert_shape(seq.obs, (5,))
    assert state.agent.actions == 5


def test_roll_moves_one_sequence_axis_consistently():
    """Vectorized rollouts place both sequences on the configured axis."""
    n_envs = 3
    seq_len = 4
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_eps_len=10, env=env))
    imc = Imc(agent=CountingAgent(), mc=mc)
    state = imc.init(
        mc.init(jax.random.split(jax.random.key(0), n_envs), env.init()),
        CountingAgent.State(actions=jnp.array(0, dtype=jnp.int32)),
    )
    roll = Roll(imc=imc, seq_len=seq_len, seq_axis=1)

    seq, _ = jax.jit(roll.sample)(state)

    chex.assert_shape(seq.act, (n_envs, seq_len))
    chex.assert_shape(seq.obs, (n_envs, seq_len))
    chex.assert_shape(seq.rew, (n_envs, seq_len))
    chex.assert_shape(seq.nobs, (n_envs, seq_len))


def test_roll_unroll_factor_preserves_results():
    """Changing scan unrolling does not change the sequence."""
    roll, state = make_roll(jax.random.key(0), seq_len=6)
    expected, _ = roll.sample(state)

    actual, _ = Roll(imc=roll.imc, seq_len=6, _unroll=3).sample(state)

    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(actual),
            jax.tree.leaves(expected),
            strict=True,
        )
    )
