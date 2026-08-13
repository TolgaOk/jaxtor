"""Tests for information-loaded sequence collection."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
from chex import dataclass

from jaxtor.sampler import LoadedRoll, Mc, VecMc


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
        del key, act
        count = state.count + 1
        return self.Step(
            nobs=count,
            rew=count.astype(jnp.float32),
            term=count == 3,
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class RichAgent:
    """Expose an action, behavior log-probability, and value."""

    @dataclass
    class State:
        """Value offset and number of committed calls."""

        offset: jax.Array
        calls: jax.Array

    @dataclass
    class Output:
        """Agent information evaluated at one observation."""

        act: jax.Array
        log_mu: jax.Array
        value: jax.Array
        token: jax.Array

    def apply(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[Output, State]:
        """Predict rich data and advance the functional call counter."""
        value = obs.astype(jnp.float32) + state.offset
        output = self.Output(
            act=jnp.zeros_like(obs, dtype=jnp.int32),
            log_mu=-obs.astype(jnp.float32),
            value=value,
            token=jnp.zeros_like(value) + state.calls,
        )
        return output, replace(state, calls=state.calls + 1)


def make_loaded_roll(
    key: jax.Array,
    *,
    seq_len: int,
) -> tuple[LoadedRoll, LoadedRoll.State]:
    """Build a scalar loaded sequence component and its persistent state."""
    env = CounterEnv()
    mc = Mc(max_eps_len=10, env=env)
    roll = LoadedRoll(agent=RichAgent(), mc=mc, seq_len=seq_len)
    state = roll.init(
        mc.init(key, env.init()),
        RichAgent.State(
            offset=jnp.array(10.0),
            calls=jnp.array(0, dtype=jnp.int32),
        ),
    )
    return roll, state


def test_loaded_roll_aligns_rich_outputs_with_mc_transitions():
    """Decision and true-successor outputs align around every MC step."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=2)

    seq, state = roll.sample(state)

    assert jnp.array_equal(seq.dec.value, jnp.array([10.0, 11.0]))
    assert jnp.array_equal(seq.mc.obs, jnp.array([0, 1]))
    assert jnp.array_equal(seq.mc.nobs, jnp.array([1, 2]))
    assert jnp.array_equal(seq.succ.value, jnp.array([11.0, 12.0]))
    assert jnp.array_equal(seq.dec.act, seq.mc.act)
    assert state.agent.calls == 2


def test_loaded_roll_reuses_normal_successors_inside_one_sequence():
    """A normal successor becomes the exact next decision."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=2)

    seq, _ = roll.sample(state)

    assert seq.succ.value[0] == seq.dec.value[1]
    assert seq.succ.log_mu[0] == seq.dec.log_mu[1]
    assert seq.succ.token[0] == seq.dec.token[1]


def test_loaded_roll_separates_true_successor_from_reset_decision():
    """A boundary retains terminal output and loads reset output next."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=4)

    seq, state = roll.sample(state)

    assert jnp.array_equal(seq.mc.obs, jnp.array([0, 1, 2, 0]))
    assert seq.succ.value[2] == 13
    assert seq.dec.value[3] == 10
    assert state.agent.calls == 4


def test_loaded_roll_does_not_persist_outputs_between_calls():
    """The next call recomputes its first decision from updated agent state."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=2)
    _, state = roll.sample(state)
    state = replace(
        state,
        agent=replace(state.agent, offset=jnp.array(100.0)),
    )

    seq, _ = roll.sample(state)

    assert seq.dec.value[0] == 102
    assert not hasattr(state, "dec")


def test_loaded_roll_is_jittable_and_tree_compatible():
    """A loaded sequence remains a regular JAX pytree under JIT."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=4)

    seq, state = jax.jit(roll.sample)(state)
    copied = jax.tree.map(lambda leaf: leaf, seq)

    chex.assert_shape(seq.dec.value, (4,))
    chex.assert_shape(seq.succ.value, (4,))
    assert jax.tree.structure(copied) == jax.tree.structure(seq)
    assert state.agent.calls == 4


def test_loaded_roll_handles_mixed_vector_boundaries():
    """Only reset lanes differ between true successors and next decisions."""
    n_envs = 3
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_eps_len=10, env=env))
    mc_state = mc.init(jax.random.split(jax.random.key(0), n_envs), env.init())
    obs = jnp.array([0, 2, 1], dtype=jnp.int32)
    mc_state = replace(
        mc_state,
        env=replace(mc_state.env, count=obs),
        last_obs=obs,
    )
    roll = LoadedRoll(
        agent=RichAgent(),
        mc=mc,
        seq_len=2,
        seq_axis=1,
    )
    state = roll.init(
        mc_state,
        RichAgent.State(offset=jnp.array(10.0), calls=jnp.array(0)),
    )

    seq, _ = jax.jit(roll.sample)(state)

    chex.assert_shape(seq.dec.value, (n_envs, 2))
    chex.assert_shape(seq.succ.value, (n_envs, 2))
    assert jnp.array_equal(seq.succ.value[:, 0], jnp.array([11, 13, 12]))
    assert jnp.array_equal(seq.dec.value[:, 1], jnp.array([11, 10, 12]))


def test_nested_vmap_preserves_loaded_roll_structure():
    """A scalar loaded sequence supports more than one mapped leading axis."""
    roll, state = make_loaded_roll(jax.random.key(0), seq_len=3)
    state = jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf),
            (2, 3, *jnp.shape(leaf)),
        ),
        state,
    )

    seq, state = jax.jit(jax.vmap(jax.vmap(roll.sample)))(state)

    chex.assert_shape(seq.dec.value, (2, 3, 3))
    chex.assert_shape(seq.mc.nobs, (2, 3, 3))
    chex.assert_shape(seq.succ.value, (2, 3, 3))
    chex.assert_shape(state.agent.calls, (2, 3))
