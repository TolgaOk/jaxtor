"""Tests for decision-MC-successor trajectory collection."""

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
        del key, act
        count = state.count + 1
        return self.Step(
            nobs=count,
            rew=count.astype(jnp.float32),
            term=count == 3,
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class ValueAgent:
    """Agent attaching a behavior log-probability and value to each decision."""

    @dataclass
    class State:
        """Number of decisions prepared by the agent."""

        decisions: jax.Array

    @dataclass
    class Decision:
        """Learning data attached to a decision."""

        act: chex.Array
        log_mu: jax.Array
        value: jax.Array

    def decide(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[Decision, State]:
        """Prepare one deterministic decision."""
        dec = self.Decision(
            act=jnp.zeros_like(obs, dtype=jnp.int32),
            log_mu=-obs.astype(jnp.float32),
            value=10.0 * obs.astype(jnp.float32),
        )
        return dec, state.replace(decisions=state.decisions + 1)


def make_roll(
    key: jax.Array,
    *,
    seqlen: int,
) -> tuple[Roll, Imc.State]:
    """Build a scalar rollout and its initialized IMC state."""
    env = CounterEnv()
    mc = Mc(max_episode_len=10, env=env)
    imc = Imc(agent=ValueAgent(), mc=mc)
    state = imc.init(
        mc.init(key, env.init()),
        ValueAgent.State(decisions=jnp.array(0, dtype=jnp.int32)),
    )
    return Roll(imc=imc, seqlen=seqlen), state


def test_roll_stacks_three_aligned_sequences():
    """Agent nodes, MC transitions, and successors all have length T."""
    roll, state = make_roll(jax.random.key(0), seqlen=5)

    trajectory, _ = roll.sample(state)

    chex.assert_shape(trajectory.dec.log_mu, (5,))
    chex.assert_shape(trajectory.mc.obs, (5,))
    chex.assert_shape(trajectory.mc.rew, (5,))
    chex.assert_shape(trajectory.mc.nobs, (5,))
    chex.assert_shape(trajectory.succ.value, (5,))


def test_roll_exposes_normal_and_boundary_alignment():
    """Normal successors continue while terminal successors precede reset nodes."""
    roll, state = make_roll(jax.random.key(0), seqlen=4)

    trajectory, state = roll.sample(state)

    assert jnp.array_equal(trajectory.mc.obs, jnp.array([0, 1, 2, 0]))
    assert jnp.array_equal(trajectory.mc.nobs, jnp.array([1, 2, 3, 1]))
    assert jnp.array_equal(trajectory.mc.term, jnp.array([False, False, True, False]))
    assert roll.imc.mc.observe(state.mc) == 1


def test_roll_uses_cached_successors_between_calls():
    """The next rollout begins at the decision cached by the previous one."""
    roll, state = make_roll(jax.random.key(0), seqlen=2)
    _, state = roll.sample(state)
    expected = state.dec
    expected_obs = roll.imc.mc.observe(state.mc)

    trajectory, _ = roll.sample(state)

    assert trajectory.mc.obs[0] == expected_obs
    assert trajectory.dec.value[0] == expected.value


def test_roll_is_jittable():
    """The complete trajectory collection compiles under JIT."""
    roll, state = make_roll(jax.random.key(0), seqlen=5)

    trajectory, state = jax.jit(roll.sample)(state)

    chex.assert_shape(trajectory.mc.obs, (5,))
    chex.assert_shape(state.dec.value, ())


def test_roll_moves_one_sequence_axis_consistently():
    """Vectorized rollouts place all three sequences on the configured axis."""
    n_envs = 3
    seqlen = 4
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_episode_len=10, env=env))
    imc = Imc(agent=ValueAgent(), mc=mc)
    state = imc.init(
        mc.init(jax.random.split(jax.random.key(0), n_envs), env.init()),
        ValueAgent.State(decisions=jnp.array(0, dtype=jnp.int32)),
    )
    roll = Roll(imc=imc, seqlen=seqlen, seq_axis=1)

    trajectory, _ = jax.jit(roll.sample)(state)

    chex.assert_shape(trajectory.mc.obs, (n_envs, seqlen))
    chex.assert_shape(trajectory.mc.rew, (n_envs, seqlen))
    chex.assert_shape(trajectory.mc.nobs, (n_envs, seqlen))
    chex.assert_shape(trajectory.succ.value, (n_envs, seqlen))


def test_roll_unroll_factor_preserves_results():
    """Changing scan unrolling does not change the trajectory."""
    roll, state = make_roll(jax.random.key(0), seqlen=6)
    expected, _ = roll.sample(state)

    actual, _ = Roll(imc=roll.imc, seqlen=6, _unroll=3).sample(state)

    assert all(
        bool(jnp.array_equal(left, right))
        for left, right in zip(
            jax.tree.leaves(actual),
            jax.tree.leaves(expected),
            strict=True,
        )
    )
