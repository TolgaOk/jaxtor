"""Tests for information-loaded rollout collection."""

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
        """Value offset and number of committed inferences."""

        offset: jax.Array
        inferences: jax.Array

    @dataclass
    class Output:
        """Agent information evaluated at one observation."""

        act: jax.Array
        log_mu: jax.Array
        value: jax.Array
        token: jax.Array

    def infer(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[Output, State]:
        """Infer rich data and advance the functional inference counter."""
        value = obs.astype(jnp.float32) + state.offset
        output = self.Output(
            act=jnp.zeros_like(obs, dtype=jnp.int32),
            log_mu=-obs.astype(jnp.float32),
            value=value,
            token=jnp.zeros_like(value) + state.inferences,
        )
        return output, state.replace(  # type: ignore[reportAttributeAccessIssue]
            inferences=state.inferences + 1
        )


def make_loaded_roll(
    key: jax.Array,
    *,
    seqlen: int,
) -> tuple[LoadedRoll, LoadedRoll.State]:
    """Build a scalar loaded rollout and its persistent state."""
    env = CounterEnv()
    mc = Mc(max_episode_len=10, env=env)
    rollout = LoadedRoll(agent=RichAgent(), mc=mc, seqlen=seqlen)
    state = rollout.init(
        mc.init(key, env.init()),
        RichAgent.State(
            offset=jnp.array(10.0),
            inferences=jnp.array(0, dtype=jnp.int32),
        ),
    )
    return rollout, state


def test_loaded_roll_aligns_rich_outputs_with_mc_transitions():
    """Predecessor and true-successor outputs align around every MC step."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=2)

    trajectory, state = rollout.sample(state)

    assert jnp.array_equal(trajectory.pre.value, jnp.array([10.0, 11.0]))
    assert jnp.array_equal(trajectory.mc.obs, jnp.array([0, 1]))
    assert jnp.array_equal(trajectory.mc.nobs, jnp.array([1, 2]))
    assert jnp.array_equal(trajectory.succ.value, jnp.array([11.0, 12.0]))
    assert jnp.array_equal(trajectory.pre.act, trajectory.mc.act)
    assert state.agent.inferences == 2


def test_loaded_roll_reuses_normal_successors_inside_one_rollout():
    """A normal successor becomes the exact next predecessor output."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=2)

    trajectory, _ = rollout.sample(state)

    assert trajectory.succ.value[0] == trajectory.pre.value[1]
    assert trajectory.succ.log_mu[0] == trajectory.pre.log_mu[1]
    assert trajectory.succ.token[0] == trajectory.pre.token[1]


def test_loaded_roll_separates_true_successor_from_reset_predecessor():
    """A boundary retains terminal inference and loads reset inference next."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=4)

    trajectory, state = rollout.sample(state)

    assert jnp.array_equal(trajectory.mc.obs, jnp.array([0, 1, 2, 0]))
    assert trajectory.succ.value[2] == 13
    assert trajectory.pre.value[3] == 10
    assert state.agent.inferences == 4


def test_loaded_roll_does_not_persist_outputs_between_calls():
    """The next call infers its first predecessor from updated agent state."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=2)
    _, state = rollout.sample(state)
    state = state.replace(  # type: ignore[reportAttributeAccessIssue]
        agent=state.agent.replace(offset=jnp.array(100.0))
    )

    trajectory, _ = rollout.sample(state)

    assert trajectory.pre.value[0] == 102
    assert not hasattr(state, "pre")


def test_loaded_roll_is_jittable_and_tree_compatible():
    """Loaded trajectory collection remains a regular JAX pytree under JIT."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=4)

    trajectory, state = jax.jit(rollout.sample)(state)
    copied = jax.tree.map(lambda leaf: leaf, trajectory)

    chex.assert_shape(trajectory.pre.value, (4,))
    chex.assert_shape(trajectory.succ.value, (4,))
    assert jax.tree.structure(copied) == jax.tree.structure(trajectory)
    assert state.agent.inferences == 4


def test_loaded_roll_handles_mixed_vector_boundaries():
    """Only reset lanes differ between true successors and next predecessors."""
    n_envs = 3
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_episode_len=10, env=env))
    mc_state = mc.init(jax.random.split(jax.random.key(0), n_envs), env.init())
    obs = jnp.array([0, 2, 1], dtype=jnp.int32)
    mc_state = mc_state.replace(  # type: ignore[reportAttributeAccessIssue]
        env=mc_state.env.replace(  # type: ignore[reportAttributeAccessIssue]
            count=obs
        ),
        last_obs=obs,
    )
    rollout = LoadedRoll(
        agent=RichAgent(),
        mc=mc,
        seqlen=2,
        seq_axis=1,
    )
    state = rollout.init(
        mc_state,
        RichAgent.State(offset=jnp.array(10.0), inferences=jnp.array(0)),
    )

    trajectory, _ = jax.jit(rollout.sample)(state)

    chex.assert_shape(trajectory.pre.value, (n_envs, 2))
    chex.assert_shape(trajectory.succ.value, (n_envs, 2))
    assert jnp.array_equal(trajectory.succ.value[:, 0], jnp.array([11, 13, 12]))
    assert jnp.array_equal(trajectory.pre.value[:, 1], jnp.array([11, 10, 12]))


def test_nested_vmap_preserves_loaded_roll_structure():
    """A scalar loaded rollout supports more than one mapped leading axis."""
    rollout, state = make_loaded_roll(jax.random.key(0), seqlen=3)
    state = jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf),
            (2, 3, *jnp.shape(leaf)),
        ),
        state,
    )

    trajectory, state = jax.jit(jax.vmap(jax.vmap(rollout.sample)))(state)

    chex.assert_shape(trajectory.pre.value, (2, 3, 3))
    chex.assert_shape(trajectory.mc.nobs, (2, 3, 3))
    chex.assert_shape(trajectory.succ.value, (2, 3, 3))
    chex.assert_shape(state.agent.inferences, (2, 3))
