"""Tests for generic agent-induced Markov-chain sampling."""

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
            rew=act.astype(jnp.float32) + 1.0,
            term=count == 2,
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class RichAgent:
    """Agent exposing policy and value data on each decision."""

    @dataclass
    class State:
        """Agent parameters and number of prepared decisions."""

        offset: jax.Array
        decisions: jax.Array

    @dataclass
    class Decision:
        """Decision data consumed by sampling and learning."""

        act: chex.Array
        log_mu: jax.Array
        value: jax.Array

    def decide(
        self,
        obs: jax.Array,
        state: State,
    ) -> tuple[Decision, State]:
        """Prepare a deterministic action, log-probability, and value."""
        act = jnp.zeros_like(obs, dtype=jnp.int32)
        dec = self.Decision(
            act=act,
            log_mu=obs.astype(jnp.float32) - 0.5,
            value=obs.astype(jnp.float32) + state.offset,
        )
        return dec, state.replace(decisions=state.decisions + 1)


def make_imc_state(
    key: jax.Array,
    *,
    offset: float = 10.0,
) -> tuple[Imc, Imc.State]:
    """Build a scalar IMC and initialize its cached decision."""
    env = CounterEnv()
    mc = Mc(max_episode_len=10, queue_size=2, env=env)
    imc = Imc(agent=RichAgent(), mc=mc)
    mc_state = mc.init(key, env.init())
    state = imc.init(
        mc_state,
        RichAgent.State(
            offset=jnp.array(offset),
            decisions=jnp.array(0, dtype=jnp.int32),
        ),
    )
    return imc, state


def test_init_caches_complete_agent_decision():
    """Initialization prepares all agent-defined decision fields once."""
    imc, state = make_imc_state(jax.random.key(0))

    dec = imc.observe(state)

    chex.assert_shape([dec.act, dec.log_mu, dec.value], ())
    assert imc.mc.observe(state.mc) == 0
    assert dec.value == 10
    assert state.agent.decisions == 1


def test_sample_returns_mc_and_true_successor():
    """A sample separates the MC transition from successor agent data."""
    imc, state = make_imc_state(jax.random.key(0))

    sample, state = imc.sample(state)

    assert sample.mc.obs == 0
    assert sample.mc.act == 0
    assert sample.mc.nobs == 1
    assert sample.mc.rew == 1
    assert not sample.mc.term
    assert not sample.mc.trun
    assert sample.succ.log_mu == 0.5
    assert sample.succ.value == 11
    assert imc.observe(state) == sample.succ


def test_boundary_separates_successor_from_reset_node():
    """The learning successor remains terminal while the cache moves to reset."""
    imc, state = make_imc_state(jax.random.key(0))
    _, state = imc.sample(state)

    sample, state = imc.sample(state)

    assert sample.mc.term
    assert sample.mc.nobs == 2
    assert sample.succ.value == 12
    assert imc.mc.observe(state.mc) == 0
    assert imc.observe(state).value == 10


def test_refresh_recomputes_cached_node_from_current_agent_state():
    """Refreshing replaces a stale decision after an external agent update."""
    imc, state = make_imc_state(jax.random.key(0))
    changed = state.replace(agent=state.agent.replace(offset=jnp.array(20.0)))

    refreshed = imc.refresh(changed)

    assert imc.mc.observe(refreshed.mc) == imc.mc.observe(state.mc)
    assert refreshed.dec.value == 20
    assert refreshed.agent.decisions == state.agent.decisions + 1


def test_sample_is_jittable_and_tree_compatible():
    """IMC state and samples remain ordinary JAX pytrees under JIT."""
    imc, state = make_imc_state(jax.random.key(0))

    sample, next_state = jax.jit(imc.sample)(state)
    copied = jax.tree.map(lambda leaf: leaf, next_state)

    assert sample.succ.value == 11
    assert jax.tree.structure(copied) == jax.tree.structure(next_state)
    assert len(jax.tree.leaves(next_state)) > 0


def test_vec_mc_decisions_preserve_batch_axes():
    """One batched agent composes with a vectorized Markov chain."""
    n_envs = 3
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_episode_len=10, queue_size=2, env=env))
    imc = Imc(agent=RichAgent(), mc=mc)
    state = imc.init(
        mc.init(jax.random.split(jax.random.key(0), n_envs), env.init()),
        RichAgent.State(offset=jnp.array(10.0), decisions=jnp.array(0)),
    )

    sample, state = jax.jit(imc.sample)(state)

    chex.assert_shape(sample.mc.rew, (n_envs,))
    chex.assert_shape(sample.mc.nobs, (n_envs,))
    chex.assert_shape(sample.succ.value, (n_envs,))
    chex.assert_shape(state.dec.act, (n_envs,))


def test_vec_mc_selects_reset_nodes_only_for_boundary_lanes():
    """Mixed vector boundaries keep normal successors and replace done lanes."""
    n_envs = 3
    env = CounterEnv()
    mc = VecMc(mc=Mc(max_episode_len=10, queue_size=2, env=env))
    mc_state = mc.init(jax.random.split(jax.random.key(0), n_envs), env.init())
    obs = jnp.array([0, 1, 0], dtype=jnp.int32)
    mc_state = mc_state.replace(
        env=mc_state.env.replace(count=obs),
        last_obs=obs,
    )
    imc = Imc(agent=RichAgent(), mc=mc)
    state = imc.init(
        mc_state,
        RichAgent.State(offset=jnp.array(10.0), decisions=jnp.array(0)),
    )

    sample, state = jax.jit(imc.sample)(state)

    assert jnp.array_equal(sample.mc.term, jnp.array([False, True, False]))
    assert jnp.array_equal(sample.mc.nobs, jnp.array([1, 2, 1]))
    assert jnp.array_equal(imc.mc.observe(state.mc), jnp.array([1, 0, 1]))
    assert jnp.array_equal(state.dec.value, jnp.array([11, 10, 11]))


def test_nested_vmap_preserves_imc_structure():
    """Scalar IMC sampling supports more than one mapped leading axis."""
    imc, state = make_imc_state(jax.random.key(0))
    state = jax.tree.map(
        lambda leaf: jnp.broadcast_to(
            jnp.asarray(leaf),
            (2, 3, *jnp.shape(leaf)),
        ),
        state,
    )

    sample, state = jax.jit(jax.vmap(jax.vmap(imc.sample)))(state)

    chex.assert_shape(sample.mc.nobs, (2, 3))
    chex.assert_shape(sample.succ.value, (2, 3))
    chex.assert_shape(state.dec.value, (2, 3))
