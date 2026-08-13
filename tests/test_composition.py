"""Cross-component correctness and computation-boundary tests."""

from dataclasses import replace

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass

from jaxtor.dist import Categorical
from jaxtor.estimate import TDEst
from jaxtor.sampler import EpisodeStats, Imc, Mc, Roll

pytestmark = pytest.mark.integration


@dataclass
class CounterEnv:
    """Deterministic counter used to make every expected value explicit."""

    @dataclass
    class State:
        """Current counter value."""

        count: jax.Array

    @dataclass
    class Step:
        """One nonterminal counter transition."""

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def init(self) -> State:
        """Initialize the counter at zero."""
        return self.State(count=jnp.array(0, dtype=jnp.int32))

    def reset(self, key: jax.Array, state: State) -> tuple[jax.Array, State]:
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
        """Increment the counter and return its value as reward."""
        del key, act
        count = state.count + 1
        return self.Step(
            nobs=count,
            rew=count.astype(jnp.float32),
            term=jnp.array(False),
            trun=jnp.array(False),
        ), self.State(count=count)


@dataclass
class Pred:
    """Value-policy output required by temporal-difference estimation."""

    v: jax.Array
    pi: Categorical


_replayed: list[int] = []


def _record_replayed(obs: jax.Array) -> None:
    """Record observation nodes actually evaluated by compiled replay."""
    _replayed.extend(int(value) for value in obs.reshape(-1))


@dataclass
class Agent:
    """Select zero actions, count them, and expose a value during replay."""

    @dataclass
    class State:
        """Number of actions consumed by the sampler."""

        actions: jax.Array

    def act(self, obs: jax.Array, state: State) -> tuple[jax.Array, State]:
        """Select zero and count exactly one sampler decision."""
        return jnp.zeros_like(obs), replace(state, actions=state.actions + 1)

    def apply(self, obs: jax.Array, state: State) -> tuple[Pred, State]:
        """Evaluate a value-policy prediction without selecting an action."""
        jax.debug.callback(_record_replayed, obs)
        value = obs.astype(jnp.float32)
        return Pred(
            v=value,
            pi=Categorical(logits=jnp.stack((value, -value), axis=-1)),
        ), state


def test_roll_estimate_and_stats_have_exact_values_and_work():
    """The composed path selects T actions and evaluates only needed TD nodes."""
    expected_adv = jnp.array([2.5, 2.0, 2.5, 2.0])
    expected_ret = jnp.array([2.5, 3.0, 2.5, 3.0])
    _replayed.clear()
    env = CounterEnv()
    agent = Agent()
    mc = Mc(max_eps_len=2, env=env)
    imc = Imc(agent=agent, mc=mc)
    roll = Roll(imc=imc, seq_len=4)
    estimator = TDEst(agent=agent, gamma=0.5, lam=1.0)
    stats = EpisodeStats()
    key = jax.random.key(0)
    state = imc.init(
        mc=mc.init(key, env.init()),
        agent=Agent.State(actions=jnp.array(0, dtype=jnp.int32)),
    )

    seq, state = jax.jit(roll.sample)(state)
    est = jax.jit(estimator.estimate)(seq, state.agent)
    metrics, _ = jax.jit(lambda seq: stats.drain(stats.update(seq, stats.init())))(seq)
    jax.effects_barrier()

    assert jnp.array_equal(seq.obs, jnp.array([0, 1, 0, 1]))
    assert jnp.array_equal(seq.nobs, jnp.array([1, 2, 1, 2]))
    assert jnp.array_equal(seq.trun, jnp.array([False, True, False, True]))
    assert jnp.allclose(est.adv, expected_adv)
    assert jnp.allclose(est.ret, expected_ret)
    assert metrics.avg_eps_rew == 3
    assert metrics.avg_eps_len == 2
    assert metrics.n_episodes == 2
    assert state.agent.actions == 4
    assert len(_replayed) == 6
    assert sorted(_replayed) == [0, 0, 1, 1, 2, 2]
    chex.assert_tree_all_finite((seq, est, metrics))
