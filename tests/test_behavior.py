"""Tests for stateful action-behavior components."""

import chex
import jax
import jax.numpy as jnp
import pytest
from chex import dataclass

from jaxtor.agent import Ou


@dataclass
class ShiftAgent:
    """Add scalar state to observations and advance that state."""

    def act(
        self,
        obs: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return a predictable action and updated child state."""
        return obs + state, state + 1


def test_ou_matches_one_process_step_under_jit():
    """Acting follows the documented recurrence and advances both children."""
    behavior = Ou(
        agent=ShiftAgent(),
        theta=0.2,
        sigma=0.3,
        dt=0.5,
        low=-10.0,
        high=10.0,
    )
    key = jax.random.key(7)
    obs = jnp.array([0.2, -0.4])
    noise = jnp.array([0.1, -0.2])
    state = behavior.init(key, noise, jnp.array(0.5))

    act, next_state = jax.jit(behavior.act)(obs, state)

    next_key, sample_key = jax.random.split(key)
    expected_noise = (
        noise
        - 0.2 * noise * 0.5
        + 0.3
        * jnp.sqrt(0.5)
        * jax.random.normal(sample_key, obs.shape, dtype=obs.dtype)
    )
    chex.assert_trees_all_close(act, obs + 0.5 + expected_noise)
    chex.assert_trees_all_close(next_state.noise, expected_noise)
    chex.assert_trees_all_equal(next_state.key, next_key)
    assert next_state.agent == 1.5


def test_ou_clips_actions_without_clipping_process_state():
    """Bounds affect emitted actions while preserving the latent process."""
    behavior = Ou(agent=ShiftAgent(), sigma=0.0, low=-1.0, high=1.0)
    state = behavior.init(
        jax.random.key(0),
        jnp.array([2.0, -2.0]),
        jnp.array(0.0),
    )

    act, next_state = behavior.act(jnp.zeros(2), state)

    chex.assert_trees_all_close(act, jnp.array([1.0, -1.0]))
    chex.assert_trees_all_close(next_state.noise, jnp.array([1.7, -1.7]))


def test_ou_supports_nested_vmap_states():
    """Explicit noise shape supports independently keyed nested vectorization."""
    behavior = Ou(agent=ShiftAgent(), sigma=0.1)
    keys = jax.random.split(jax.random.key(1), 12).reshape((3, 4))
    state = behavior.init(
        keys,
        jnp.zeros((3, 4, 2)),
        jnp.zeros((3, 4)),
    )
    obs = jnp.zeros((3, 4, 2))

    act, next_state = jax.vmap(jax.vmap(behavior.act))(obs, state)

    assert act.shape == (3, 4, 2)
    assert next_state.noise.shape == (3, 4, 2)
    assert next_state.key.shape == keys.shape
    chex.assert_trees_all_equal(next_state.agent, jnp.ones((3, 4)))


def test_ou_rejects_action_noise_shape_mismatch():
    """A mismatched process state fails at the component boundary."""
    behavior = Ou(agent=ShiftAgent())
    state = behavior.init(
        jax.random.key(0),
        jnp.zeros(3),
        jnp.array(0.0),
    )

    with pytest.raises(AssertionError):
        behavior.act(jnp.zeros(2), state)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"theta": -0.1}, "theta must be non-negative"),
        ({"sigma": -0.1}, "sigma must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"low": 1.0, "high": 1.0}, "low must be less than high"),
    ],
)
def test_ou_rejects_invalid_configuration(kwargs, message):
    """Invalid process parameters fail during configuration."""
    with pytest.raises(ValueError, match=message):
        Ou(agent=ShiftAgent(), **kwargs)
