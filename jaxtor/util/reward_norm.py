"""Reward normalization via rolling discounted returns.

Normalizes rewards by dividing by the standard deviation of a rolling
discounted return estimate, following Huang et al. (2022).

Example:
    >>> from jaxtor.util import RunningStats
    >>> rms = RunningStats()
    >>> rn = RewardNorm(gamma=0.99, clip=10.0, rms=rms, seq_axis=1)
    >>> state = rn.init(batch_shape=(n_envs,))
    >>> norm_rewards, state = rn.update(rewards, dones, state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class Variance(Protocol):
    """State capability required for variance normalization."""

    var: jax.Array


class RMS[StateT: Variance](Protocol):
    """Running-statistics capability required by ``RewardNorm``."""

    def init(self, shape: tuple[int, ...] = ()) -> StateT: ...

    def update(self, batch: jax.Array, state: StateT) -> StateT: ...


@dataclass
class RewardNorm[RmsStateT: Variance]:
    """Reward normalization via rolling discounted returns.

    Attributes:
        gamma: Discount factor for rolling return estimation.
        rms: Running mean/variance component following the RMS protocol.
        seq_axis: Axis containing consecutive rewards.
        clip: If not None, clip normalized rewards to [-clip, clip].
        enabled: Whether normalization and statistics updates are active.

    Public dataclasses:
        State: Rolling returns and their running statistics.

    Public methods:
        init: Initialize return carries and scalar statistics.
        update: Normalize rewards and update rolling-return statistics.
    """

    gamma: float
    rms: RMS[RmsStateT]
    seq_axis: int = 0
    clip: float | None = None
    enabled: bool = True

    @dataclass
    class State[StatsT]:
        """Reward normalization state.

        Attributes:
            ret: Running discounted returns with the environment batch shape.
            rms: Running statistics state.
        """

        ret: jax.Array
        rms: StatsT

    @dataclass
    class _Step:
        """One reward and boundary slice across environment lanes."""

        rew: jax.Array
        done: jax.Array

    def init(
        self,
        batch_shape: tuple[int, ...] = (),
    ) -> RewardNorm.State[RmsStateT]:
        """Initialize one return carry per environment lane."""
        return self.State(
            ret=jnp.zeros(batch_shape),
            rms=self.rms.init(),
        )

    def _update_step(
        self,
        ret: jax.Array,
        step: RewardNorm._Step,
    ) -> tuple[jax.Array, jax.Array]:
        """Accumulate one reward and clear the carry after a boundary."""
        ret = ret * self.gamma + step.rew
        return jnp.where(step.done, 0, ret), ret

    def update(
        self,
        rewards: jax.Array,
        dones: jax.Array,
        state: RewardNorm.State[RmsStateT],
    ) -> tuple[jax.Array, RewardNorm.State[RmsStateT]]:
        """Update rolling return stats and return normalized rewards.

        Args:
            rewards: Rewards containing one sequence axis.
            dones: Episode boundaries with the same shape as ``rewards``.
            state: Current reward normalization state.

        Returns:
            Normalized rewards and updated state.
        """

        if not self.enabled:
            return rewards, self.State(ret=state.ret, rms=state.rms)

        chex.assert_equal_shape([rewards, dones])

        rew_t = jnp.moveaxis(rewards, self.seq_axis, 0)
        done_t = jnp.moveaxis(dones.astype(jnp.bool_), self.seq_axis, 0)
        if rew_t.shape[0] < 1:
            raise ValueError("reward sequence must not be empty")
        chex.assert_equal_shape([rew_t[0], state.ret])
        final_ret, all_rets = jax.lax.scan(
            self._update_step,
            state.ret,
            self._Step(rew=rew_t, done=done_t),
        )

        rms = self.rms.update(all_rets.reshape(-1), state.rms)
        norm_rewards = rewards / jnp.sqrt(rms.var + 1e-8)
        if self.clip is not None:
            norm_rewards = jnp.clip(norm_rewards, -self.clip, self.clip)

        return norm_rewards, RewardNorm.State(ret=final_ret, rms=rms)
