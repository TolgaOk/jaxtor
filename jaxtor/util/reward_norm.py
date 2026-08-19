"""Reward normalization via rolling discounted returns.

Normalizes rewards by dividing by the standard deviation of a rolling
discounted return estimate, following Huang et al. (2022).

Example:
    >>> from jaxtor.util import RunningStats
    >>> stats = RunningStats()
    >>> norm = RewardNorm(gamma=0.99, clip=10.0, stats=stats, seq_axis=1)
    >>> state = norm.init(batch_shape=(n_envs,))
    >>> norm_rewards, state = norm.update(rewards, dones, state)
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


class Stats[S: Variance](Protocol):
    """Running-statistics capability required by ``RewardNorm``."""

    def init(self, shape: tuple[int, ...] = ()) -> S: ...
    def update(self, batch: jax.Array, state: S) -> S: ...


@dataclass
class RewardNorm[StatsS: Variance]:
    """Reward normalization via rolling discounted returns.

    Required protocols::

        stats.init(shape) -> stats_state
        stats.update(batch, stats_state) -> stats_state
        stats_state.var: jax.Array

    Attributes:
        gamma: Discount factor for rolling return estimation.
        stats: Running-statistics component used for return normalization.
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
    stats: Stats[StatsS]
    seq_axis: int = 0
    clip: float | None = None
    enabled: bool = True

    @dataclass
    class State[StatsData]:
        """Reward normalization state.

        Attributes:
            ret: Running discounted returns with the environment batch shape.
            stats: Running statistics state.
        """

        ret: jax.Array
        stats: StatsData

    @dataclass
    class _Step:
        """One reward and boundary slice across environment lanes."""

        rew: jax.Array
        done: jax.Array

    def __post_init__(self) -> None:
        """Validate discounting and optional clipping configuration."""
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if self.clip is not None and self.clip < 0:
            raise ValueError("clip must be nonnegative")

    def init(
        self,
        batch_shape: tuple[int, ...] = (),
    ) -> RewardNorm.State[StatsS]:
        """Initialize one return carry per environment lane."""
        return self.State(
            ret=jnp.zeros(batch_shape),
            stats=self.stats.init(),
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
        state: RewardNorm.State[StatsS],
    ) -> tuple[jax.Array, RewardNorm.State[StatsS]]:
        """Update rolling return stats and return normalized rewards.

        Args:
            rewards: Rewards containing one sequence axis.
            dones: Episode boundaries with the same shape as ``rewards``.
            state: Current reward normalization state.

        Returns:
            Normalized rewards and updated state.
        """

        if not self.enabled:
            return rewards, state

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

        stats = self.stats.update(all_rets.reshape(-1), state.stats)
        norm_rewards = rewards / jnp.sqrt(stats.var + 1e-8)
        if self.clip is not None:
            norm_rewards = jnp.clip(norm_rewards, -self.clip, self.clip)

        return norm_rewards, self.State(ret=final_ret, stats=stats)
