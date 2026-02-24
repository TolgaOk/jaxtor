"""Reward normalization via rolling discounted returns.

Normalizes rewards by dividing by the standard deviation of a rolling
discounted return estimate, following Huang et al. (2022).

Example:
    >>> from jaxtor.util import RunningStats
    >>> rms = RunningStats()
    >>> rn = RewardNorm(gamma=0.99, clip=10.0, rms=rms)
    >>> state = RewardNorm.State(
    ...     ret=jnp.zeros(n_envs),
    ...     rms=RunningStats.State(
    ...         mean=jnp.float32(0.0), var=jnp.float32(1.0),
    ...         count=jnp.float32(1e-4),
    ...     ),
    ... )
    >>> norm_rewards, state = rn.update(rewards, dones, state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax
import jax.numpy as jnp
from chex import dataclass


class RMS(Protocol):
    class State(Protocol):
        var: chex.Array

    def update(self, batch: chex.Array, state: RMS.State) -> RMS.State: ...


@dataclass
class RewardNorm:
    """Reward normalization via rolling discounted returns.

    Attributes:
        gamma: Discount factor for rolling return estimation.
        rms: Running mean/variance component following the RMS protocol.
        clip: If not None, clip normalized rewards to [-clip, clip].
    """

    gamma: float
    rms: RMS
    clip: float | None = None

    @dataclass
    class State:
        """Reward normalization state.

        Attributes:
            ret: Per-environment running discounted return, shape (n_envs,).
            rms: Running statistics state.
        """

        ret: chex.Array
        rms: RMS.State

    def update(
        self,
        rewards: chex.Array,
        dones: chex.Array,
        state: RewardNorm.State,
    ) -> tuple[chex.Array, RewardNorm.State]:
        """Update rolling return stats and return normalized rewards.

        Args:
            rewards: Shape (n_envs, seqlen).
            dones: Shape (n_envs, seqlen), episode boundary flags.
            state: Current reward normalization state.

        Returns:
            Normalized rewards and updated state.
        """

        def scan_fn(ret, step_data):
            rew, done = step_data
            ret = ret * self.gamma * (1.0 - done) + rew
            return ret, ret

        rew_t = jnp.transpose(rewards)
        done_t = jnp.transpose(dones.astype(jnp.float32))
        final_ret, all_rets = jax.lax.scan(scan_fn, state.ret, (rew_t, done_t))

        rms = self.rms.update(all_rets.reshape(-1), state.rms)
        norm_rewards = rewards / jnp.sqrt(rms.var + 1e-8)
        if self.clip is not None:
            norm_rewards = jnp.clip(norm_rewards, -self.clip, self.clip)

        return norm_rewards, RewardNorm.State(ret=final_ret, rms=rms)
