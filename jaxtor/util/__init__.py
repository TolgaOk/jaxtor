"""Utility components for batching, normalization, and statistics.

Components:
    Minibatches: Shuffled equal-size batches from aligned pytrees.
    ObsNorm: Explicitly updated observation normalization.
    RewardNorm: Reward normalization via rolling discounted returns.
    RunningStats: Online mean/variance tracking (Welford algorithm).
"""

from jaxtor.util.minibatches import Minibatches
from jaxtor.util.obs_norm import ObsNorm
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats

__all__ = [
    "Minibatches",
    "ObsNorm",
    "RewardNorm",
    "RunningStats",
]
