"""Utility components for normalization and statistics.

Components:
    ObsNorm: Explicitly updated observation normalization.
    RewardNorm: Reward normalization via rolling discounted returns.
    RunningStats: Online mean/variance tracking (Welford algorithm).
"""

from jaxtor.util import obs_norm, reward_norm, running_stats
from jaxtor.util.obs_norm import ObsNorm
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats

__all__ = [
    "obs_norm",
    "reward_norm",
    "running_stats",
    "ObsNorm",
    "RewardNorm",
    "RunningStats",
]
