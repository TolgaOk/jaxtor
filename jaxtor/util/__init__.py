"""Utility components for normalization and statistics.

Components:
    running_stats: Online mean/variance tracking (Welford algorithm).
    reward_norm: Reward normalization via rolling discounted returns.
"""

from jaxtor.util import reward_norm, running_stats
from jaxtor.util.reward_norm import RewardNorm
from jaxtor.util.running_stats import RunningStats

__all__ = ["running_stats", "reward_norm", "RunningStats", "RewardNorm"]
