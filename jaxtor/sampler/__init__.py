"""Sampling components for agent-environment interaction.

Components:
    Mc: Single-environment sampler with episode lifecycle handling.
    VecMc: Vectorized sampler for multiple parallel environments.
    Imc: Minimal action-only agent-MC interaction.
    Roll: Fixed-length collection from a minimal Imc.
    EpisodeStats: Completed-episode statistics from sequences.
    Sweep: Stochastic sweep over all (s,a) pairs.
    ExpSweep: Exact n-step propagation (backward/forward).
"""

from jaxtor.sampler.exp_sweep import ExpSweep
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.rollout import Roll
from jaxtor.sampler.stats import EpisodeStats
from jaxtor.sampler.sweep import Sweep

__all__ = [
    "Mc",
    "VecMc",
    "Imc",
    "Roll",
    "EpisodeStats",
    "Sweep",
    "ExpSweep",
]
