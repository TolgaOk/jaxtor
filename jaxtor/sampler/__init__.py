"""Sampling components for agent-environment interaction.

Components:
    Mc: Single-environment sampler with episode lifecycle handling.
    VecMc: Vectorized sampler for multiple parallel environments.
    Imc: Induced Markov chain (agent-MC interaction).
    Roll: N-step trajectory collector.
    EpisodeStats: Completed-episode statistics from trajectories.
    Sweep: Stochastic sweep over all (s,a) pairs.
    ExpSweep: Exact n-step propagation (backward/forward).
"""

from jaxtor.sampler.exp_sweep import ExpSweep
from jaxtor.sampler.stats import EpisodeStats
from jaxtor.sampler.imc import Imc
from jaxtor.sampler.mc import Mc, VecMc
from jaxtor.sampler.rollout import Roll
from jaxtor.sampler.sweep import Sweep

__all__ = ["Mc", "VecMc", "Imc", "Roll", "EpisodeStats", "Sweep", "ExpSweep"]
