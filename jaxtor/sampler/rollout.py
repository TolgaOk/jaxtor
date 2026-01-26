"""N-step trajectory collection utilities.

Implements trajectory collection via a configurable single-step sampler protocol.
Any object implementing the IMC protocol can be used.

Example:
    >>> import jax
    >>> from jaxtor.sampler import rollout, imc, mc
    >>>
    >>> # Create IMC
    >>> mc_sampler = mc.MarkovChain(max_episode_len=1000, queue_size=100, env=env)
    >>> imc_step = imc.InducedMarkovChain(agent=agent, mc=mc_sampler)
    >>> sampler = rollout.Rollout(imc=imc_step, seqlen=8)
    >>>
    >>> # Initialize state (via IMC, not Rollout)
    >>> key = jax.random.PRNGKey(0)
    >>> mc_state = mc_sampler.init(key)
    >>> state = imc_step.init(mc=mc_state, agent=agent_state)
    >>>
    >>> # Collect trajectory
    >>> transitions, state = sampler.sample(state)
    >>>
    >>> # Get metrics (via IMC, not Rollout)
    >>> metrics, state = imc_step.metrics(state)
    >>>
    >>> # Vectorized version
    >>> vsampler = rollout.VecRollout(imc=imc_step, seqlen=8, num_envs=4)
    >>> transitions, states = vsampler.sample(batched_states)
"""

from __future__ import annotations

from typing import Protocol, TypeVar

import jax
from chex import dataclass


ImcState = TypeVar("ImcState")
StepTransition = TypeVar("StepTransition")
BatchTransition = StepTransition


class IMC(Protocol[ImcState, StepTransition]):
    def sample(self, state: ImcState) -> tuple[StepTransition, ImcState]: ...


@dataclass
class Rollout:
    """N-step trajectory collector.

    Works with any single-step sampler implementing the IMC protocol.

    Attributes:
        imc: Single-step sampler following the IMC protocol.
        seqlen: Number of steps to collect per trajectory.
        _unroll: Number of loop iterations to unroll in scan (default: 1).
    """

    imc: IMC
    seqlen: int
    _unroll: int = 1

    def sample(self, state: ImcState) -> tuple[BatchTransition, ImcState]:
        """Collect seqlen transitions.

        Args:
            state: Current IMC state.

        Returns:
            Stacked transitions with shape (seqlen, ...) and updated state.
        """

        def step(state, _):
            transition, state = self.imc.sample(state)
            return state, transition

        state, transitions = jax.lax.scan(
            step, state, length=self.seqlen, unroll=self._unroll
        )
        return transitions, state


@dataclass
class VecRollout:
    """Vectorized N-step trajectory collector for multiple environments.

    Collects trajectories from num_envs parallel environments simultaneously.
    Uses vmap internally for efficient batched execution.

    Attributes:
        imc: Single-step sampler following the IMC protocol.
        seqlen: Number of steps to collect per trajectory.
        num_envs: Number of parallel environments.
        _unroll: Number of loop iterations to unroll in scan (default: 1).
    """

    imc: IMC
    seqlen: int
    num_envs: int
    _unroll: int = 1

    def sample(self, state: ImcState) -> tuple[BatchTransition, ImcState]:
        """Collect seqlen transitions from num_envs environments.

        Args:
            state: Batched IMC state with leading dimension num_envs.

        Returns:
            Transitions with shape (num_envs, seqlen, ...) and updated state.
        """

        def step(state, _):
            transition, state = jax.vmap(self.imc.sample)(state)
            return state, transition

        state, transitions = jax.lax.scan(
            step, state, length=self.seqlen, unroll=self._unroll
        )
        transitions = jax.tree.map(lambda x: x.swapaxes(0, 1), transitions)
        return transitions, state
