"""N-step trajectory collection utilities.

Collects fixed-length trajectories using jax.lax.scan over any IMC-compatible
sampler.

Classes:
    Rollout: N-step trajectory collector.

Example:
    >>> imc = InducedMarkovChain(agent=agent, mc=mc_sampler)
    >>> rollout = Rollout(imc=imc, seqlen=20)
    >>> state = imc.init(key, mc_state, agent_state)
    >>> transitions, state = rollout.sample(state)

    >>> vec_mc = VecMC(mc=mc_sampler, n_env=4)
    >>> imc = InducedMarkovChain(agent=batched_agent, mc=vec_mc)
    >>> rollout = Rollout(imc=imc, seqlen=20)
    >>> mc_state = vec_mc.init(key, env_state)
    >>> state = imc.init(key, mc_state, agent_state)
    >>> transitions, state = rollout.sample(state)
"""

from __future__ import annotations

from typing import Generic, Protocol, TypeVar

import jax
from chex import dataclass

Transition = TypeVar("Transition")
ImcState = TypeVar("ImcState")


class IMC(Protocol[Transition, ImcState]):
    def sample(self, state: ImcState) -> tuple[Transition, ImcState]: ...


@dataclass
class Rollout(Generic[Transition, ImcState]):
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

    def sample(self, state: ImcState) -> tuple[Transition, ImcState]:
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
