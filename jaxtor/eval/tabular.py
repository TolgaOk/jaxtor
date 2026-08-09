"""Convergence evaluation for tabular value-learning agents.

``Eval`` compares an agent's Q-values with their previous values, the Bellman
optimality target, and known optimal Q-values.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar, cast

import chex
import jax
import jax.numpy as jnp
from chex import dataclass  # pyright: ignore[reportUnknownVariableType]
from jaxdp.base import bellman_optimality_operator as jaxdp_bellman
from jaxdp.base import greedy_policy as jaxdp_greedy
from jaxdp.base import policy_evaluation as jaxdp_policy_evaluation


class Mdp(Protocol):
    """Tabular MDP surface consumed by this module."""

    initial: jax.Array
    terminal: jax.Array

    @property
    def state_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...


_greedy_policy = cast(
    Callable[[jax.Array], jax.Array],
    jaxdp_greedy.q,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
)
_evaluate_policy = cast(
    Callable[[Mdp, jax.Array, float], jax.Array],
    jaxdp_policy_evaluation.q,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
)
_bellman_optimality = cast(
    Callable[[Mdp, jax.Array, float], jax.Array],
    jaxdp_bellman.q,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
)


def optimal_q(mdp: Mdp, gamma: float, n_iters: int = 20) -> jax.Array:
    """Compute optimal Q-values via policy iteration.

    Alternates greedy policy extraction and exact policy evaluation for a fixed
    number of iterations.

    Args:
        mdp: Tabular MDP instance.
        gamma: Discount factor.
        n_iters: Number of policy-iteration steps.

    Returns:
        Optimal Q-values with shape ``(A, S)``.
    """
    q = jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
        (mdp.action_size, mdp.state_size)
    )
    for _ in range(n_iters):
        q = _evaluate_policy(mdp, _greedy_policy(q), gamma)
    return q


AgentState = TypeVar("AgentState", contravariant=True)


class Agent(Protocol[AgentState]):
    """Q-value surface consumed by ``Eval``."""

    def q_vals(self, state: AgentState, obs: jax.Array) -> jax.Array: ...


@dataclass  # pyright: ignore[reportUntypedClassDecorator]
class Eval(Generic[AgentState]):
    """Evaluate convergence of a tabular value-learning agent.

    Attributes:
        mdp: Tabular MDP being solved.
        gamma: Discount factor.
        agent: Agent exposing Q-values for arbitrary state indices.
        opt_q: Reference optimal Q-values with shape ``(A, S)``.
    """

    mdp: Mdp
    gamma: float
    agent: Agent[AgentState]
    opt_q: jax.Array

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class State:
        """Dynamic convergence history.

        Attributes:
            prev_q: Q-values observed during the previous evaluation.
            step: Number of completed evaluations.
        """

        prev_q: jax.Array
        step: jax.Array

    @dataclass  # pyright: ignore[reportUntypedClassDecorator]
    class Metrics:
        """Convergence diagnostics for one evaluation.

        Attributes:
            diff_l1: Mean absolute Q-value change from the previous evaluation.
            diff_linf: Maximum absolute Q-value change.
            bellman_l1: Mean absolute Bellman optimality error.
            bellman_linf: Maximum absolute Bellman optimality error.
            value_l1: Mean absolute error against optimal Q-values.
            value_linf: Maximum absolute error against optimal Q-values.
            value_norm: Relative L2 error against optimal Q-values.
            pi_eval_min: Minimum value under the current greedy policy.
            pi_eval_rho: Expected greedy-policy return from the initial state.
            pi_diff_l1: Mean absolute greedy-policy change.
            pi_diff_linf: Maximum absolute greedy-policy change.
            iteration: Number of completed evaluations.
        """

        diff_l1: jax.Array
        diff_linf: jax.Array
        bellman_l1: jax.Array
        bellman_linf: jax.Array
        value_l1: jax.Array
        value_linf: jax.Array
        value_norm: jax.Array
        pi_eval_min: jax.Array
        pi_eval_rho: jax.Array
        pi_diff_l1: jax.Array
        pi_diff_linf: jax.Array
        iteration: jax.Array

    def _q_values(self, agent_state: AgentState) -> jax.Array:
        """Read the complete Q-table through the agent protocol."""
        all_states = jnp.arange(  # pyright: ignore[reportUnknownMemberType]
            self.mdp.state_size
        )
        q_values = self.agent.q_vals(agent_state, all_states)
        chex.assert_shape(
            q_values,
            (self.mdp.action_size, self.mdp.state_size),
        )
        return q_values

    def init(self, agent_state: AgentState) -> Eval.State:
        """Initialize convergence history from an agent state.

        Args:
            agent_state: Agent state providing the initial Q-values.

        Returns:
            Initial evaluator state.
        """
        return self.State(
            prev_q=self._q_values(agent_state),
            step=jnp.zeros(  # pyright: ignore[reportUnknownMemberType]
                (), dtype=jnp.int32
            ),
        )

    def evaluate(
        self,
        state: Eval.State,
        agent_state: AgentState,
    ) -> tuple[Eval.Metrics, Eval.State]:
        """Evaluate the current agent and advance convergence history.

        Args:
            state: Current evaluator state.
            agent_state: Agent state to evaluate.

        Returns:
            Convergence metrics and the advanced evaluator state.
        """
        new_q = self._q_values(agent_state)
        chex.assert_equal_shape([new_q, state.prev_q, self.opt_q])

        non_term = (1 - self.mdp.terminal)[None, :]
        n_non_term = jnp.sum(non_term)

        diff = new_q - state.prev_q
        diff_l1 = jnp.sum(jnp.abs(diff) * non_term) / n_non_term
        diff_linf = jnp.max(jnp.abs(diff) * non_term)

        bellman_target = _bellman_optimality(self.mdp, new_q, self.gamma)
        bellman_error = new_q - bellman_target
        bellman_l1 = jnp.sum(jnp.abs(bellman_error) * non_term) / n_non_term
        bellman_linf = jnp.max(jnp.abs(bellman_error) * non_term)

        value_error = new_q - self.opt_q
        value_l1 = jnp.sum(jnp.abs(value_error) * non_term) / n_non_term
        value_linf = jnp.max(jnp.abs(value_error) * non_term)
        value_norm = cast(
            jax.Array,
            jnp.linalg.norm(value_error * non_term)
            / jnp.linalg.norm(self.opt_q * non_term),
        )

        prev_pi = _greedy_policy(state.prev_q)
        new_pi = _greedy_policy(new_q)
        pi_diff = new_pi - prev_pi
        pi_diff_l1 = jnp.sum(jnp.abs(pi_diff) * non_term) / n_non_term
        pi_diff_linf = jnp.max(jnp.abs(pi_diff) * non_term)

        greedy_q = _evaluate_policy(self.mdp, new_pi, self.gamma)
        pi_eval_min = jnp.min(greedy_q)
        pi_eval_rho = jnp.sum(self.mdp.initial * jnp.max(greedy_q, axis=0))
        next_step = state.step + 1

        return (
            Eval.Metrics(
                diff_l1=diff_l1,
                diff_linf=diff_linf,
                bellman_l1=bellman_l1,
                bellman_linf=bellman_linf,
                value_l1=value_l1,
                value_linf=value_linf,
                value_norm=value_norm,
                pi_eval_min=pi_eval_min,
                pi_eval_rho=pi_eval_rho,
                pi_diff_l1=pi_diff_l1,
                pi_diff_linf=pi_diff_linf,
                iteration=next_step,
            ),
            self.State(prev_q=new_q, step=next_step),
        )
