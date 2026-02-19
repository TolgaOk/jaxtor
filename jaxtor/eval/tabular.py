"""Tabular evaluation metrics.

Computes convergence diagnostics for tabular value-learning agents by comparing
learned Q-values against the Bellman optimality target and the true optimal values.

Classes:
    Eval: Evaluator for tabular agents with full convergence metrics.

Example:
    >>> evaluator = Eval(mdp=mdp, gamma=0.99, agent=agent)
    >>> state = Eval.State(prev_agent=agent_state, step=0)
    >>> state, metrics = evaluator.metric(state, opt_q, agent_state)
"""

from __future__ import annotations

from typing import Protocol

import chex
import jax.numpy as jnp
from chex import dataclass
from jaxdp.base import bellman_optimality_operator as bellman_op
from jaxdp.base import greedy_policy, policy_evaluation
from jaxdp.mdp import MDP


class Agent(Protocol):
    class State(Protocol): ...

    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array: ...


@dataclass
class Eval:
    """Evaluator for tabular value-learning agents.

    Compares learned Q-values against the Bellman optimality target and the true
    optimal values, tracking both value and policy convergence across iterations.

    Attributes:
        mdp: Tabular MDP instance.
        gamma: Discount factor.
        agent: Agent following the Agent protocol.
    """

    mdp: MDP
    gamma: chex.Numeric
    agent: Agent

    @dataclass
    class State:
        """State of the evaluator.

        Attributes:
            prev_agent: Agent state from the previous evaluation step.
            step: Current evaluation iteration count.
        """

        prev_agent: Agent.State
        step: int

    @dataclass
    class Metrics:
        """Convergence diagnostics for a single evaluation step.

        Attributes:
            diff_l1: Mean absolute Q-value change from previous step.
            diff_linf: Max absolute Q-value change from previous step.
            bellman_l1: Mean absolute Bellman optimality error.
            bellman_linf: Max absolute Bellman optimality error.
            value_l1: Mean absolute error against optimal Q-values.
            value_linf: Max absolute error against optimal Q-values.
            value_norm: Relative L2 error against optimal Q-values.
            pi_eval_min: Minimum value under the current greedy policy.
            pi_eval_rho: Expected return of the greedy policy from the initial state.
            pi_diff_l1: Mean absolute greedy policy change from previous step.
            pi_diff_linf: Max absolute greedy policy change from previous step.
            iteration: Current evaluation iteration.
        """

        diff_l1: chex.Array
        diff_linf: chex.Array
        bellman_l1: chex.Array
        bellman_linf: chex.Array
        value_l1: chex.Array
        value_linf: chex.Array
        value_norm: chex.Array
        pi_eval_min: chex.Array
        pi_eval_rho: chex.Array
        pi_diff_l1: chex.Array
        pi_diff_linf: chex.Array
        iteration: chex.Array

    def metric(
        self,
        state: Eval.State,
        opt_q: chex.Array,
        agent_state: Agent.State,
    ) -> tuple[Eval.State, Eval.Metrics]:
        """Compute convergence metrics for the current agent state.

        Args:
            state: Current evaluator state.
            opt_q: Optimal Q-values for the MDP.
            agent_state: Current agent state to evaluate.

        Returns:
            Updated evaluator state and computed metrics.
        """
        all_states = jnp.arange(self.mdp.state_size)
        new_q = self.agent.q_vals(agent_state, all_states)
        prev_q = self.agent.q_vals(state.prev_agent, all_states)

        non_term = (1 - self.mdp.terminal)[None, :]  # (1, S)
        n_non_term = jnp.sum(non_term)

        diff = new_q - prev_q
        diff_l1 = jnp.sum(jnp.abs(diff) * non_term) / n_non_term
        diff_linf = jnp.max(jnp.abs(diff) * non_term)

        bellman_target = bellman_op.q(self.mdp, new_q, self.gamma)
        bellman_error = new_q - bellman_target
        bellman_l1 = jnp.sum(jnp.abs(bellman_error) * non_term) / n_non_term
        bellman_linf = jnp.max(jnp.abs(bellman_error) * non_term)

        value_error = new_q - opt_q
        value_l1 = jnp.sum(jnp.abs(value_error) * non_term) / n_non_term
        value_linf = jnp.max(jnp.abs(value_error) * non_term)
        value_norm = jnp.linalg.norm(value_error * non_term) / jnp.linalg.norm(
            opt_q * non_term
        )

        prev_pi = greedy_policy.q(prev_q)
        new_pi = greedy_policy.q(new_q)
        pi_diff = new_pi - prev_pi
        pi_diff_l1 = jnp.sum(jnp.abs(pi_diff) * non_term) / n_non_term
        pi_diff_linf = jnp.max(jnp.abs(pi_diff) * non_term)

        greedy_q = policy_evaluation.q(self.mdp, new_pi, self.gamma)
        pi_eval_min = jnp.min(greedy_q)

        greedy_v = jnp.max(greedy_q, axis=0)
        pi_eval_rho = jnp.sum(self.mdp.initial * greedy_v)

        return (
            state.replace(step=state.step + 1, prev_agent=agent_state),
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
                iteration=state.step + 1,
            ),
        )
