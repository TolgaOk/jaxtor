"""Evaluation utilities.

Components:
    McEval: Sampling-based evaluator with batched environment support.
    TabularEval: Convergence diagnostics for tabular value-learning agents.
    optimal_q: Optimal Q-values via policy iteration.
"""

from jaxtor.eval.mc import Eval as McEval
from jaxtor.eval.tabular import Eval as TabularEval, optimal_q

__all__ = ["McEval", "TabularEval", "optimal_q"]
