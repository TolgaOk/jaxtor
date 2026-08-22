"""Evaluation utilities.

Components:
    McEval: Sampling-based evaluator with batched environment support.
    TabularEval: Convergence diagnostics for tabular Q-values
        (requires ``jaxtor[env]``).
    optimal_q: Optimal Q-values via policy iteration
        (requires ``jaxtor[env]``).
"""

from jaxtor.eval.mc import Eval as McEval

__all__ = ["McEval"]

try:
    from jaxtor.eval.tabular import Eval as TabularEval, optimal_q

    __all__ += ["TabularEval", "optimal_q"]
except ImportError:
    pass
