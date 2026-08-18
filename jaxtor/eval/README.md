# Evaluation

Evaluation components produce metrics while threading the evaluated state explicitly.

## Components

| Component | Role |
| --- | --- |
| `McEval` | Samples an induced Markov chain for a fixed number of steps and summarizes completed episodes. |
| `TabularEval` | Measures convergence of a tabular Q-value agent. |
| `optimal_q` | Computes reference optimal Q-values by policy iteration. |

`TabularEval` and `optimal_q` require the `env` optional dependency.

## Quickstart

```python
import jax

from jaxtor.eval import McEval

evaluator = McEval(imc=imc, episode_len=1_000)
metrics, state = jax.jit(evaluator.evaluate)(state)
```

## Details

### Sampled episodes

`McEval` advances its sampler state and reports statistics for episodes that finish during the evaluation window.
Metrics include mean, standard deviation, minimum, and maximum return, mean episode length, completed episode count, and truncation rate.

### Tabular convergence

`TabularEval` consumes an agent with `q_vals(observations, state)`.
It compares the current Q-table with its previous value, the Bellman optimality target, and known optimal Q-values.

```python
q_star = optimal_q(mdp, gamma=0.99)
evaluator = TabularEval(mdp=mdp, gamma=0.99, agent=agent, opt_q=q_star)
state = evaluator.init(agent_state)

metrics, state = evaluator.evaluate(updated_agent_state, state)
```

The metrics cover Q-value change, Bellman error, error against `q_star`, and the quality and stability of the greedy policy.
