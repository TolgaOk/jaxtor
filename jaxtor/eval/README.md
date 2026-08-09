# Evaluation

## Sampled episodes

`McEval` runs a fixed number of sampler steps, aggregates completed episodes,
and returns the sampler state advanced by those steps. This example assumes the
agent, vectorized Markov chain, and their initial states already exist.

```python
import jax

from jaxtor.eval import McEval
from jaxtor.sampler import Imc

imc = Imc(agent=agent, mc=vec_mc)
state = imc.init(mc=mc_state, agent=agent_state)

evaluator = McEval(imc=imc, episode_len=1_000)
evaluate = jax.jit(evaluator.evaluate)

metrics, state = evaluate(state)

print(metrics.avg_eps_rew)
print(metrics.n_episodes)

transition, state = imc.sample(state)
```

## Tabular convergence

`TabularEval` compares the current Q-table with its previous value, the Bellman
optimality target, and known optimal Q-values. The agent provides
`q_vals(agent_state, states)`, which returns a Q-table with shape `(A, S)`.

```python
import jax

from jaxtor.eval import TabularEval, optimal_q

q_star = optimal_q(mdp, gamma=0.99)

evaluator = TabularEval(
    mdp=mdp,
    gamma=0.99,
    agent=agent,
    opt_q=q_star,
)
state = evaluator.init(agent_state)
evaluate = jax.jit(evaluator.evaluate)

metrics, state = evaluate(state, updated_agent_state)

print(metrics.bellman_linf)
print(metrics.value_norm)
print(metrics.iteration)
```
