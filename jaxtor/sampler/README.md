# Sampling

Sampling components compose environments, agents, sequences, and episode statistics through small stateful interfaces.

## Components

| Component | Role |
| --- | --- |
| `Mc` | Samples an open Markov chain and owns resets and episode limits. |
| `VecMc` | Vectorizes one `Mc` over independent environment states. |
| `Imc` | Forms an induced Markov chain by selecting an agent action before each step. |
| `Roll` | Stacks samples from any stateful sampler into a fixed-length sequence. |
| `EpisodeStats` | Accumulates completed-episode statistics from sequences. |
| `Sweep` | Samples every state-action pair of a tabular Markov chain. |
| `ExpSweep` | Propagates tabular values or occupancy exactly for multiple steps. |

## Quickstart

```python
import jax

from jaxtor.sampler import Imc, Mc, Roll

mc = Mc(max_eps_len=1_000, env=env)
imc = Imc(agent=agent, mc=mc)
roll = Roll(imc=imc, seq_len=2_048)

sequence, state = jax.jit(roll.sample)(state)
```

## Details

### Open and induced Markov chains

`Mc` accepts an action and returns an aligned `obs`, `act`, `rew`, `term`, `trun`, and `nobs` transition.
`VecMc` applies the same interface to independent environment lanes.
`Imc` supplies actions through `agent.act` and returns the underlying transition directly.

For vectorized rollouts, set `seq_axis=1` on `Roll` to produce arrays shaped `(environment, time, ...)`.

### Episode statistics

Partial episodes remain in state across rollouts.
`drain` returns statistics for completed episodes and clears only the completed-episode accumulators.

```python
stats = EpisodeStats(seq_axis=1)
stats_state = stats.update(sequence, stats_state)
metrics, stats_state = stats.drain(stats_state)
```

### Exact and exhaustive tabular sampling

```python
transitions, states = Sweep(mc=mc).sample(key, env_state)

exp = ExpSweep(n_step=5)
q_sequence = exp.backward(q, mdp, policy)
occupancy_sequence = exp.forward(occupancy, mdp, policy)
```
