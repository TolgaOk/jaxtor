# jaxtor

JAX-based sampling and evaluation utilities for reinforcement learning environments.

## Installation

```bash
uv sync                          # core (jax, chex, jaxtyping, jaxdp)
uv sync --group dev              # + pytest, pyright, ruff
uv sync --group example          # + equinox, gymnax, rlax, tyro, rich
```

## Components

### Environments (`jaxtor.env`)
- `GymnaxEnv` — [`gymnax`](https://github.com/RobertTLange/gymnax)  environment adapter (no auto-reset)
- `TabularEnv` — tabular MDPs

### Samplers (`jaxtor.sampler`)
- `Mc` — Markov chain sampler with episode statistics tracking
- `VecMc` — vectorized parallel MC sampler
- `Imc` — induced Markov chain (agent + `Mc`)
- `Roll` — N-step rollout sampler
- `Sweep` — sweep over all (s,a) pairs (for tabular MDPs)
- `ExpSweep` — exact sweep over all (s,a) pairs (for tabular MDPs)

### Evaluation (`jaxtor.eval`)
- `McEval` — Markov chain evaluation (sampled)
- `TabularEval` — tabular MDP evaluation (exact)
## Examples

- [`examples/q_learning.py`](examples/q_learning.py) — tabular Q-learning on Garnet MDP
- [`examples/reinforce.py`](examples/reinforce.py) — REINFORCE on CartPole-v1
