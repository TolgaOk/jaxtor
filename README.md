# jaxtor

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org)
[![JAX 0.8+](https://img.shields.io/badge/JAX-0.8%2B-green)](https://github.com/jax-ml/jax)
[![version](https://img.shields.io/badge/version-0.2.0--alpha-orange)](https://github.com/TolgaOk/jaxtor)

JAX-based **sampling** and **evaluation** components over unified **environment** interface for building reinforcement learning pipelines.

![banner](doc/banner.svg)

> **Compositionality over inheritance.** Components are `@dataclass` namespaces of pure functions, stateless by design, with states defined and passed explicitly.

> **Protocols over concrete types.** Each component is self-contained and declares its own `Protocol` for what it needs, with no internal imports between components. Any component that satisfies the protocol is a valid input.

## Installation

Install it via [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                          # core (jax, chex, jaxtyping)
uv sync --group env              # + gymnasium, gymnax, jaxdp, distrax
uv sync --group example          # + equinox, gymnax, rlax, tyro, rich
uv sync --group dev              # + pytest, pyright, ruff
```


## Components

`jaxtor` unifies several environment libraries and provides components for sampling and evaluation.

```python
roll = Roll(                                # Rollout sampler
    seqlen=2048,
    imc=Imc(                                # Induced Markov chain
        agent=agent,                        # Behavior policy
        mc=Mc(                              # Markov chain
            max_episode_len=1000,
            queue_size=10,                  # queue for episode statistics
            env=gymnax.make("CartPole-v1"),
        ),
    ),
)
roll_state = roll.imc.init(...)             # contains the sampler state
trans, roll_state = jax.jit(roll.sample)(roll_state)

>>> trans.obs.shape
(2048, 8)
```

### Environments

- `GymEnv` — [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) adapter. Wraps CPU-based gym envs (MuJoCo, Atari, classic control) and makes them `jit`/`vmap` compatible via `io_callback` + `custom_vmap`.
- `GymnaxEnv` — [`gymnax`](https://github.com/RobertTLange/gymnax) adapter. Pure-JAX environments, fully `jit`/`vmap`/`grad`-compatible.
- `TabularEnv` — [`jaxdp`](https://github.com/TolgaOk/jaxdp) adapter. Tabular MDPs with exact transition matrices.

(see [`README`](jaxtor/env/README.md) for details)

### Samplers

- `Mc` — Markov chain sampler with episode statistics tracking
- `VecMc` — vectorized parallel MC sampler
- `Imc` — induced Markov chain (agent + `Mc`)
- `MuImc` — induced Markov chain with behavior log-probability tracking
- `Roll` — N-step rollout sampler
- `Sweep` — sweep over all (s,a) pairs (for tabular MDPs)
- `ExpSweep` — exact sweep over all (s,a) pairs (for tabular MDPs)

(see [`README`](jaxtor/sampler/README.md) for details)

### Evaluation

- `McEval` — Markov chain evaluation (sampled)
- `TabularEval` — tabular MDP evaluation (exact) using `jaxdp`

## Examples

Single-script implementations of common RL algorithms.

- [`examples/q_learning.py`](examples/q_learning.py) — tabular Q-learning on Garnet MDP (`jaxdp`)
- [`examples/reinforce.py`](examples/reinforce.py) — REINFORCE on CartPole-v1 (`gymnax`)
- [`examples/naf.py`](examples/naf.py) — [NAF](https://arxiv.org/abs/1603.00748) on MuJoCo (`gymnasium`)
- [`examples/ppo.py`](examples/ppo.py) — [PPO](https://arxiv.org/abs/1707.06347) on MuJoCo (`gymnasium`)
