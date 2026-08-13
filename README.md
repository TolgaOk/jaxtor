# jaxtor

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org)
[![JAX 0.8+](https://img.shields.io/badge/JAX-0.8%2B-green)](https://github.com/jax-ml/jax)
[![version](https://img.shields.io/badge/version-0.2.0-orange)](https://github.com/TolgaOk/jaxtor)

Composable components for building reinforcement learning algorithms in <img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" height="16" alt="JAX" style="vertical-align: middle">.

![Jaxtor component banner](doc/banner.svg)

## Installation

Install either core, with environment adapters (`env`), or with example dependencies (`example`).

```bash
uv add "jaxtor @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
uv add "jaxtor[env] @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
uv add "jaxtor[example] @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
```


## Usage

Components nest directly. Connect an environment, sampler, and agent, then
transform the resulting operation with JAX.

```python
roll = Roll(                                # fixed-length collector
    seq_len=2048,
    imc=Imc(                                # agent + sampler interaction (Induced MC)
        agent=agent,
        mc=Mc(                              # episode lifecycle (Markov Chain)
            max_eps_len=1000,
            env=gymnax.make("CartPole-v1"),
        ),
    ),
)
roll_state = roll.imc.init(...)
seq, roll_state = jax.jit(roll.sample)(roll_state)

>>> seq.obs.shape
(2048, ...)
```

The construction exposes the dependency tree. Each component owns a narrow
role and declares the smallest protocol it consumes, so compatible
implementations can be substituted without changing their users. Dynamic data
lives in explicit state pytrees, while the algorithm loop remains in user code.

## Components

| Area | Components |
| --- | --- |
| Environments | `tabular`, `gymnax`, `gymnasium`, `envpool`, `mjx` |
| Agents | `Module`, `Model`, `NormModel`, semantic heads, `VPi`, `VQPi` |
| Distributions | `Categorical`, `DiagNormal`, `Draw`, `Mode` |
| Sampling | `Mc`, `VecMc`, `Imc`, `Roll`, `LoadedRoll`, `EpisodeStats` |
| Estimation | `TDEst` |
| Utilities | `Minibatches`, `ObsNorm`, `RewardNorm`, `RunningStats` |
| Evaluation | sampled-episode and exact tabular evaluators |

See the [component architecture](doc/components.md), [component design
rules](doc/component-design.md), [environment adapters](jaxtor/env/README.md),
[samplers](jaxtor/sampler/README.md), and [evaluation](jaxtor/eval/README.md).

## Examples

Single-script implementations of common RL algorithms.

- [`examples/q_learning.py`](examples/q_learning.py) — tabular Q-learning on Garnet MDP (`jaxdp`)
- [`examples/reinforce.py`](examples/reinforce.py) — REINFORCE on CartPole-v1 (`gymnax`)
- [`examples/naf.py`](examples/naf.py) — [NAF](https://arxiv.org/abs/1603.00748) on continuous-control environments (`gymnasium` or `envpool`)
- [`examples/ppo.py`](examples/ppo.py) — [PPO](https://arxiv.org/abs/1707.06347) on CartPole-v1 (`gymnax`)
