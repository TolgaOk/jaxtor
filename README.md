# jaxtor

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org)
[![JAX 0.8+](https://img.shields.io/badge/JAX-0.8%2B-green)](https://github.com/jax-ml/jax)
[![version](https://img.shields.io/badge/version-0.2.0-orange)](https://github.com/TolgaOk/jaxtor)

Composable **sampling** and **evaluation** components for reinforcement learning in <img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" height="16" alt="JAX" style="vertical-align: middle">.

![banner](doc/banner.svg)

> **Compositionality over inheritance**: Components are `@dataclass` namespaces of pure functions with explicit state.

> **Protocols over concrete types**: Each component declares its own protocol; any object satisfying it has a valid type.

## Installation

Install either core, with environment adapters (`env`), or with example dependencies (`example`).

```bash
uv add "jaxtor @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
uv add "jaxtor[env] @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
uv add "jaxtor[example] @ git+https://github.com/TolgaOk/jaxtor@v0.2.0"
```


## Components

Components nest bottom-up. Plug together an env, sampler, and agent, then `jit` the whole thing:

```python
roll = Roll(                                # N-step rollout collector
    seqlen=2048,
    imc=Imc(                                # agent + sampler interaction (Induced MC)
        agent=agent,
        mc=Mc(                              # episode lifecycle (Markov Chain)
            max_episode_len=1000,
            env=gymnax.make("CartPole-v1"),
        ),
    ),
)
roll_state = roll.imc.init(...)
trans, roll_state = jax.jit(roll.sample)(roll_state)

>>> trans.obs.shape
(2048, ...)
```

Component reference. All environments expose the same unified protocol; swap `gymnax` for `gymnasium` and every sampler, evaluator, and JAX transform works unchanged.

<table>
  <tr>
    <th>Layer</th><th>Components</th><th>Description</th><th>Dependencies</th>
  </tr>
  <tr>
    <td rowspan="4"><a href="jaxtor/env/README.md"><b>Environments</b></a></td>
    <td><code>tabular</code></td>
    <td>Index-based MDPs</td>
    <td><a href="https://github.com/TolgaOk/jaxdp">jaxdp</a></td>
  </tr>
  <tr>
    <td><code>gymnax</code></td>
    <td>Pure-JAX envs</td>
    <td><a href="https://github.com/RobertTLange/gymnax">gymnax</a></td>
  </tr>
  <tr>
    <td><code>mjx</code></td>
    <td>GPU-native MuJoCo (v5 locomotion), one-to-one with Gymnasium</td>
    <td><a href="https://mujoco.readthedocs.io/en/stable/mjx.html">mujoco-mjx</a></td>
  </tr>
  <tr>
    <td><code>gymnasium</code></td>
    <td>CPU envs (MuJoCo, Atari) via io_callback</td>
    <td><a href="https://github.com/Farama-Foundation/Gymnasium">gymnasium</a></td>
  </tr>
  <tr>
    <td rowspan="5"><a href="jaxtor/sampler/README.md"><b>Samplers</b></a></td>
    <td><code>Mc</code> · <code>VecMc</code></td>
    <td>Single-env and parallel episode sampler</td>
    <td>—</td>
  </tr>
  <tr>
    <td><code>Imc</code></td>
    <td>Wires agent to Mc</td>
    <td>—</td>
  </tr>
  <tr>
    <td><code>MuImc</code></td>
    <td>Imc + behavior log-prob</td>
    <td>—</td>
  </tr>
  <tr>
    <td><code>Roll</code></td>
    <td>N-step trajectory via scan</td>
    <td>—</td>
  </tr>
  <tr>
    <td><code>Sweep</code> · <code>ExpSweep</code></td>
    <td>All (s,a) pairs — stochastic / exact</td>
    <td><a href="https://github.com/TolgaOk/jaxdp">jaxdp</a></td>
  </tr>
  <tr>
    <td rowspan="2"><b>Evaluation</b></td>
    <td><code>McEval</code></td>
    <td>Episode stats from rollouts</td>
    <td>—</td>
  </tr>
  <tr>
    <td><code>TabularEval</code></td>
    <td>Exact convergence diagnostics</td>
    <td><a href="https://github.com/TolgaOk/jaxdp">jaxdp</a></td>
  </tr>
</table>

## Examples

Single-script implementations of common RL algorithms.

- [`examples/q_learning.py`](examples/q_learning.py) — tabular Q-learning on Garnet MDP (`jaxdp`)
- [`examples/reinforce.py`](examples/reinforce.py) — REINFORCE on CartPole-v1 (`gymnax`)
- [`examples/naf.py`](examples/naf.py) — [NAF](https://arxiv.org/abs/1603.00748) on MuJoCo (`gymnasium`)
- [`examples/ppo.py`](examples/ppo.py) — [PPO](https://arxiv.org/abs/1707.06347) on MuJoCo (`gymnasium`)
