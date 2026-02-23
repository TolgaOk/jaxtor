# `jaxtor`

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org)
[![JAX 0.8+](https://img.shields.io/badge/JAX-0.8%2B-green)](https://github.com/jax-ml/jax)
[![version](https://img.shields.io/badge/version-0.2.0--alpha-orange)](https://github.com/TolgaOk/jaxtor)

JAX-based sampling and evaluation components for building reinforcement learning pipelines.

> **Compositionality over inheritance.** Components are `@dataclass` namespaces of pure functions, stateless by design, with states defined and passed explicitly.

> **Protocols over concrete types.** Each component is self-contained and declares its own `Protocol` for what it needs, with no internal imports between components. Any component that satisfies the protocol is a valid input.

### Installation

```bash
uv sync                          # core (jax, chex, jaxtyping)
uv sync --group env              # + gymnasium, gymnax, jaxdp, distrax
uv sync --group dev              # + pytest, pyright, ruff
uv sync --group example          # + equinox, gymnax, rlax, tyro, rich
```


### Components

`jaxtor` unifies several environment libraries and provides components for sampling and evaluation.

**example**
```python
roll = Roll(                         # Rollout sampler
    seqlen=2048,
    seq_axis=1,
    imc=Imc(                         # Induced Markov chain
        agent=agent,                 # Behavior policy
        mc=Mc(                       # Markov chain
            max_episode_len=1000,
            queue_size=10,
            env=gymnax.make("CartPole-v1"),
        ),
    ),
)
# `roll_state` contains the agent state, env state, and episode statistics
roll_state = roll.imc.init(...)
transitions, roll_state = jax.jit(roll.sample)(roll_state)  # (2048, ...)
```

**Environments**

- `GymEnv` — [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) adapter. 
> Wraps existing CPU-based gym envs and makes them `jit`/`vmap` compatible via `io_callback` + `custom_vmap`, so they work with rest of the `jaxtor` components out of the box. However, we gain no JIT speedup since the envs run outside JAX, unlike gymnasium's own [JAX support](https://gymnasium.farama.org/main/api/functional/) which requires reimplementing envs in pure JAX.

```python
env = gymnasium.make("Hopper-v5", num_envs=4, async_envs=True)
states = jax.vmap(env.init)(keys)
step = jax.jit(jax.vmap(env.step))
*transition, states = step(keys, acts, states)
```

- `GymnaxEnv` — [`gymnax`](https://github.com/RobertTLange/gymnax) adapter
- `TabularEnv` —  [`jaxdp`](https://github.com/TolgaOk/jaxdp) adapter for tabular MDPs

**Samplers** (see [`README`](jaxtor/sampler/README.md) for details)
- `Mc` — Markov chain sampler with episode statistics tracking
- `VecMc` — vectorized parallel MC sampler
- `Imc` — induced Markov chain (agent + `Mc`)
- `Roll` — N-step rollout sampler
- `Sweep` — sweep over all (s,a) pairs (for tabular MDPs)
- `ExpSweep` — exact sweep over all (s,a) pairs (for tabular MDPs)

**Evaluation**
- `McEval` — Markov chain evaluation (sampled)
- `TabularEval` — tabular MDP evaluation (exact) using `jaxdp`

### Full Examples

- [`examples/q_learning.py`](examples/q_learning.py) — tabular Q-learning on Garnet MDP
- [`examples/reinforce.py`](examples/reinforce.py) — REINFORCE on CartPole-v1
