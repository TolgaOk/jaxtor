# Environment adapters

Every adapter exposes the same `Env` protocol and can be passed to sampling components such as `Mc`.

## Components

| Module | Component and options | Backend |
| --- | --- | --- |
| `gymnasium` | `GymEnv`; registered environments through `make`, custom runtimes through `from_factory`, optional `async_envs=True` | Host CPU through `io_callback` |
| `envpool` | `GymEnv`; any environment supported by the installed EnvPool version | Vectorized host CPU |
| `gymnax` | `GymnaxEnv`; any environment supported by the installed Gymnax version | Pure JAX |
| `mjx` | `MjxEnv`; `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Swimmer-v5` | MJX with XLA or NVIDIA Warp |
| `tabular` | `TabularEnv`; `mid-garnet`, `graph`, `cliffworld`, `cliff-walking`, `four-rooms`, `frozen-lake` | Exact tabular JAX MDPs |

## Quickstart

```python
from jaxtor.env import gymnax
from jaxtor.sampler import Mc

env = gymnax.make("CartPole-v1")
mc = Mc(max_eps_len=500, env=env)
```

## Details

### Common interface

`init` creates backend state, `reset` begins an episode, `step` returns the next observation, reward, termination, and truncation, and `obs` reads the current observation.
Episode resets and time limits are owned by `Mc`.

### Gymnasium and EnvPool

`gymnasium.make` creates a configured `GymEnv`.
Executing `init` creates its process-local runtime.
Mapping `init` with `jax.vmap` creates one runtime whose capacity matches the complete key batch.
Call `env.close(state)` when the runtime is no longer needed and outside JAX transformations.

Gymnasium and EnvPool execute on the host.
JIT allows them to compose with JAX code, but it does not provide device acceleration or gradients through their environment steps.
EnvPool always creates a vector runtime, and its native autoreset behavior is adapted so `Mc` retains episode ownership.

### Pure-JAX environments

Gymnax environments support `jit`, `vmap`, and gradients directly.
MJX uses the Gymnasium v5 XML assets and matching observation, reward, and termination definitions for its four supported locomotion environments.

MJX uses its JAX/XLA backend by default.
`impl="warp"` selects MuJoCo Warp on NVIDIA GPUs and accepts `nconmax` and `njmax`.
The Warp path is not differentiable or parity-tested.
Ant and Humanoid are excluded because MJX does not reproduce their CPU MuJoCo contact dynamics and contact-force observations.

### Tabular environments

`tabular.make(name)` creates one of the named environments in the component table.
Custom configurations are available through `tabular.garnet.make(config)`, `tabular.graph.make(config)`, and `tabular.gridworld.make(config)`.

Gridworld boards use `#` for walls, `P` for starts, `@` for terminal goals, `X` for penalties, `+` for rewards, `=` for absorbing cells, and spaces for passable cells.
