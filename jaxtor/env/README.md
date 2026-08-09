# Environment Adapters

Three backends with the same `Env` protocol

## `GymEnv` — [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium)

Wraps CPU-based gym envs (MuJoCo, Atari, classic control) and makes them `jit`/`vmap` compatible via `io_callback` + `custom_vmap`, so they work with the rest of the `jaxtor` components out of the box. Omitting `num_envs` creates a scalar environment for `Mc`. Passing a positive integer, including one, creates a sync or async vector runtime for `VecMc`. However, we gain no JIT speedup or `grad` compatibility, since the envs run outside JAX, unlike gymnasium's own [JAX support](https://gymnasium.farama.org/main/api/functional/), which requires reimplementing envs in pure JAX.

```python
from jaxtor.env import gymnasium

# A numeric size allocates a vector runtime for VecMc.
env = gymnasium.make("Hopper-v5", num_envs=16, async_envs=True)
```

## `MjxEnv` — [`mujoco-mjx`](https://mujoco.readthedocs.io/en/stable/mjx.html)

GPU-native MuJoCo (MJX): pure-JAX, `jit`/`vmap`/`grad`-compatible, on-device. Reuses the exact Gymnasium v5 XML and obs/reward/termination, so it matches `gymnasium.make(name)` **one-to-one** (machine precision under `float64`).

```python
from jaxtor.env import mjx

env = mjx.make("Hopper-v5")                          # MJX-JAX (XLA), default
env = mjx.make("Hopper-v5", impl="warp", nconmax=..., njmax=...)  # NVIDIA Warp
```

Supported: `Hopper-v5`, `Walker2d-v5`, `HalfCheetah-v5`, `Swimmer-v5`.

**Excluded (Ant/Humanoid):** MJX's collision algorithm differs from CPU MuJoCo, so 3D multi-contact dynamics — and `cfrc_ext` contact forces — diverge from Gymnasium; one-to-one parity is impossible. Gymnasium's own [MJX port](https://github.com/Farama-Foundation/Gymnasium/pull/834) stalled here too.

**Warp backend:** on NVIDIA GPUs, `impl="warp"` selects [MuJoCo Warp](https://mujoco.readthedocs.io/en/latest/mjwarp/) — faster on contact-rich scenes, but not differentiable, not parity-tested, and requires `mujoco_warp` + CUDA.

## `GymnaxEnv` — [`gymnax`](https://github.com/RobertTLange/gymnax)

Pure-JAX environments, fully `jit`/`vmap`/`grad`-compatible. All computation stays on-device.

```python
from jaxtor.env import gymnax

env = gymnax.make("CartPole-v1")
```

## `tabular` — [`jaxdp`](https://github.com/TolgaOk/jaxdp)

Tabular MDPs with exact transition matrices.

### Pre-defined environments

```python
from jaxtor.env import tabular

env = tabular.make("mid-garnet")      # 50S, 10A random MDP
env = tabular.make("graph")           # 6-state graph (Fastest Convergence for Q-Learning)
env = tabular.make("cliffworld")      # right-side cliff gridworld
env = tabular.make("cliff-walking")   # Sutton & Barto classic cliff walking
env = tabular.make("four-rooms")      # Sutton, Precup, Singh 1999
env = tabular.make("frozen-lake")     # 4x4 frozen lake with p_slip=1/3
```

### Custom configurations

```python
env = tabular.garnet.make(tabular.garnet.Config(state_size=100, action_size=4))
env = tabular.graph.make(tabular.graph.Config(max_episode_len=500))
env = tabular.gridworld.make(tabular.gridworld.Config(
    board=(
        "#####",
        "#  @#",
        "# #X#",
        "#P  #",
        "#####",
    ),
    p_slip=0.1,
))
```

Board characters: `#` wall, `P` start, `@` terminal goal, `X` penalty, `+` reward, `=` absorbing, ` ` passable.
