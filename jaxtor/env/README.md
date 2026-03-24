# Environment Adapters

Three backends with the same `Env` protocol

## `GymEnv` — [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium)

Wraps CPU-based gym envs (MuJoCo, Atari, classic control) and makes them `jit`/`vmap` compatible via `io_callback` + `custom_vmap`, so they work with the rest of the `jaxtor` components out of the box. Supports sync and async vectorized environments. However, we gain no JIT speedup or `grad` compatibility, since the envs run outside JAX, unlike gymnasium's own [JAX support](https://gymnasium.farama.org/main/api/functional/), which requires reimplementing envs in pure JAX.

```python
from jaxtor.env import gymnasium

# Unlike JAX native envs, we need to specify the number of envs upfront.
env = gymnasium.make("Hopper-v5", num_envs=16, async_envs=True)
```

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

