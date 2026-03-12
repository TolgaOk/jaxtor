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

## `TabularEnv` — [`jaxdp`](https://github.com/TolgaOk/jaxdp)

Tabular MDPs with exact transition matrices. Supports Garnet (random MDPs), graph, and gridworld configurations.

```python
from jaxtor.env import tabular

# GridWorld
# P: player, X: lava, @: goal, #: wall
env = tabular.gridworld.make(tabular.gridworld.Config(
    board=[
        "#####",
        "#  @#",
        "# #X#",
        "#P  #",
        "#####",
    ],
    p_slip=0.1,
))
```

