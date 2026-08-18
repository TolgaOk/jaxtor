# Utilities

Utility components provide explicit state for normalization, running statistics, and minibatch preparation.

## Components

| Component | Role |
| --- | --- |
| `RunningStats` | Tracks running mean and variance with the Welford algorithm. |
| `ObsNorm` | Normalizes observations and updates their statistics explicitly. |
| `RewardNorm` | Normalizes rewards using the variance of rolling discounted returns. |
| `Minibatches` | Shuffles aligned pytrees and splits them into equal-sized minibatches. |

## Quickstart

```python
from jaxtor.util import Minibatches, ObsNorm, RewardNorm, RunningStats

obs_norm = ObsNorm(stats=RunningStats(clip=10.0))
reward_norm = RewardNorm(gamma=0.99, rms=RunningStats(), seq_axis=1)
minibatches = Minibatches(count=4, sample_ndim=2)
```

## Details

### Observation statistics

`ObsNorm.apply` reads the current statistics.
`ObsNorm.update` adds new observations at the algorithm boundary, so applying a model does not silently change its normalization state.

```python
obs_state = obs_norm.update(observations, obs_state)
normalized, obs_state = obs_norm.apply(observations, obs_state)
```

### Reward normalization

`RewardNorm` tracks rolling discounted returns independently for each environment lane and normalizes rewards by their running standard deviation.
Termination and truncation flags reset the corresponding return carries.

```python
rewards, reward_state = reward_norm.update(
    rewards,
    termination | truncation,
    reward_state,
)
```

### Minibatches

`Minibatches.shuffle` collapses the configured leading sample axes, applies one shared permutation to every pytree leaf, and returns `(minibatch, sample, ...)` arrays.

```python
batches = minibatches.shuffle(key, batch)
```
