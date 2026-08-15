# Agent components

Agent components assemble models, reinforcement-learning heads, and action
distributions while keeping dynamic data in explicit `State` pytrees.

## Components

### Models

| Component | Role |
| --- | --- |
| `Module` | Adapts a partitioned callable pytree to the stateful transform interface. |
| `Model` | Applies a feature transform followed by a prediction head. |
| `NormModel` | Normalizes inputs before applying another model. |

### Agents

| Component | Role |
| --- | --- |
| `Pi` | Composes a shared body with a policy head. |
| `VPi` | Composes a shared body with state-value and policy heads. |
| `VQPi` | Composes a shared body with state-value, action-value, and policy heads. |

### Heads

| Component | Output |
| --- | --- |
| `VHead` | One state value, `V(s)`. |
| `QHead` | Values for every finite action, `Q(s, ·)`. |
| `QsaHead` | One state-action value, `Q(s, a)`. |
| `CategoricalHead` | A categorical policy distribution. |
| `DiagNormalHead` | A diagonal-Normal policy distribution. |

### Distributions and inference

| Component | Role |
| --- | --- |
| `Categorical` | Samples and evaluates finite actions. |
| `DiagNormal` | Samples and evaluates continuous vector actions. |
| `Evaluation` | Holds an action's log-probability and distribution entropy. |
| `VNextVInference` | Aligns current and successor values with a sampled sequence. |
| `VPiNextVInference` | Aligns current policies, current values, and successor values. |

### State helpers and interfaces

`Param`, `Partition`, `partition`, and `combine` separate trainable leaves from
the rest of a component state. `Function`, `Transform`, `Normalizer`, and
`Distribution` are the exported interfaces for compatible custom components.

## Quickstart

Use `Pi` when acting requires only a policy:

```python
from jaxtor.agent import CategoricalHead, Pi

pi = CategoricalHead(n_actions=n_actions, logits=policy_net)
agent = Pi(body=body, pi=pi)
state = agent.init(key, body=body_state, pi=pi.init(policy_net_state))

prediction, state = agent.apply(obs, state)
action, state = agent.act(obs, state)
```

Add a value head when the algorithm also predicts `V(s)`:

```python
from jaxtor.agent import CategoricalHead, VHead, VPi

agent = VPi(
    body=body,
    v=VHead(net=value_net),
    pi=CategoricalHead(n_actions=n_actions, logits=policy_net),
)

prediction, state = agent.apply(obs, state)
action, state = agent.act(obs, state)
```

## Details

### Acting and prediction

`apply` returns every configured prediction. `act` evaluates only the body and
policy dependencies required to select an action. Setting `deterministic=True`
selects the distribution mode.

### Trainable state

`Param` marks trainable leaves without changing their JAX behavior. `partition`
returns complementary trainable and frozen trees, and `combine` reconstructs
the complete state after an optimizer update.

### Sequence inference

The inference components replay an agent over observations collected by
`Roll`. Natural terminations receive a zero successor value, while truncations
and open sequence tails evaluate their true successor observation.
