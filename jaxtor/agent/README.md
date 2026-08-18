# Agent components

Agent components assemble models, reinforcement-learning heads, and action distributions while keeping dynamic data in explicit `State` pytrees.

Composition is structural.
A parent accepts only the child capability it uses, then nests the child states under its own `State`:

```text
Module:              array       -> features
VHead:               features    -> V(s)
CategoricalHead:     features    -> categorical distribution
VPi:                 observation -> {V(s), distribution}
Ou:                  observation -> correlated bounded action
VPiNextVInference:   sequence    -> {V(s_t), pi(s_t), V(s_{t+1})}
```

## Action-value names

The names distinguish how an action-value is represented and evaluated.

| Name | Meaning |
| --- | --- |
| `Qvec` | The finite-action vector `[Q(s, a_1), ..., Q(s, a_A)]`. |
| `Q` / `q` | An evaluated scalar `Q(s, a)` that an agent's `q` method computes from observations and actions. |
| `Qsa` | A model that takes `(s, a)` directly and returns `q`. |

`Qsa` describes the input signature.
Continuous-action critics commonly use this signature, but it does not require a continuous action space.

## Components

### Models

| Component | Requires | Produces |
| --- | --- | --- |
| `Module` | A partitioned array callable and an array input. | The callable output and parameter state. |
| `Model` | `body: In -> Feat`, `head: Feat -> Pred`, and `In`. | `Pred` and nested body/head state. |
| `NormModel` | A normalizer, `model: In -> Pred`, and `In`. | `Pred` and nested normalization/model state. |

### Agents

| Component | Requires | Produces |
| --- | --- | --- |
| `Pi` | An observation body and policy head. | A policy through `pi`, or a selected action. |
| `VPi` | An observation body, value head, and policy head. | `ValuePolicy(v, pi)` through `vpi`, a value through `v`, or a selected action. |
| `VQPi` | An observation body, value head, finite-action Q head, and policy head. | `ValueQPolicy(v, q, pi)` through `vqpi`, or a selected action. |
| `Quadratic` | An observation body plus value, location, and precision heads. | An evaluated value through `v`, `Q(s, a)` through `q`, or the maximizing action. |
| `Ou` | An action-selecting agent, key, and initial action-shaped noise. | A bounded action with temporally correlated exploration and nested state. |

### Heads

| Component | Requires | Produces |
| --- | --- | --- |
| `VHead` | Features and a transform ending in one value. | `V(s)` with the feature axis removed. |
| `QHead` | Features and a transform ending in `n_actions` values. | `Q(s, .)` over the final action axis. |
| `QsaHead` | Features, actions with matching leading axes, and a scalar-value transform. | One `Q(s, a)` per input pair. |
| `CategoricalHead` | Features and a logits transform. | A `Categorical` distribution. |
| `DiagNormalHead` | Features and location and log-scale transforms. | A `DiagNormal` distribution. |

### Distributions

| Component | Requires | Produces |
| --- | --- | --- |
| `Categorical` | Logits; a key for `sample` or action for `evaluate`. | A finite action, mode, or `Evaluation`. |
| `DiagNormal` | Location and log scale; a key or continuous action. | A vector action, mode, or `Evaluation`. |
| `Normal` | Location and positive-definite covariance; a key or action. | A vector action, mode, or `Evaluation`. |
| `Evaluation` | Log-probability and entropy arrays. | A methodless result containing `logp` and `entropy`. |

### Sequence inference

| Component | Requires | Produces |
| --- | --- | --- |
| `VPiNextVInference` | A sequence with `obs`, `nobs`, `term`, and `trun`; an agent providing `vpi` and `v`. | `Inference(v_tm1, pi_tm1, v_t)` for policy-gradient returns. |
| `QNextVInference` | A sequence with actions; an agent providing `q` and `v`. | `Inference(q_t, v_t)` for RLax off-policy returns. |

### State helpers

| Component | Requires | Produces |
| --- | --- | --- |
| `Param` | A trainable array pytree. | A trainable-leaf marker. |
| `Partition` | Parameter and frozen state-shaped trees. | A methodless pair named `params` and `frozen`. |
| `partition` | A component state containing `Param` leaves. | Complementary `params` and `frozen` state-shaped trees. |
| `combine` | Complementary parameter and frozen trees. | The reconstructed component state. |

### Capability interfaces

| Interface | Requires | Produces |
| --- | --- | --- |
| `Function` | `In`. | `Out`. |
| `Transform` | `In` and `State`. | `Out` and updated `State`. |
| `Normalizer` | A value and normalization state. | A normalized value and state; `update` produces updated state. |
| `Distribution` | A key for `sample` or action for `evaluate`. | An action, evaluation, or deterministic mode. |
| `VPiAgent` | Observations and agent state. | Values through `v`, or joint values and policies through `vpi`. |
| `QVAgent` | Observations, actions, and agent state. | Action-values through `q`, or values through `v`. |

## Quickstart

Use `Pi` when acting requires only a policy:

```python
from jaxtor.agent import CategoricalHead, Pi

pi = CategoricalHead(n_actions=n_actions, logits=policy_net)
agent = Pi(body=body, policy=pi)
state = agent.init(key, body=body_state, pi=pi.init(policy_net_state))

policy, state = agent.pi(obs, state)
action, state = agent.act(obs, state)
```

Add a value head when the algorithm also predicts `V(s)`:

```python
from jaxtor.agent import CategoricalHead, VHead, VPi

agent = VPi(
    body=body,
    value=VHead(net=value_net),
    policy=CategoricalHead(n_actions=n_actions, logits=policy_net),
)

value_policy, state = agent.vpi(obs, state)
value, state = agent.v(obs, state)
action, state = agent.act(obs, state)
```

Use `Quadratic` for a diagonal normalized advantage function:

```python
action, state = agent.act(obs, state)
value, state = agent.v(obs, state)
q, state = agent.q(obs, action, state)
```

Wrap a deterministic agent when physical control benefits from persistent exploration:

```python
from jaxtor.agent import Ou

behavior = Ou(agent=deterministic_agent, sigma=0.1)
state = behavior.init(key, jnp.zeros((n_envs, act_size)), agent_state)
action, state = behavior.act(obs, state)
```

## Details

### Semantic endpoints

Agents expose the computations their algorithms request: `pi`, `v`, `q`, or a shared endpoint such as `vpi`.
`act` evaluates only the dependencies required to select an action.
Policy agents select a distribution sample or mode, while `Quadratic` selects its maximizing action.

### Trainable state

`Param` marks trainable leaves without changing their JAX behavior.
`partition` returns complementary trainable and frozen trees, and `combine` reconstructs the complete state after an optimizer update.

### Sequence inference

The inference components replay an agent over observations collected by `Roll`.
They return every stored successor value unchanged.
The consuming RLax calculation owns terminal discounting.
