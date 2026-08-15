# Agent components

Agent components assemble models, reinforcement-learning heads, and action
distributions while keeping dynamic data in explicit `State` pytrees.

Composition is structural. A parent accepts only the child capability it uses,
then nests the child states under its own `State`:

```text
Module:              array       -> features
VHead:               features    -> V(s)
CategoricalHead:     features    -> categorical distribution
VPi:                 observation -> {V(s), distribution}
VPiNextVInference:   sequence    -> {V(s_t), pi(s_t), V(s_{t+1})}
```

## Action-value names

The names distinguish how an action-value is represented and evaluated.

| Name | Meaning |
| --- | --- |
| `Qfn` | A state-bound function `a -> Q(s, a)`, exposed through `evaluate(act)`. |
| `Qvec` | The finite-action vector `[Q(s, a_1), ..., Q(s, a_A)]`. |
| `Q` / `q` | An evaluated scalar `Q(s, a)`. Lowercase `q` is used for values. |
| `Qsa` | A model that takes `(s, a)` directly and returns `q`. |

`Qsa` describes the input signature. Continuous-action critics commonly use
this signature, but it does not require a continuous action space.

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
| `Pi` | An observation body and policy head. | `Pred(pi)` or a selected action. |
| `VPi` | An observation body, value head, and policy head. | `Pred(v, pi)` or a selected action. |
| `VQPi` | An observation body, value head, finite-action Q head, and policy head. | `Pred(v, q, pi)` or a selected action. |
| `Naf` | An observation body plus value, location, and precision heads. | `Pred(v, qfn, pi)` or a selected action. |

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
| `VNextVInference` | A sequence with `obs`, `nobs`, `term`, and `trun`; an agent whose prediction has `v`. | `Inference(v_tm1, v_t)` aligned to the sequence. |
| `VPiNextVInference` | The same sequence; an agent whose prediction has `v` and `pi`. | `Inference(v_tm1, pi_tm1, v_t)` aligned to the sequence. |
| `QfnVnextInference` | A sequence with actions; an agent whose prediction has `v` and `qfn`. | `Inference(q_t, v_t)` aligned for RLax off-policy returns. |

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

Use `Naf` for a diagonal normalized advantage function:

```python
action, state = naf.act(obs, state)
pred, state = naf.apply(obs, state)
q = pred.qfn.evaluate(action)
logp = pred.pi.evaluate(action).logp
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
