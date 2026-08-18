# Component design rules

## Functional components

- A component is a configured dataclass containing its fixed parameters and
  child components.
- Dynamic values live in a nested `State` dataclass. State is passed explicitly
  and returned as a new pytree instead of being mutated in place.
- Component methods remain compatible with JAX transformations when their
  inputs and state contain JAX-compatible values.

## Independent modules

- Each component module depends on behavior rather than a concrete sibling
  implementation.
- A component exposes behavior required by its own role. Composition-specific
  data belongs to the composing component.
- The consuming module defines a small local `Protocol` containing only the
  methods and state fields it needs.
- Protocols stay compact: one-line purpose, required fields, and required
  signatures only. Member behavior belongs in the consuming component's docs.
- Dataclass-backed protocol data is declared as annotated fields.
- Structural protocols keep component files independently usable and avoid
  coupling their implementations.

## Common API

- `init(...) -> State` creates component state when initialization is needed.
- The main method uses its domain verb, such as `step`, `sample`, `evaluate`, or
  `update`.
- State is the final argument when other inputs are present.
- Methods that produce data and advance state return `(data, state)`. Methods
  that only advance state return `state`.

```python
state = component.init(...)
output, state = component.sample(input, state)
```

## Composition

- Inline a child component when only its parent uses it.
- Give a child a direct name when the caller uses its API, shares it, or manages
  its lifecycle. Use that name instead of navigating through its parent.

## Dimensions

- Public computation boundaries check expected ranks, shapes, and shared leading
  dimensions with Chex assertions.
- Shape checks describe the component contract close to the operation that
  requires it.
