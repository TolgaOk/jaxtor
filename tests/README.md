# Tests

Each `test_<component>.py` file owns one component's numerical, state, shape,
and transformation contracts. `test_composition.py` checks behavior and work
across component boundaries. `test_ppo_example.py` checks import safety, seeded
replay across configurations, and the real PPO update against an independent
TD(λ) recurrence and RLax clipped loss. It requires the example dependencies.

- `backend`: optional third-party environment adapters
- `integration`: multiple Jaxtor components composed end to end
- `statistical`: distributional checks with larger samples

The lightweight component suite is:

```sh
pytest -m "not backend and not statistical"
```
