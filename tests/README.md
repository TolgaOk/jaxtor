# Tests

Each `test_<component>.py` file owns one component's numerical, state, shape,
and transformation contracts. `test_composition.py` checks behavior and work
across component boundaries.

- `backend`: optional third-party environment adapters
- `integration`: multiple Jaxtor components composed end to end
- `statistical`: distributional checks with larger samples

The lightweight component suite is:

```sh
pytest -m "not backend and not statistical"
```
