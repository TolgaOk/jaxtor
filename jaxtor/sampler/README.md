# Sampler API

## Components

- `Mc`: open environment sampler with `observe(state)` and `sample(act, state)`.
- `VecMc`: `vmap`-based parallel `Mc`.
- `Imc`: selects one agent action and returns the resulting MC transition.
- `Roll`: stacks samples from any stateful sampler.
- `LoadedRoll`: stacks rich agent outputs at both transition endpoints.
- `EpisodeStats`: accumulates completed episodes from sequences.
- `Sweep`: stochastic samples over all `(s, a)` pairs.
- `ExpSweep`: exact forward and backward propagation.

## Minimal interaction

An ordinary `Imc` agent only selects an action. `Imc` and `Roll` return the
underlying MC transition directly, without another data wrapper.

```python
@dataclass
class EGreedy:
    eps: float

    @dataclass
    class State:
        key: jax.Array
        q: jax.Array

    def act(self, obs, state):
        key, explore_key, act_key = jrd.split(state.key, 3)
        q = state.q[:, obs]
        greedy = jnp.argmax(q)
        random = jrd.randint(act_key, (), 0, q.shape[0])
        act = jnp.where(jrd.uniform(explore_key) < self.eps, random, greedy)
        return act, replace(state, key=key)
```

```python
mc = Mc(max_eps_len=100, env=env)
imc = Imc(agent=EGreedy(eps=0.1), mc=mc)
state = imc.init(mc.init(mc_key, env_state), agent_state)
transition, state = imc.sample(state)

roll = Roll(imc=imc, seq_len=20)
seq, state = roll.sample(state)
# seq: stacked obs, act, rew, term, trun, nobs
```

## Loaded rollout

Use `LoadedRoll` when learning needs agent-defined data such as behavior
log-probabilities, values, or Q-values at both endpoints.

```python
@dataclass
class Output:
    act: chex.Array
    log_mu: chex.Array
    value: chex.Array


class ActorCritic:
    def apply(self, obs, state):
        output = Output(...)
        return output, state

roll = LoadedRoll(agent=ActorCritic(), mc=mc, seq_len=20)
seq, state = roll.sample(roll.init(mc_state, agent_state))

seq.dec   # T decisions whose actions were consumed
seq.mc    # T underlying MC transitions
seq.succ  # T outputs at each true nobs
```

At a boundary, `succ` describes the true terminal or truncated `nobs`; the next
`dec` is computed from the reset observation. Normal successors are reused
inside the scan. No predicted output is persisted between sampling calls, so an
externally updated agent state cannot leave stale derived data.

With `VecMc`, set `seq_axis=1` for `(N, T, ...)` arrays.

Episode statistics remain alongside the sampler state. Partial episodes carry
across rollouts, while `drain` clears only completed-episode accumulators.

```python
stats = EpisodeStats(seq_axis=1)
stats_state = stats.init(batch_shape=(n_envs,))

seq, state = roll.sample(state)
stats_state = stats.update(seq.mc, stats_state)  # LoadedRoll
metrics, stats_state = stats.drain(stats_state)
```

## Exact and exhaustive tabular sampling

```python
first, mc_states = Sweep(mc=mc).sample(key, env_state)

exp = ExpSweep(n_step=5)
q_seq = exp.backward(q, mdp, mu)
occupancy_seq = exp.forward(initial, mdp, mu)
```
