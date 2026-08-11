# Sampler API

## Components

- `Mc`: open environment sampler with `observe(state)` and `sample(act, state)`.
- `VecMc`: `vmap`-based parallel `Mc`.
- `Imc`: caches agent data and returns an MC transition with its true successor.
- `Roll`: stacks aligned decisions, MC transitions, and successors.
- `EpisodeStats`: accumulates completed episodes from trajectories.
- `Sweep`: stochastic samples over all `(s, a)` pairs.
- `ExpSweep`: exact forward and backward propagation.

## Agent contract

An agent defines a decision. Only `act` is required
by `Imc`; algorithms may add fields such as `log_mu`, `value`, or `q`.

```python
@dataclass
class EGreedy:
    eps: float

    @dataclass
    class State:
        key: jax.Array
        q: jax.Array

    @dataclass
    class Decision:
        act: chex.Array
        q: jax.Array

    def decide(self, obs, state):
        key, explore_key, act_key = jrd.split(state.key, 3)
        q = state.q[:, obs]
        greedy = jnp.argmax(q)
        random = jrd.randint(act_key, (), 0, q.shape[0])
        act = jnp.where(jrd.uniform(explore_key) < self.eps, random, greedy)
        return self.Decision(act=act, q=q), state.replace(key=key)
```

## Single step

```python
mc = Mc(max_episode_len=100, env=env)
mc_state = mc.init(mc_key, env_state)

imc = Imc(agent=EGreedy(eps=0.1), mc=mc)
state = imc.init(mc_state, agent_state)

dec = imc.observe(state)
sample, state = imc.sample(state)

# dec: agent decision at sample.mc.obs
# sample.mc: obs, act, rew, term, trun, nobs
# sample.succ: agent decision computed at sample.mc.nobs
```

At an autoreset boundary, `sample.mc.nobs` remains the terminal observation and
`sample.succ` is computed there, while `imc.observe(state)` is the reset decision
used next.

## Rollout

```python
roll = Roll(imc=imc, seqlen=20)
trajectory, state = roll.sample(state)

trajectory.dec   # T decisions consumed by sampling
trajectory.mc    # T underlying MC transitions
trajectory.succ  # T decisions computed at trajectory.mc.nobs
```

With `VecMc`, set `seq_axis=1` for `(N, T, ...)` arrays. All three trajectory
pytrees use the same sequence axis.

Episode statistics remain alongside the sampler state. Partial episodes carry
across rollouts, while `drain` clears only completed-episode accumulators.

```python
stats = EpisodeStats(seq_axis=1)
stats_state = stats.init(batch_shape=(n_envs,))

trajectory, state = roll.sample(state)
stats_state = stats.update(trajectory.mc, stats_state)
metrics, stats_state = stats.drain(stats_state)
```

If agent parameters change while an `Imc.State` is retained, refresh its cached
decision before further sampling:

```python
state = state.replace(agent=updated_agent_state)
state = imc.refresh(state)
```

## Exact and exhaustive tabular sampling

```python
first, mc_states = Sweep(mc=mc).sample(key, env_state)

exp = ExpSweep(n_step=5)
q_seq = exp.backward(q, mdp, mu)
occupancy_seq = exp.forward(initial, mdp, mu)
```
