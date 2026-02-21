# Sampler API

## Components

- `Mc` - open sampler: `sample(act, state)`
- `Imc` - closed sampler: `sample(state)`
- `Roll` - N-step: `sample(state) -> (seqlen, ...)`
- `Sweep` - all (s,a): `sample(key, env) -> (A*S, ...)`
- `ExpSweep` - exact propagation: `backward/forward(arr, mdp, mu) -> (n_step, A, S)`

## E-Greedy Agent

```python
@dataclass
class EGreedy:
    eps: float

    @dataclass
    class State:
        key: jrd.PRNGKey
        q: jnp.ndarray  # (A, S)

    def act(self, obs, state):
        key, k1, k2 = jrd.split(state.key, 3)
        greedy = jnp.argmax(state.q[:, obs])
        rand = jrd.randint(k1, (), 0, state.q.shape[0])
        act = jnp.where(jrd.uniform(k2) < self.eps, rand, greedy)
        return act, state.replace(key=key)
```

## 1. Markov Chain

```python
mc = Mc(max_episode_len=100, queue_size=10, env=env)
mc_state = mc.init(key, env.init(key))

act = jrd.randint(key, (), 0, A)
trans, mc_state = mc.sample(act, mc_state)
```

## 2. Induced Markov Chain

```python
imc = Imc(agent=EGreedy(eps=0.1), mc=mc)

imc_state = Imc.State(mc=mc_state, agent=EGreedy.State(key=key, q=q))
trans, imc_state = imc.sample(imc_state)
```


## 3. Vectorized Rollout
```python
agent_key, *mc_keys = jrd.split(key, n_env + 1)
mc_state = jax.vmap(mc.init, in_axes=(0, None))(mc_keys, env_state)

agent_keys = jrd.split(agent_key, n_env)
agent_state = EGreedy.State(key=agent_keys, q=jnp.zeros((A, S)))

roll = Roll(
    imc=Imc(
        agent=EGreedy(eps=0.1),
        mc=mc
    ),
    seqlen=20
)

vec_roll = jax.vmap(
    roll.sample,
    in_axes=(Imc.State(mc=0, agent=EGreedy.State(key=0, q=None)),),
    out_axes=(0, Imc.State(mc=0, agent=EGreedy.State(key=0, q=None)))
)

trans, imc_state = vec_roll(Imc.State(mc=mc_state, agent=agent_state))
# trans.obs.shape == (n_env, 20, ...)
```

## 4. Sweep + Rollout

```python
sweep = Sweep(mc=mc)
first_trans, mc_states = sweep.sample(key, env_state)
# first_trans.obs.shape == (A*S, ...)

agent_keys = jrd.split(key, A * S)
agent_state = EGreedy.State(key=agent_keys, q=jnp.zeros((A, S)))

roll = Roll(
    imc=Imc(
        agent=EGreedy(eps=0.1),
        mc=mc
    ),
    seqlen=20 - 1
)

vec_roll = jax.vmap(
    roll.sample,
    in_axes=(Imc.State(mc=0, agent=EGreedy.State(key=0, q=None)),),
    out_axes=(0, Imc.State(mc=0, agent=EGreedy.State(key=0, q=None)))
)

trans, imc_state = vec_roll(Imc.State(mc=mc_states, agent=agent_state))
# trans.obs.shape == (A*S, 20, ...)
```

## 5. ExpSweep

```python
exp = ExpSweep(n_step=5)
mu = jnp.ones((A, S)) / A

# Backward: { (P^\mu)^k Q }_{k=0}^4
q_seq = exp.backward(q_arr, mdp, mu)
# q_seq.shape == (5, A, S)

# Forward: { \pi^T ( P^\mu )^k }_{k=0}^4
pi_seq = exp.forward(pi_arr, mdp, mu)
# pi_seq.shape == (5, A, S)
```
