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
        q: jnp.ndarray  # (A, S)

    def act(self, key, obs, state):
        greedy = jnp.argmax(state.q[:, obs])
        rand = jrd.randint(key, (), 0, state.q.shape[0])
        act = jax.lax.select(jrd.uniform(key) < self.eps, rand, greedy)
        return act, state
```

## 1. Mc with Random Action

Single-step sampling with explicit action input.

```python
mc = Mc(max_episode_len=100, queue_size=10, env=env)
env_state = env.init(key)
mc_state = mc.init(key, env_state)

act = jrd.randint(key, (), 0, A)
trans, mc_state = mc.sample(act, mc_state)
```

## 2. Vectorized Imc

Parallel sampling across `n_env` environments using an agent policy.

```python
imc = Imc(
    agent=EGreedy(eps=0.1),
    mc=Mc(max_episode_len=100, queue_size=10, env=env)
)
env_state = env.init(key)

imc_states = Imc.State(
    key=jrd.split(key, n_env),
    mc=jax.vmap(imc.mc.init, in_axes=(0, None))(jrd.split(key, n_env), env_state),
    agent=EGreedy.State(q=q),
)

trans, imc_states = jax.vmap(
    imc.sample, in_axes=(Imc.State(key=0, mc=0, agent=None),)
)(imc_states)
# trans.obs.shape == (n_env,)
```

## 3. Vectorized Roll

Collect N-step trajectories from parallel environments.

```python
roll = Roll(imc=imc, seqlen=20)

imc_states = Imc.State(
    key=jrd.split(key, n_env),
    mc=jax.vmap(imc.mc.init, in_axes=(0, None))(jrd.split(key, n_env), env_state),
    agent=EGreedy.State(q=q),
)

trans, imc_states = jax.vmap(
    roll.sample, in_axes=(Imc.State(key=0, mc=0, agent=None),)
)(imc_states)
# trans.obs.shape == (n_env, 20)
```

## 4. Sweep + Roll

Sample first transition from all (A*S) state-action pairs, then continue with agent policy.

```python
mc = Mc(max_episode_len=100, queue_size=10, env=env)
sweep = Sweep(mc=mc)
roll = Roll(
    imc=Imc(
        agent=EGreedy(eps=0.1),
        mc=mc),
    seqlen=n_step - 1
)

first, mc_states = sweep.sample(key, env_state)

imc_states = Imc.State(
    key=jrd.split(key, A*S),
    mc=mc_states,
    agent=EGreedy.State(q=q),
)
rest, _ = jax.vmap(
    roll.sample, in_axes=(Imc.State(key=0, mc=0, agent=None),)
)(imc_states)

trans = jax.tree.map(lambda f, r: jnp.concatenate([f[:, None], r], axis=1), first, rest)
# trans.obs.shape == (A*S, n_step)
```

## 5. ExpSweep

Exact N-step propagation using transition dynamics (no sampling).

```python
exp = ExpSweep(n_step=5)
mu = jnp.ones((A, S)) / A  # uniform policy

# Backward: Σ_{k=0}^4 (P^\mu)^k Q
q_seq = exp.backward(q_arr, mdp, mu)
# q_seq.shape == (5, A, S)

# Forward: Σ_{k=0}^4 \pi^T ( P^\mu )^k
pi_seq = exp.forward(pi_arr, mdp, mu)
# pi_seq.shape == (5, A, S)
```
