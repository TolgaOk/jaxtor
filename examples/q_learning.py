"""Tabular Q-learning on a Garnet MDP using `jaxtor`, `rlax`, and `jaxdp`.

Online ε-greedy Q-learning: one transition per step via Imc, then update:

    Q(s,a) ← Q(s,a) + α(k) · δ(s,a)

where δ(s,a) = r + γ max_a' Q(s',a') - Q(s,a) is the TD error (rlax.q_learning)
and α(k) = α_init / (1 + k / α_period)^α_power is a decaying step size.

- `jaxtor`:
  - sampler: `TabularEnv → Mc → Imc` for single-step transitions.
  - evaluation: `eval.tabular.Eval` for convergence diagnostics against Q*.
- `rlax`: `q_learning` TD error.
- `rich`: progress bar and formatted output.

"""

from __future__ import annotations

import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax
import tyro
from chex import dataclass
from rich import print
from rich.progress import track

from jaxtor.env import tabular
from jaxtor.eval.tabular import Eval as Evaluator, optimal_q
from jaxtor.sampler import Imc, Mc


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    env_name: str = "mid-garnet"
    n_steps: int = 1_000_000
    alpha_init: float = 0.5
    alpha_power: float = 0.25
    alpha_period: float = 10_000.0
    gamma: float = 0.99
    epsilon: float = 0.1
    eval_freq: int = 10_000
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """ε-greedy tabular Q-learning agent with an (A, S) Q-table."""

    epsilon: float

    @dataclass
    class State:
        key: chex.Array
        q: chex.Array  # (A, S)

    @dataclass
    class Decision:
        """Q-learning data attached to one decision."""

        act: chex.Array
        q: chex.Array

    def decide(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[Agent.Decision, Agent.State]:
        """Prepare an ε-greedy action and its Q-values."""
        key, act_key, explore_key = jrd.split(state.key, 3)
        q = state.q[:, obs]
        greedy = jnp.argmax(q)
        random = jrd.randint(act_key, (), 0, state.q.shape[0])
        action = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return self.Decision(act=action, q=q), state.replace(key=key)

    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array:
        """Q-values for given state indices."""
        return state.q[:, obs]


@jax.jit
def train_step(state: Imc.State, k: int) -> Imc.State:
    """One transition + Q-learning update with decaying step size."""
    dec = imc.observe(state)
    sample, state = imc.sample(state)
    q = state.agent.q
    alpha = cfg.alpha_init / (1.0 + k / cfg.alpha_period) ** cfg.alpha_power
    discount = jnp.where(sample.mc.term, 0.0, cfg.gamma)
    td = rlax.q_learning(
        dec.q,
        dec.act,
        sample.mc.rew,
        discount,
        sample.succ.q,
    )
    new_q = q.at[dec.act, sample.mc.obs].add(alpha * td)
    state = state.replace(agent=state.agent.replace(q=new_q))
    return imc.refresh(state)


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

cfg = tyro.cli(Config)
env = tabular.make(cfg.env_name)

agent = Agent(epsilon=cfg.epsilon)
imc = Imc(
    agent=agent,
    mc=Mc(
        max_episode_len=env.config.max_episode_len,
        env=env,
    ),
)

key = jrd.PRNGKey(cfg.seed)
key, env_key, agent_key = jrd.split(key, 3)
env_state = env.init(env_key)
S, A = env_state.mdp.state_size, env_state.mdp.action_size

opt_q = optimal_q(env_state.mdp, cfg.gamma)
opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))

evaluator = Evaluator(
    mdp=env_state.mdp,
    gamma=cfg.gamma,
    agent=agent,
    opt_q=opt_q,
)
jit_eval = jax.jit(evaluator.evaluate)
agent_state = Agent.State(key=agent_key, q=jnp.zeros((A, S)))
imc_state = imc.init(mc=imc.mc.init(agent_key, env_state), agent=agent_state)
eval_state = evaluator.init(agent_state)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

print(f"[bold green]Q-learning on {cfg.env_name}[/bold green] ({S}S, {A}A)")

t0 = time.time()
for k in track(range(cfg.n_steps), description="Training"):
    imc_state = train_step(imc_state, k)

    if (k + 1) % cfg.eval_freq == 0:
        m, eval_state = jit_eval(eval_state, imc_state.agent)
        print(
            f"  step {k + 1:6d}"
            f"  bellman={float(m.bellman_linf):.4f}"
            f"  value={float(m.value_norm):.4f}"
            f"  ρ(π)={float(m.pi_eval_rho):.3f}"
        )

elapsed = time.time() - t0
print(
    f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
    f"  value_norm={float(m.value_norm):.6f}"
    f"  bellman_linf={float(m.bellman_linf):.6f}"
    f"  ρ*(π)={opt_rho:.3f}"
)
