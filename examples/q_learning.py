"""Tabular Q-learning on a Garnet MDP using `jaxtor`, `rlax`, and `rich`.

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

import dataclasses
import time

import chex
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax
import tyro
from chex import dataclass
from jaxdp.base import greedy_policy, policy_evaluation
from jaxdp.mdp import MDP
from rich import print
from rich.progress import track

from jaxtor.env.tabular import garnet
from jaxtor.eval.tabular import Eval as Evaluator
from jaxtor.sampler import Imc, Mc


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    garnet: garnet.Config = dataclasses.field(default_factory=garnet.Config)
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

    def act(
        self, obs: chex.Array, state: Agent.State
    ) -> tuple[chex.Array, Agent.State]:
        """ε-greedy action selection."""
        key, act_key, explore_key = jrd.split(state.key, 3)
        greedy = jnp.argmax(state.q[:, obs])
        random = jrd.randint(act_key, (), 0, state.q.shape[0])
        action = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return action, state.replace(key=key)

    def q_vals(self, state: Agent.State, obs: chex.Array) -> chex.Array:
        """Q-values for given state indices. Returns (A, len(obs))."""
        return state.q[:, obs]


def optimal_q(mdp: MDP, gamma: float, n_iters: int = 20) -> chex.Array:
    """Compute optimal Q-values via policy iteration."""
    q = jnp.zeros((mdp.action_size, mdp.state_size))
    for _ in range(n_iters):
        q = policy_evaluation.q(mdp, greedy_policy.q(q), gamma)
    return q


@jax.jit
def train_step(state: Imc.State, k: int) -> Imc.State:
    """One transition + Q-learning update with decaying step size."""
    trans, state = imc.sample(state)
    q = state.agent.q
    alpha = cfg.alpha_init / (1.0 + k / cfg.alpha_period) ** cfg.alpha_power
    discount = jnp.where(trans.term, 0.0, cfg.gamma)
    td = rlax.q_learning(
        q[:, trans.obs], trans.act, trans.rew, discount, q[:, trans.nobs]
    )
    new_q = q.at[trans.act, trans.obs].add(alpha * td)
    return state.replace(agent=state.agent.replace(q=new_q))


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

cfg = tyro.cli(Config)
key = jrd.PRNGKey(cfg.seed)
S, A = cfg.garnet.state_size, cfg.garnet.action_size

agent = Agent(epsilon=cfg.epsilon)
imc = Imc(
    agent=agent,
    mc=Mc(
        max_episode_len=cfg.garnet.max_episode_len,
        queue_size=20,
        env=garnet.make(cfg.garnet),
    ),
)

key, env_key, agent_key = jrd.split(key, 3)
env_state = imc.mc.env.init(env_key)

opt_q = optimal_q(env_state.mdp, cfg.gamma)
opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))

evaluator = Evaluator(mdp=env_state.mdp, gamma=cfg.gamma, agent=agent)
jit_eval = jax.jit(evaluator.metric)
imc_state = Imc.State(
    mc=imc.mc.init(agent_key, env_state),
    agent=Agent.State(key=agent_key, q=jnp.zeros((A, S))),
)
eval_state = Evaluator.State(prev_agent=imc_state.agent, step=0)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

print(f"[bold green]Q-learning on Garnet[/bold green] ({S}S, {A}A)")

t0 = time.time()
for k in track(range(cfg.n_steps), description="Training"):
    imc_state = train_step(imc_state, k)

    if (k + 1) % cfg.eval_freq == 0:
        eval_state, m = jit_eval(eval_state, opt_q, imc_state.agent)
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
