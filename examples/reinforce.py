"""REINFORCE on CartPole-v1 (`gymnax`) using `jaxtor`, `rlax`, and `equinox`.

Policy gradient method from Williams (1992) "Simple statistical gradient-following
algorithms for connectionist reinforcement learning", Machine Learning, 8(3-4).

The gradient estimator is:

    ∇J(θ) = E_π [ ∑_t ∇ log π_θ(a_t|s_t) · G_t ]

where G_t = ∑_{k=0}^{T-t} γ^k r_{t+k} is the discounted return from step t.
A baseline-free variant with an entropy bonus H(π) to encourage exploration:

    L(θ) = -E_π [ log π_θ(a_t|s_t) · G_t ] - c_ent · H(π_θ)

- `jaxtor`:
  - rollout sampler: `GymnaxEnv → VecMc → Imc → Roll` for parallel rollouts.
  - evaluation: `GymnaxEnv → VecMc → Imc → Eval` for evaluation.
- `equinox`: network with tanh activation, MLP (4 → 64 → 2).
- `rlax`: `discounted_returns`, `policy_gradient_loss`, `entropy_loss` primitives.

"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax
import tyro
import chex
from chex import dataclass
from rich import print
from rich.progress import track

from jaxtor.env.gymnax import make
from jaxtor.eval.mc import Eval as Evaluator
from jaxtor.sampler import Mc, VecMc, Imc, Roll


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Training configuration for tyro CLI."""

    n_iters: int = 100
    hidden: int = 64
    lr: float = 3e-3
    gamma: float = 0.99
    entropy_coef: float = 0.01
    tau_pi: float = 1e-4  # target policy temperature
    n_envs: int = 16
    seqlen: int = 200
    max_episode_len: int = 500
    tau_mu: float = 1.0  # behavior policy temperature
    eval_freq: int = 5
    eval_envs: int = 20
    seed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Agent (satisfies jaxtor Agent protocol)
# ──────────────────────────────────────────────────────────────────────────────


class MLP(eqx.Module):
    """Two-layer equinox MLP with tanh activation."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array

    def __init__(self, obs_dim: int, hidden: int, act_dim: int, *, key: jax.Array):
        k1, k2 = jrd.split(key)
        self.w1 = jrd.normal(k1, (obs_dim, hidden)) * jnp.sqrt(2.0 / obs_dim)
        self.b1 = jnp.zeros(hidden)
        self.w2 = jrd.normal(k2, (hidden, act_dim)) * jnp.sqrt(2.0 / hidden)
        self.b2 = jnp.zeros(act_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.tanh(x @ self.w1 + self.b1)
        return x @ self.w2 + self.b2


@dataclass
class Agent:
    """Softmax policy agent with temperature scaling.

    tau=1.0: standard softmax sampling.
    tau->0: approaches greedy (argmax).
    """

    tau: float = 1.0

    @dataclass
    class State:
        key: jax.Array
        params: eqx.Module

    def act(self, obs: chex.Array, state: Agent.State) -> tuple[jax.Array, Agent.State]:
        """Sample one action for an observation."""
        key, act_key = jrd.split(state.key)
        logits = state.params(obs)
        action = jrd.categorical(act_key, logits / self.tau)
        return action, state.replace(key=key)


# ──────────────────────────────────────────────────────────────────────────────
# REINFORCE loss
# ──────────────────────────────────────────────────────────────────────────────


def reinforce_loss(
    params: eqx.Module,
    trajectory: Mc.Transition,
    gamma: float,
    entropy_coef: float,
) -> chex.Numeric:
    """REINFORCE policy gradient loss with entropy bonus."""
    logits = jax.vmap(jax.vmap(params))(trajectory.obs)

    discount_t = jnp.where(trajectory.term, 0.0, gamma)

    returns = jax.vmap(rlax.discounted_returns, in_axes=(0, 0, None))(
        trajectory.rew, discount_t, jnp.float32(0.0)
    )

    w_t = jnp.ones_like(returns)
    pg_loss = jax.vmap(rlax.policy_gradient_loss)(
        logits, trajectory.act.astype(jnp.int32), returns, w_t
    ).mean()
    ent_loss = jax.vmap(rlax.entropy_loss)(logits, w_t).mean()

    return pg_loss + entropy_coef * ent_loss


# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

cfg = tyro.cli(Config)
key = jrd.PRNGKey(cfg.seed)
env = make("CartPole-v1")
vec_mc = VecMc(
    mc=Mc(
        max_episode_len=cfg.max_episode_len,
        env=env,
    )
)
behavior_agent = Agent(tau=cfg.tau_mu)
target_agent = Agent(tau=cfg.tau_pi)

# Training rollout sampler
roll = Roll(
    imc=Imc(agent=behavior_agent, mc=vec_mc),
    seqlen=cfg.seqlen,
    seq_axis=1,
)

# Eval component
evaluator = Evaluator(
    imc=Imc(agent=target_agent, mc=vec_mc),
    episode_len=cfg.max_episode_len,
)


@jax.jit
def train_step(state: Imc.State) -> tuple[Imc.State, chex.Numeric]:
    """Sample a rollout, compute REINFORCE loss, and update params."""
    trajectory, state = roll.sample(state)
    params = state.agent.params
    loss, grads = jax.value_and_grad(reinforce_loss)(
        params, trajectory, cfg.gamma, cfg.entropy_coef
    )
    new_params = jax.tree.map(lambda p, g: p - cfg.lr * g, params, grads)
    state = eqx.tree_at(lambda s: s.agent.params, state, new_params)
    return state, loss


jit_eval = jax.jit(evaluator.evaluate)

# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

print("[bold green]CartPole REINFORCE[/bold green]")

# Init states
key, params_key, env_key, agent_key, eval_key = jrd.split(key, 5)
imc_state = roll.imc.init(
    vec_mc.init(jrd.split(key, cfg.n_envs), env=env.init(env_key)),
    Agent.State(key=agent_key, params=MLP(4, cfg.hidden, 2, key=params_key)),
)

t0 = time.time()
for i in track(range(cfg.n_iters), description="Training"):
    imc_state, loss = train_step(imc_state)

    if (i + 1) % cfg.eval_freq == 0:
        eval_key, env_key, k = jrd.split(eval_key, 3)
        m, eval_state = jit_eval(
            evaluator.imc.init(
                vec_mc.init(jrd.split(k, cfg.eval_envs), env.init(env_key)),
                imc_state.agent.replace(key=eval_key),
            )
        )
        steps = (i + 1) * cfg.n_envs * cfg.seqlen
        print(
            f"  iter {i + 1:4d}  loss={float(loss):+.4f}"
            f"  rew={float(m.avg_eps_rew):.1f}±{float(m.std_eps_rew):.1f}"
            f"  len={float(m.avg_eps_len):.1f}"
            f"  steps={steps:,}"
        )

elapsed = time.time() - t0
print(
    f"\n[bold green]Completed[/bold green] in {elapsed:.1f}s"
    f"  rew={float(m.avg_eps_rew):.1f}±{float(m.std_eps_rew):.1f}"
    f"  (over {int(m.n_episodes)} eps)"
)
