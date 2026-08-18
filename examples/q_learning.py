"""Tabular Q-learning on a Garnet MDP with explicit Jaxtor components.

The default recipe applies one online epsilon-greedy update per sampled
transition and evaluates convergence against the exact optimal Q-table.

Components::

    Composition
    │
    ├── agent: epsilon-greedy table
    ├── imc: Imc
    │   ├── agent
    │   └── mc: Mc
    │       └── TabularEnv
    └── evaluator: TabularEval
        └── agent

State::

    TrainState: Imc.State
    ├── agent: Q-table and sampling key
    └── mc: environment, observation, episode index, and sampling key

Flow::

    Main loop ↻
    ├→ sample one transition
    ├→ compute the Q-learning TD error
    ├→ update one table entry
    └→ periodically report convergence and greedy-policy return
"""

from __future__ import annotations

from dataclasses import dataclass as static_dataclass
from dataclasses import replace

from chex import dataclass
import jax
import jax.numpy as jnp
import jax.random as jrd
import rlax
import tyro

from jaxtor.env import tabular
from jaxtor.eval import TabularEval, optimal_q
from jaxtor.sampler import Imc, Mc


@static_dataclass(frozen=True)
class Config:
    """Command-line Q-learning configuration."""

    env_name: str = "mid-garnet"
    n_steps: int = 1_000_000
    alpha_init: float = 0.5
    alpha_power: float = 0.25
    alpha_period: float = 10_000.0
    gamma: float = 0.99
    epsilon: float = 0.1
    eval_freq: int = 10_000
    seed: int = 0


@dataclass
class Agent:
    """Epsilon-greedy finite-action Q-table agent.

    Attributes:
        state_size: Number of finite states.
        action_size: Number of finite actions.
        epsilon: Probability of selecting a uniformly random action.

    Public dataclasses:
        State: Q-table and action-selection key.

    Public methods:
        init: Initialize the Q-table and sampling key.
        qvec: Return ``Q(s, .)`` for arbitrary state indices.
        act: Select an epsilon-greedy action.
    """

    state_size: int
    action_size: int
    epsilon: float

    @dataclass
    class State:
        """Dynamic Q-table agent state.

        Attributes:
            key: Action-selection key.
            q: Action values with shape ``(A, S)``.
        """

        key: jax.Array
        q: jax.Array

    def __post_init__(self) -> None:
        """Validate the static table and exploration configuration."""
        if self.state_size < 1:
            raise ValueError("state_size must be positive")
        if self.action_size < 1:
            raise ValueError("action_size must be positive")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")

    def init(self, key: jax.Array) -> Agent.State:
        """Initialize a zero Q-table and action-selection key."""
        return self.State(
            key=key,
            q=jnp.zeros((self.action_size, self.state_size)),
        )

    def qvec(self, obs: jax.Array, state: Agent.State) -> jax.Array:
        """Return finite-action values for the supplied state indices."""
        return state.q[:, obs]

    def act(self, obs: jax.Array, state: Agent.State) -> tuple[jax.Array, Agent.State]:
        """Select an epsilon-greedy action and advance the sampling key."""
        key, random_key, explore_key = jrd.split(state.key, 3)
        greedy = jnp.argmax(self.qvec(obs, state))
        random = jrd.randint(random_key, (), 0, self.action_size)
        act = jnp.where(jrd.uniform(explore_key) < self.epsilon, random, greedy)
        return act, replace(state, key=key)


type TrainState = Imc.State[Mc.State[tabular.TabularEnv.State], Agent.State]


cfg = tyro.cli(Config) if __name__ == "__main__" else Config()
mc_key, env_key, agent_key = jrd.split(jrd.key(cfg.seed), 3)


env = tabular.make(cfg.env_name)
env_state = env.init(env_key)
state_size = env_state.mdp.state_size
action_size = env_state.mdp.action_size


agent = Agent(
    state_size=state_size,
    action_size=action_size,
    epsilon=cfg.epsilon,
)
agent_state = agent.init(agent_key)
mc = Mc(max_eps_len=env.config.max_eps_len, env=env)
imc = Imc(agent=agent, mc=mc)


opt_q = optimal_q(env_state.mdp, cfg.gamma)
opt_rho = float(jnp.sum(env_state.mdp.initial * jnp.max(opt_q, axis=0)))
evaluator = TabularEval(
    mdp=env_state.mdp,
    gamma=cfg.gamma,
    agent=agent,
    opt_q=opt_q,
)
evaluate = jax.jit(evaluator.evaluate)


@jax.jit
def update(state: TrainState, step: int) -> TrainState:
    """Sample one transition and update its Q-table entry."""
    transition, state = imc.sample(state)
    q = state.agent.q
    alpha = cfg.alpha_init / (1.0 + step / cfg.alpha_period) ** cfg.alpha_power
    discount = jnp.where(transition.term, 0.0, cfg.gamma)
    td = rlax.q_learning(
        q[:, transition.obs],
        transition.act,
        transition.rew,
        discount,
        q[:, transition.nobs],
    )
    q = q.at[transition.act, transition.obs].add(alpha * td)
    return replace(state, agent=replace(state.agent, q=q))


def train() -> TrainState:
    """Initialize dynamic state and train the configured Q-learning recipe."""
    state = imc.init(
        mc=mc.init(mc_key, env_state),
        agent=agent_state,
    )
    eval_state = evaluator.init(agent_state)

    print(f"Q-learning on {cfg.env_name} ({state_size}S, {action_size}A)")
    for step in range(1, cfg.n_steps + 1):
        state = update(state, step - 1)
        if step % cfg.eval_freq == 0:
            metrics, eval_state = evaluate(state.agent, eval_state)
            print(
                f"step={step:7d}"
                f"  bellman={float(metrics.bellman_linf):.4f}"
                f"  value={float(metrics.value_norm):.4f}"
                f"  rho={float(metrics.pi_eval_rho):.3f}"
            )

    print(f"optimal_rho={opt_rho:.3f}")
    return state


if __name__ == "__main__":
    train()
