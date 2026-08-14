"""EnvPool backend for fast CPU environments.

``make`` exposes EnvPool through ``GymEnv``. ``SameStep`` converts EnvPool's
autoreset behavior to the same-step semantics expected by the sampler stack.

    env = make("Hopper-v5", num_envs=8)
    mc = VecMc(mc=Mc(max_eps_len=1_000, env=env))
    state = mc.init(keys, env.init(key))
    transition, state = mc.sample(act, state)
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from jaxtor.env.gymnasium import (
    ArraySpace,
    GymEnv,
    ResetResult,
    StepResult,
    from_factory,
)

__all__ = ["make"]


class Runtime(Protocol):
    """Raw EnvPool surface consumed by the SAME_STEP adapter."""

    num_envs: int
    single_observation_space: ArraySpace
    single_action_space: ArraySpace
    all_env_ids: np.ndarray

    def reset(
        self, *, seed: int | None = None, env_id: npt.ArrayLike | None = None
    ) -> ResetResult: ...
    def step(self, actions: npt.ArrayLike, *, env_id: npt.ArrayLike) -> StepResult: ...
    def close(self) -> None: ...


def import_envpool() -> Any:
    """Import ``envpool``, stubbing the optional GUI suites (procgen/vizdoom).

    Those suites' native extensions need a Qt runtime and are unused here; their
    eager import in ``envpool.entry`` would otherwise fail on hosts without Qt.
    """
    import sys
    import types

    for mod in (
        "envpool.procgen",
        "envpool.procgen.registration",
        "envpool.vizdoom",
        "envpool.vizdoom.registration",
    ):
        sys.modules.setdefault(mod, types.ModuleType(mod))
    import envpool

    return envpool


class SameStep:
    """Adapt an EnvPool env from ``NEXT_STEP`` to gymnasium ``SAME_STEP`` autoreset.

    Presents the minimal gymnasium-``VectorEnv`` surface that
    :func:`jaxtor.env.gymnasium.from_factory` consumes. On a done step, resets
    just the terminated envs (``reset(env_id=...)``) to fetch their reset obs
    immediately and clear EnvPool's pending auto-reset, then returns ``obs`` =
    reset obs with the terminal obs in ``info["final_obs"]``.

    Attributes:
        num_envs: Number of parallel environments.
        single_observation_space: Per-env observation space.
        single_action_space: Per-env action space.
    """

    def __init__(self, env: Runtime):
        """Wrap one already-seeded EnvPool runtime."""
        self.env = env
        self.num_envs = env.num_envs
        self.single_observation_space = env.single_observation_space
        self.single_action_space = env.single_action_space

    def reset(self, *, seed: int | None = None) -> ResetResult:
        """Reset all envs; the pool's RNG was fixed by its factory seed."""
        return self.env.reset(seed=seed)

    def step(self, actions: npt.ArrayLike) -> StepResult:
        """Step a pool subset and convert autoreset to SAME_STEP.

        Stepping a subset (via ``env_id``) lets a vmap of size ``m <= num_envs``
        use only ``m`` of the pre-allocated environments. Terminal observations
        are moved into ``info["final_obs"]`` and replaced with reset observations.
        """
        act = np.asarray(actions)
        ids = self.env.all_env_ids[: len(act)]
        obs, rew, term, trun, _ = self.env.step(act, env_id=ids)
        obs = np.asarray(obs)
        rew = np.asarray(rew)
        term = np.asarray(term)
        trun = np.asarray(trun)
        done = np.logical_or(term, trun)
        final = np.empty(len(act), dtype=object)
        final[:] = None
        if done.any():
            idx = np.nonzero(done)[0]
            reset_obs, _ = self.env.reset(env_id=ids[done])
            reset_obs = np.asarray(reset_obs)
            obs = obs.copy()
            for k, i in enumerate(idx):
                final[i] = obs[i].copy()
                obs[i] = reset_obs[k]
        return obs, rew, term, trun, {"final_obs": final}

    def close(self) -> None:
        """Close the underlying EnvPool env."""
        self.env.close()


def make(name: str, num_envs: int = 1, **kwargs: Any) -> GymEnv:
    """Create a jaxtor ``GymEnv`` backed by EnvPool (fast CPU MuJoCo).

    EnvPool fixes its RNG at construction. The returned component therefore
    creates a fresh pool from the key-derived seed in ``env.init(key)``.
    EnvPool is always vectorized, including when ``num_envs=1``.
    EnvPool can step a prefix selected by ``env_id``, so the resulting adapter
    permits a mapped axis smaller than the allocated pool.

    Args:
        name: EnvPool/Gymnasium env id (e.g. ``"Hopper-v5"``).
        num_envs: Number of parallel environments.
        **kwargs: EnvPool config — ``seed``, ``num_threads``,
            ``reset_noise_scale``, ``max_episode_steps``, ``frame_skip``, reward
            weights, ``xml_file``, etc.

    Returns:
        GymEnv instance (same interface as :func:`jaxtor.env.gymnasium.make`).
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")

    envpool = import_envpool()
    try:
        spec = envpool.make_spec(name, num_envs=num_envs, **kwargs)
    except KeyError as error:
        raise ValueError(f"{name!r} is not supported by EnvPool") from error

    def factory(seed: int | None) -> tuple[SameStep, np.ndarray]:
        """Create a pool with an int32 seed and adapt its autoreset behavior."""
        kw = dict(kwargs)
        if seed is not None:
            kw["seed"] = int(seed) & 0x7FFFFFFF
        raw_env = cast(
            Runtime,
            envpool.make(name, env_type="gymnasium", num_envs=num_envs, **kw),
        )
        vec_env = SameStep(raw_env)
        try:
            obs, _ = vec_env.reset()
        except Exception:
            vec_env.close()
            raise
        return vec_env, np.asarray(obs)

    return from_factory(
        factory,
        num_envs,
        allow_subset=True,
        observation_space=spec.gymnasium_observation_space,
        action_space=spec.gymnasium_action_space,
    )
