"""MJX environment adapter.

Pure-JAX, on-device MuJoCo (MuJoCo XLA) behind jaxtor's Env protocol. Each env
reuses the exact Gymnasium v5 XML and reproduces its observation, reward, and
termination, so ``MjxEnv`` matches ``gymnasium.make(name)`` one-to-one (float64).
The sampler handles episode reset (no auto-reset), like the gymnax adapter.

Supported: ``Hopper-v5``, ``Walker2d-v5``, ``HalfCheetah-v5``, ``Swimmer-v5``.
The 3D contact-rich envs (Ant, Humanoid) are excluded: MJX's collision algorithm
diverges from CPU MuJoCo, so one-to-one parity is impossible.

Backend defaults to MJX-JAX (XLA). On NVIDIA GPUs, pass ``impl="warp"`` for the
MuJoCo Warp backend (faster on contact-rich scenes, but not differentiable, not
parity-tested, and needs ``mujoco_warp`` + CUDA).

Refs: MJX (https://mujoco.readthedocs.io/en/stable/mjx.html); Gymnasium v5
(https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/mujoco).

Example:
    >>> import jax
    >>> from jaxtor.env import mjx
    >>> env = mjx.make("Hopper-v5")
    >>> state = env.init(jax.random.PRNGKey(0))
    >>> obs, state = env.reset(jax.random.PRNGKey(0), state)
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable
from typing import Protocol, TypedDict, Unpack

import jax
import jax.numpy as jnp
import jax.random as jrd
import chex
from chex import dataclass
from mujoco import _structs as structs
from mujoco import mjx

_INF = float("inf")


class LocomotionOptions(TypedDict, total=False):
    """Typed task overrides accepted by locomotion builders."""

    forward_reward_weight: float
    ctrl_cost_weight: float
    healthy_reward: float
    terminate_when_unhealthy: bool
    healthy_z_range: tuple[float, float]
    healthy_angle_range: tuple[float, float]
    healthy_state_range: tuple[float, float]
    exclude_positions: int
    velocity_clip: float


class MakeOptions(LocomotionOptions, total=False):
    """Typed environment and task overrides accepted by :func:`make`."""

    reset_noise_scale: float
    frame_skip: int


class Task(Protocol):
    """Per-environment observation, reward, and termination semantics."""

    def obs(self, data: mjx.Data) -> chex.Array: ...

    def reward(
        self, x_velocity: chex.Numeric, act: chex.Array, data: mjx.Data
    ) -> chex.Array: ...

    def terminal(self, data: mjx.Data) -> chex.Array: ...


@dataclasses.dataclass(frozen=True)
class Locomotion:
    """Forward-locomotion task (Hopper/Walker2d/HalfCheetah/Swimmer family).

    Obs is ``qpos[exclude_positions:]`` and ``qvel`` clipped to ``±velocity_clip``;
    reward is ``forward_reward_weight*x_velocity + healthy_reward*is_healthy -
    ctrl_cost_weight*||act||^2``. Health bounds left at ``inf`` are inactive.

    Attributes:
        forward_reward_weight: Weight on forward (x) velocity.
        ctrl_cost_weight: Weight on the squared-action control cost.
        healthy_reward: Per-step reward while healthy.
        terminate_when_unhealthy: Whether leaving the healthy set terminates.
        healthy_z_range: Bounds on torso height ``qpos[1]`` for health.
        healthy_angle_range: Bounds on torso angle ``qpos[2]`` for health.
        healthy_state_range: Bounds on ``concat(qpos, qvel)[2:]`` for health.
        exclude_positions: Number of leading ``qpos`` entries dropped from obs.
        velocity_clip: Symmetric clip applied to ``qvel`` in the observation.
    """

    forward_reward_weight: float
    ctrl_cost_weight: float
    healthy_reward: float = 0.0
    terminate_when_unhealthy: bool = False
    healthy_z_range: tuple[float, float] = (-_INF, _INF)
    healthy_angle_range: tuple[float, float] = (-_INF, _INF)
    healthy_state_range: tuple[float, float] = (-_INF, _INF)
    exclude_positions: int = 1
    velocity_clip: float = _INF

    def is_healthy(self, data: mjx.Data) -> chex.Array:
        """Whether the body satisfies all (active) health bounds."""
        z, angle = data.qpos[1], data.qpos[2]
        state = jnp.concatenate([data.qpos, data.qvel])[2:]
        min_state, max_state = self.healthy_state_range
        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range
        return (
            jnp.all((min_state < state) & (state < max_state))
            & (min_z < z)
            & (z < max_z)
            & (min_angle < angle)
            & (angle < max_angle)
        )

    def obs(self, data: mjx.Data) -> chex.Array:
        """Observation: trailing qpos and clipped qvel."""
        velocity = jnp.clip(data.qvel, min=-self.velocity_clip, max=self.velocity_clip)
        return jnp.concatenate([data.qpos[self.exclude_positions :], velocity])

    def reward(
        self, x_velocity: chex.Numeric, act: chex.Array, data: mjx.Data
    ) -> chex.Array:
        """Reward: forward velocity plus healthy bonus minus control cost."""
        forward = self.forward_reward_weight * x_velocity
        survive = self.is_healthy(data) * self.healthy_reward
        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(act))
        return forward + survive - ctrl_cost

    def terminal(self, data: mjx.Data) -> chex.Array:
        """Whether the episode terminates due to leaving the healthy set."""
        return jnp.logical_and(
            jnp.logical_not(self.is_healthy(data)), self.terminate_when_unhealthy
        )


@dataclasses.dataclass(frozen=True)
class Backend:
    """MJX physics backend selection.

    Attributes:
        impl: ``None``/``"jax"`` (default XLA) or ``"warp"`` (NVIDIA Warp; needs
            ``mujoco_warp`` + CUDA, not differentiable, not parity-tested).
        nconmax: Max contacts per step (Warp only).
        njmax: Max constraints per step (Warp only).
    """

    impl: str | None = None
    nconmax: int | None = None
    njmax: int | None = None

    def put_model(self, model: structs.MjModel) -> mjx.Model:
        """Put a host model on device under this backend."""
        return mjx.put_model(model, impl=self.impl)

    def make_data(self, model: mjx.Model) -> mjx.Data:
        """Allocate a fresh data buffer; Warp contact/constraint caps if set."""
        return mjx.make_data(
            model,
            impl=self.impl,
            nconmax=self.nconmax,
            njmax=self.njmax,
        )


@dataclass
class MjxEnv:
    """Adapter for a single MJX MuJoCo environment.

    Attributes:
        mjx_model: Device-resident MJX model.
        task: Per-environment observation/reward/termination semantics.
        backend: Physics backend (XLA by default; Warp on NVIDIA).
        init_qpos: Default generalized positions, shape ``(nq,)``.
        init_qvel: Default generalized velocities, shape ``(nv,)``.
        frame_skip: Number of physics substeps per ``step``.
        dt: Control timestep, ``model.opt.timestep * frame_skip``.
        reset_noise_scale: Half-width (uniform) or std scale (Gaussian) of the
            reset perturbation.
        reset_qvel_gaussian: Sample reset ``qvel`` from a scaled Gaussian
            (HalfCheetah/Ant) instead of the uniform default.
    """

    mjx_model: mjx.Model
    task: Task
    init_qpos: chex.Array
    init_qvel: chex.Array
    frame_skip: int
    dt: float
    reset_noise_scale: float
    reset_qvel_gaussian: bool = False
    backend: Backend = Backend()

    @dataclass
    class State:
        """Environment state.

        Attributes:
            data: Inner MJX physics state (``mjx.Data``).
        """

        data: mjx.Data

    @dataclass
    class Step:
        """Single-step transition result.

        Attributes:
            nobs: Next observation.
            rew: Reward.
            term: Natural termination flag.
            trun: Truncation flag (always ``False``; the sampler enforces the
                episode time limit).
        """

        nobs: chex.Array
        rew: chex.Array
        term: chex.Array
        trun: chex.Array

    def _physics_step(self, data: mjx.Data, act: chex.Array) -> mjx.Data:
        """Advance the physics ``frame_skip`` substeps under constant control."""
        data = data.replace(ctrl=act)
        return jax.lax.fori_loop(
            0, self.frame_skip, lambda _, d: mjx.step(self.mjx_model, d), data
        )

    def init(self, key: chex.PRNGKey) -> MjxEnv.State:
        """Initialize the environment state at the default pose.

        Args:
            key: Random key (unused; the sampler reseeds via ``reset``).

        Returns:
            Initialized state.
        """
        data = self.backend.make_data(self.mjx_model).replace(
            qpos=self.init_qpos, qvel=self.init_qvel
        )
        return self.State(data=mjx.forward(self.mjx_model, data))

    def step(
        self, key: chex.PRNGKey, act: chex.Array, state: State
    ) -> tuple[Step, State]:
        """Step the environment without auto-reset.

        Args:
            key: Random key (unused; MJX dynamics are deterministic).
            act: Action, shape ``(nu,)``.
            state: Current state.

        Returns:
            Step result and next state.
        """
        x_before = state.data.qpos[0]
        data = self._physics_step(state.data, act)
        x_velocity = (data.qpos[0] - x_before) / self.dt
        return (
            self.Step(
                nobs=self.task.obs(data),
                rew=self.task.reward(x_velocity, act, data),
                term=self.task.terminal(data),
                trun=jnp.bool_(False),
            ),
            dataclasses.replace(state, data=data),
        )

    def reset(self, key: chex.PRNGKey, state: State) -> tuple[chex.Array, State]:
        """Reset to a new episode with uniform pose/velocity noise.

        Args:
            key: Random key for the reset perturbation.
            state: Current state (structure preserved, contents replaced).

        Returns:
            Initial observation and reset state.
        """
        key_pos, key_vel = jrd.split(key)
        scale = self.reset_noise_scale
        qpos = self.init_qpos + jrd.uniform(
            key_pos, self.init_qpos.shape, minval=-scale, maxval=scale
        )
        if self.reset_qvel_gaussian:
            qvel = self.init_qvel + scale * jrd.normal(key_vel, self.init_qvel.shape)
        else:
            qvel = self.init_qvel + jrd.uniform(
                key_vel, self.init_qvel.shape, minval=-scale, maxval=scale
            )
        data = self.backend.make_data(self.mjx_model).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)
        return self.task.obs(data), dataclasses.replace(state, data=data)

    def obs(self, state: State) -> chex.Array:
        """Get the observation from a state.

        Args:
            state: Current state.

        Returns:
            Current observation.
        """
        return self.task.obs(state.data)


def _load_model(xml_file: str) -> structs.MjModel:
    """Load a Gymnasium MuJoCo asset by filename into a ``mujoco.MjModel``."""
    import gymnasium.envs.mujoco as _gym_mujoco

    path = os.path.join(os.path.dirname(_gym_mujoco.__file__), "assets", xml_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gymnasium MuJoCo asset not found: {path}")
    return structs.MjModel.from_xml_path(path)


def _build(
    xml_file: str,
    task: Task,
    frame_skip: int,
    reset_noise_scale: float,
    backend: Backend,
    reset_qvel_gaussian: bool = False,
) -> MjxEnv:
    """Assemble an ``MjxEnv`` from a Gymnasium asset and a task."""
    model = _load_model(xml_file)
    return MjxEnv(
        mjx_model=backend.put_model(model),
        task=task,
        init_qpos=jnp.asarray(model.qpos0),
        init_qvel=jnp.zeros(model.nv),
        frame_skip=frame_skip,
        dt=model.opt.timestep * frame_skip,
        reset_noise_scale=reset_noise_scale,
        reset_qvel_gaussian=reset_qvel_gaussian,
        backend=backend,
    )


def _hopper(
    backend: Backend,
    reset_noise_scale: float = 5e-3,
    frame_skip: int = 4,
    **overrides: Unpack[LocomotionOptions],
) -> MjxEnv:
    """Build the ``Hopper-v5`` MJX environment."""
    return _build(
        "hopper.xml",
        dataclasses.replace(
            Locomotion(
                forward_reward_weight=1.0,
                ctrl_cost_weight=1e-3,
                healthy_reward=1.0,
                terminate_when_unhealthy=True,
                healthy_z_range=(0.7, _INF),
                healthy_angle_range=(-0.2, 0.2),
                healthy_state_range=(-100.0, 100.0),
                exclude_positions=1,
                velocity_clip=10.0,
            ),
            **overrides,
        ),
        frame_skip,
        reset_noise_scale,
        backend,
    )


def _walker2d(
    backend: Backend,
    reset_noise_scale: float = 5e-3,
    frame_skip: int = 4,
    **overrides: Unpack[LocomotionOptions],
) -> MjxEnv:
    """Build the ``Walker2d-v5`` MJX environment."""
    return _build(
        "walker2d_v5.xml",
        dataclasses.replace(
            Locomotion(
                forward_reward_weight=1.0,
                ctrl_cost_weight=1e-3,
                healthy_reward=1.0,
                terminate_when_unhealthy=True,
                healthy_z_range=(0.8, 2.0),
                healthy_angle_range=(-1.0, 1.0),
                exclude_positions=1,
                velocity_clip=10.0,
            ),
            **overrides,
        ),
        frame_skip,
        reset_noise_scale,
        backend,
    )


def _half_cheetah(
    backend: Backend,
    reset_noise_scale: float = 0.1,
    frame_skip: int = 5,
    **overrides: Unpack[LocomotionOptions],
) -> MjxEnv:
    """Build the ``HalfCheetah-v5`` MJX environment (no termination)."""
    return _build(
        "half_cheetah.xml",
        dataclasses.replace(
            Locomotion(
                forward_reward_weight=1.0,
                ctrl_cost_weight=0.1,
                exclude_positions=1,
            ),
            **overrides,
        ),
        frame_skip,
        reset_noise_scale,
        backend,
        reset_qvel_gaussian=True,
    )


def _swimmer(
    backend: Backend,
    reset_noise_scale: float = 0.1,
    frame_skip: int = 4,
    **overrides: Unpack[LocomotionOptions],
) -> MjxEnv:
    """Build the ``Swimmer-v5`` MJX environment (no termination)."""
    return _build(
        "swimmer.xml",
        dataclasses.replace(
            Locomotion(
                forward_reward_weight=1.0,
                ctrl_cost_weight=1e-4,
                exclude_positions=2,
            ),
            **overrides,
        ),
        frame_skip,
        reset_noise_scale,
        backend,
    )


Builder = Callable[..., MjxEnv]


_REGISTRY: dict[str, Builder] = {
    "Hopper-v5": _hopper,
    "Walker2d-v5": _walker2d,
    "HalfCheetah-v5": _half_cheetah,
    "Swimmer-v5": _swimmer,
}


def make(
    name: str,
    impl: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
    **kwargs: Unpack[MakeOptions],
) -> MjxEnv:
    """Create an MJX environment adapter.

    Args:
        name: Environment name (e.g. ``"Hopper-v5"``).
        impl: Physics backend, ``None``/``"jax"`` (default XLA) or ``"warp"``
            (NVIDIA Warp; see module docstring caveats).
        nconmax: Max contacts per step (Warp only).
        njmax: Max constraints per step (Warp only).
        **kwargs: Task/reset overrides (see the builder and ``Locomotion``).

    Returns:
        MjxEnv instance.

    Raises:
        ValueError: If ``name`` is not a registered MJX environment.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown MJX environment {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](Backend(impl, nconmax, njmax), **kwargs)
