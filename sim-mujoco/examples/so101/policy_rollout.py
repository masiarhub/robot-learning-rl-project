"""Run a trained policy in the SO101 MuJoCo eval environments.

This script provides:
  - A Policy abstract base class that you subclass for your own checkpoint format.
  - A ready-to-use TorchJITPolicy for TorchScript (.pt) checkpoints.
  - SO101JointPolicy for RSL-RL checkpoints trained in IsaacLab with joint control.
  - A rollout loop for Eval 1/2/3 that collects per-episode success/reward.

Usage
-----
# Run 5 rollouts of Eval 1 with an IsaacLab RSL-RL policy:
    python policy_rollout.py --eval1 --policy path/to/policy.pt \\
        --n-rollouts 5

# Non-headless (with GUI):
    python policy_rollout.py --eval1 --policy path/to/policy.pt --no-headless

# Eval 2 with a fixed target color:
    python policy_rollout.py --eval2 --policy path/to/policy.pt \\
        --target-color blue

# Eval 3 with a custom bowl position (x y z in meters, robot/shared frame):
    python policy_rollout.py --eval3 --policy path/to/policy.pt \\
        --bowl-xyz 0.35 0.15 0.003

# Dry-run with a random-action policy (no checkpoint needed):
    python policy_rollout.py --eval1 --random --n-rollouts 3

See docs/so101_obs_action_spaces.md for the full space specification.
"""

from __future__ import annotations

import argparse
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import gymnasium as gym
from rcs._core.sim import SimConfig
from rcs.envs.configs import SO101Eval1, SO101Eval2, SO101Eval3
import rcs.envs.configs  # noqa: F401 – registers gym environments
import rcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Per-checkpoint configuration
# ---------------------------------------------------------------------------

@dataclass
class IsaacLabPolicyConfig:
    """Describes how a specific IsaacLab checkpoint was trained.

    Used to auto-configure IsaacLabJointPolicy and make_env() without any
    manual flag passing: load the checkpoint → detect the config → everything
    else follows automatically.
    """
    name: str
    default_joint_pos: np.ndarray   # shape (6,): 5 arm joints + 1 gripper, radians
    arm_action_scale: float
    gripper_action_scale: float     # ignored when binary_gripper=True
    binary_gripper: bool
    obs_variant: str                # "without_fk" | "no_init_cube" | "only_init_cube"
    cube_x_center: float            # x centre of cube spawn range in robot frame, metres
    cube_x_width: float             # total x width of cube spawn range, metres
    cube_y_width: float             # total y width of cube spawn range, metres
    max_steps: int                  # episode_length_s × 50 Hz


# Registry: checkpoint filename stem → config.
POLICY_CONFIGS: dict[str, IsaacLabPolicyConfig] = {
    "policy_no_camera_20260519": IsaacLabPolicyConfig(
        name="without_fk",
        default_joint_pos=np.array([0.0, -1.4, 0.4, 1.4, -1.57, 0.2], dtype=np.float32),
        arm_action_scale=0.5,
        gripper_action_scale=0.3,
        binary_gripper=False,
        obs_variant="without_fk",
        cube_x_center=0.248,
        cube_x_width=0.20,
        cube_y_width=0.40,
        max_steps=400,
    ),
    "policy_no_camera_binary_gripper": IsaacLabPolicyConfig(
        name="binary_gripper",
        default_joint_pos=np.array([0.0, -0.6, -0.6, 1.57, -1.57, 0.0], dtype=np.float32),
        arm_action_scale=0.5,
        gripper_action_scale=0.0,   # unused; binary_gripper=True
        binary_gripper=True,
        obs_variant="no_init_cube",
        cube_x_center=0.25,
        cube_x_width=0.20,
        cube_y_width=0.50,
        max_steps=250,
    ),
    "policy_no_camera_continuous_gripper": IsaacLabPolicyConfig(
        name="continuous_gripper",
        default_joint_pos=np.array([0.0, -0.4, -0.3, 1.57, -1.57, 0.3], dtype=np.float32),
        arm_action_scale=0.5,
        gripper_action_scale=0.5,
        binary_gripper=False,
        obs_variant="no_init_cube",
        cube_x_center=0.25,
        cube_x_width=0.20,
        cube_y_width=0.50,
        max_steps=250,
    ),
    "policy_no_camera_only_init_cube_obs": IsaacLabPolicyConfig(
        name="only_init_cube",
        default_joint_pos=np.array([0.0, -0.4, -0.3, 1.57, -1.57, 0.2], dtype=np.float32),
        arm_action_scale=0.5,
        gripper_action_scale=0.3,
        binary_gripper=False,
        obs_variant="only_init_cube",
        cube_x_center=0.248,
        cube_x_width=0.20,
        cube_y_width=0.40,
        max_steps=400,
    ),
}

_DEFAULT_POLICY_CONFIG = POLICY_CONFIGS["policy_no_camera_20260519"]


def detect_policy_config(checkpoint_path: str) -> IsaacLabPolicyConfig:
    """Return the IsaacLabPolicyConfig matching the checkpoint filename, or the default."""
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if stem in POLICY_CONFIGS:
        cfg = POLICY_CONFIGS[stem]
        logger.info("Auto-detected policy config '%s' for checkpoint '%s'", cfg.name, stem)
        return cfg
    logger.warning(
        "No policy config found for checkpoint '%s'; using default ('%s'). "
        "Add an entry to POLICY_CONFIGS to suppress this warning.",
        stem, _DEFAULT_POLICY_CONFIG.name,
    )
    return _DEFAULT_POLICY_CONFIG


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------

class Policy(ABC):
    """Abstract base class for policies that run inside the RCS MuJoCo envs."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load checkpoint from *path*. Called once before rollouts start."""

    @abstractmethod
    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return an action dict given the full observation dict.

        The returned dict must follow the RCS action format::

            {
                "robot": {
                    "tquat":   np.ndarray shape (7,) [dx, dy, dz, qx, qy, qz, qw],
                    "gripper": np.ndarray shape (1,)  0.0=close  1.0=open,
                }
            }
        """


class RandomPolicy(Policy):
    """Random-action policy. Useful as a sanity-check baseline."""

    def __init__(self, action_space: gym.spaces.Dict):
        self._action_space = action_space

    def load(self, path: str) -> None:
        pass  # nothing to load

    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._action_space.sample()


class TorchJITPolicy(Policy):
    """Load a TorchScript (.pt) checkpoint exported with torch.jit.save().

    The scripted module must accept a single 1-D float32 tensor (the flattened
    proprioceptive observation) and return a 1-D float32 tensor of length 8
    (7 arm + 1 gripper).

    If your model expects a different input (e.g. images + proprioception as
    separate tensors), subclass this and override *_build_input* and
    *_parse_output* instead of re-implementing the whole class.

    Parameters
    ----------
    obs_keys:
        Which keys from ``obs["robot"]`` to stack into the proprioceptive input
        vector. Defaults to ["tquat", "joints", "gripper"] → 13-dim vector.
    device:
        Torch device string, e.g. "cpu" or "cuda:0".
    """

    def __init__(
        self,
        obs_keys: list[str] | None = None,
        device: str = "cpu",
    ):
        self.obs_keys = obs_keys or ["tquat", "joints", "gripper"]
        self.device = device
        self._module = None

    def load(self, path: str) -> None:
        import torch  # lazy import – only needed when this policy is used

        self._module = torch.jit.load(path, map_location=self.device)
        self._module.eval()
        logger.info("Loaded TorchScript policy from %s", path)

    def _build_input(self, obs: dict[str, Any]):
        """Flatten proprioceptive keys into a single float32 tensor."""
        import torch

        robot_obs = obs["robot"]
        parts = [np.asarray(robot_obs[k], dtype=np.float32).ravel() for k in self.obs_keys]
        vec = np.concatenate(parts)
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)  # (1, D)

    def _parse_output(self, output) -> dict[str, Any]:
        """Convert the model output tensor to an RCS action dict."""
        arr = output.squeeze(0).detach().cpu().numpy()
        return {
            "robot": {
                "tquat":   arr[:7].astype(np.float64),
                "gripper": arr[7:8].astype(np.float32),
            }
        }

    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._module is None:
            raise RuntimeError("Call load() before predict().")
        inp = self._build_input(obs)
        with __import__("torch").no_grad():
            out = self._module(inp)
        return self._parse_output(out)


class IsaacLabJointPolicy(Policy):
    """Load a TorchScript checkpoint trained in Isaac Lab (task_1/no_camera/init_pos_hover).

    Mirrors the exact obs/action setup in
    Sim-to-Real/task_1/task_1_no_camera/20260519/2026-05-19_11-29-20_init_pos_hover/params/env.yaml:

    Obs (27, in order):
      [0:6]   joint_pos_rel = joint_pos - DEFAULT_JOINT_POS  (5 arm + 1 gripper)
      [6:12]  joint_vel_rel = joint_vel  (default vel is 0)
      [12:15] object_position = current cube xyz in robot root frame  (updates every step)
      [15:18] initial_object_position = cube xyz captured at episode reset, in robot root frame
      [18:21] bowl_position = bowl xyz + (0, 0, 0.12) hover offset, in robot root frame
      [21:27] last_action   = raw policy output from previous step (6,)

    Verified against normalizer statistics stored in the checkpoint:
      obs[15:18] std_z = 0.000  → z never changes → initial cube pos on table ✓
      obs[12:15] mean_z = 0.066 → cube on table + occasionally lifted ✓  (EE hover z ≈ 0.25 ✗)

    Action (6):
      [0:5] arm joint targets:    target = DEFAULT[:5] + action[:5] * 0.5
      [5]   gripper joint target: target = 0.2          + action[5]  * 0.3
            (binarised to RCS open/close because the env uses binary gripper)

    Requires:
      - env created with ControlMode.JOINTS and RelativeTo.NONE (absolute targets)
      - q_home overridden to DEFAULT_JOINT_POS so the robot starts at IL's init pose
      - call set_env() before rollouts so _parse_output can read live joint state
      - call reset_episode() after env.reset() to freeze the initial cube position
    """

    # task_1/no_camera/init_pos_hover (matches normalizer means/stds in policy.pt)
    DEFAULT_JOINT_POS = np.array([0.0, -1.4, 0.4, 1.4, -1.57, 0.2], dtype=np.float32)
    ARM_ACTION_SCALE = 0.5
    GRIPPER_ACTION_SCALE = 0.3
    BOWL_HOVER_HEIGHT = 0.12

    # MuJoCo robot base sits at world z = −0.03 m; IsaacLab robot base is at world z = 0.
    # All robot-frame z coordinates are therefore 0.03 m higher in MuJoCo than in IsaacLab.
    # Subtract this constant from every position obs before feeding to the policy so the
    # network sees the same z distribution it was trained on.
    _Z_CORRECTION: float = 0.03

    _CUBE_BODY = "PickTask_box_body"
    _BOWL_BODY = "bowl_bowl_body"
    _ROBOT_BASE_BODY = "robotbase"

    # Exponential smoothing on arm targets: smoothed = α*new + (1-α)*prev.
    # MuJoCo runs 10 substeps per policy step at dt=0.002 (vs IsaacLab's 2 substeps at
    # dt=0.01), so joints over-converge between actions → jerky motion without smoothing.
    # The kp/kv gains are tuned for dt=0.002; changing to dt=0.01 destabilises the
    # Euler integrator for lighter joints (kv*dt/I exceeds the stability threshold).
    ARM_SMOOTH_ALPHA = 0.5

    # Hard per-step joint change limit (rad).
    # Caps commanded joint velocity at MAX_JOINT_DELTA_RAD × policy_freq (50 Hz) = 75 °/s.
    MAX_JOINT_DELTA_RAD: float = np.deg2rad(1.5)

    # Velocity limit matching IsaacLab ImplicitActuatorCfg velocity_limit_sim (rad/s).
    # MuJoCo has no equivalent actuator-level cap; we log violations instead.
    VEL_LIMIT: float = 1.5

    _GRIPPER_MIN: float = -0.17453292519943295
    _GRIPPER_MAX: float = 1.7453292519943295

    def __init__(self, config: IsaacLabPolicyConfig | None = None, device: str = "cpu"):
        self._config = config if config is not None else _DEFAULT_POLICY_CONFIG
        self.device = device
        self._module = None
        self._env = None
        self._last_action = np.zeros(6, dtype=np.float32)
        self._initial_cube_pos_robot = np.zeros(3, dtype=np.float32)
        self._body_ids: dict[str, int] | None = None
        self._prev_arm_target: np.ndarray | None = None
        self._joint_qpos_adr: np.ndarray | None = None
        self._joint_qvel_adr: np.ndarray | None = None
        self._debug_logged = False

    def make_zero_action(self) -> dict[str, Any]:
        """Hold-pose action for warm-up: arm at init pose, gripper at config default."""
        gripper_default = float(self._config.default_joint_pos[5])
        gripper_norm = float(np.clip(
            (gripper_default - self._GRIPPER_MIN) / (self._GRIPPER_MAX - self._GRIPPER_MIN),
            0.0, 1.0,
        ))
        return {
            "robot": {
                "joints":  self._config.default_joint_pos[:5].astype(np.float64),
                "gripper": np.array([gripper_norm], dtype=np.float32),
            }
        }

    def set_env(self, env) -> None:
        self._env = env

    def load(self, path: str) -> None:
        import torch
        self._module = torch.jit.load(path, map_location=self.device)
        self._module.eval()
        logger.info("Loaded IsaacLab TorchScript policy from %s", path)

    def _cache_body_ids(self) -> None:
        import mujoco
        m = self._env.get_wrapper_attr("sim").model
        self._body_ids = {
            name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (self._CUBE_BODY, self._BOWL_BODY, self._ROBOT_BASE_BODY)
        }
        # Compute qpos/qvel addresses from the model so we don't rely on hardcoded
        # indices that depend on XML ordering.
        joint_names = [f"robot{i}" for i in range(1, 7)]  # robot1..robot6
        self._joint_qpos_adr = np.array(
            [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names],
            dtype=int,
        )
        self._joint_qvel_adr = np.array(
            [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names],
            dtype=int,
        )
        logger.info(
            "Joint qpos addresses: %s  (expected contiguous 6-block)", self._joint_qpos_adr
        )
        for name, bid in self._body_ids.items():
            logger.info("  body '%s' → id %d", name, bid)

    def reset_episode(self) -> None:
        """Capture the initial cube position (frozen for the rest of the episode) and clear last_action.

        Also forces the gripper joint to the IsaacLab init value (0.2 rad) — the RCS
        env's home_on_reset only moves the arm to q_home, leaving the gripper at its
        binary-open width (~1.745 rad) which is ~6σ outside the training distribution.
        """
        import mujoco
        self._last_action = np.zeros(6, dtype=np.float32)
        self._prev_arm_target = None
        if self._body_ids is None:
            self._cache_body_ids()
        sim = self._env.get_wrapper_attr("sim")
        d = sim.data
        gripper_qpos_adr = int(self._joint_qpos_adr[5])  # joint index 5 = robot6
        gripper_qvel_adr = int(self._joint_qvel_adr[5])
        d.qpos[gripper_qpos_adr] = float(self._config.default_joint_pos[5])
        d.qvel[gripper_qvel_adr] = 0.0
        # Also reset the actuator ctrl so the PD controller doesn't immediately
        # fight the qpos we just set.
        actuator_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot6")
        if actuator_id >= 0:
            d.ctrl[actuator_id] = float(self._config.default_joint_pos[5])
        sim.step(2)  # let the new gripper qpos propagate through xpos / xmat
        cube_w = d.xpos[self._body_ids[self._CUBE_BODY]].copy()
        base_w = d.xpos[self._body_ids[self._ROBOT_BASE_BODY]].copy()
        self._initial_cube_pos_robot = (cube_w - base_w).astype(np.float32)
        self._initial_cube_pos_robot[2] -= self._Z_CORRECTION

    def _build_input(self, obs: dict[str, Any]):
        import torch
        if self._body_ids is None:
            self._cache_body_ids()
        d = self._env.get_wrapper_attr("sim").data

        joint_pos = d.qpos[self._joint_qpos_adr].astype(np.float32)
        joint_vel = d.qvel[self._joint_qvel_adr].astype(np.float32)
        joint_pos_rel = joint_pos - self._config.default_joint_pos

        base_w = d.xpos[self._body_ids[self._ROBOT_BASE_BODY]]

        # Current cube position in robot root frame (updated every step).
        # Subtract _Z_CORRECTION to convert MuJoCo robot-frame z to IsaacLab robot-frame z.
        cube_pos = (d.xpos[self._body_ids[self._CUBE_BODY]] - base_w).astype(np.float32)
        cube_pos[2] -= self._Z_CORRECTION

        # Bowl position + 0.12 m z hover offset, in robot root frame.
        bowl_w = d.xpos[self._body_ids[self._BOWL_BODY]].copy()
        bowl_w[2] += self.BOWL_HOVER_HEIGHT
        bowl_pos = (bowl_w - base_w).astype(np.float32)
        bowl_pos[2] -= self._Z_CORRECTION

        variant = self._config.obs_variant
        if variant == "without_fk":
            # 27-dim: jp_rel(6) | jv(6) | cube_pos(3) | init_cube_pos(3) | bowl(3) | action(6)
            vec = np.concatenate([
                joint_pos_rel,
                joint_vel,
                cube_pos,
                self._initial_cube_pos_robot,
                bowl_pos,
                self._last_action,
            ])
        elif variant == "no_init_cube":
            # 24-dim: jp_rel(6) | jv(6) | cube_pos(3) | bowl(3) | action(6)
            vec = np.concatenate([
                joint_pos_rel,
                joint_vel,
                cube_pos,
                bowl_pos,
                self._last_action,
            ])
        elif variant == "only_init_cube":
            # 27-dim: jp_rel(6) | jv(6) | ee_pos(3) | init_cube_pos(3) | bowl(3) | action(6)
            # ee_pos from tquat (TCP in robot base frame); same z-correction applies.
            ee_pos = np.array(obs["robot"]["tquat"][:3], dtype=np.float32)
            ee_pos[2] -= self._Z_CORRECTION
            vec = np.concatenate([
                joint_pos_rel,
                joint_vel,
                ee_pos,
                self._initial_cube_pos_robot,
                bowl_pos,
                self._last_action,
            ])
        else:
            raise ValueError(f"Unknown obs_variant: {variant!r}")

        if not self._debug_logged:
            self._debug_logged = True
            pos_label = "ee_pos" if variant == "only_init_cube" else "cube_pos (current)"
            pos_val = (np.array(obs["robot"]["tquat"][:3], dtype=np.float32) - [0, 0, self._Z_CORRECTION]
                       if variant == "only_init_cube" else cube_pos)
            logger.info(
                "OBS DUMP (first step, z-corrected to IsaacLab frame)\n"
                "  variant            : %s\n"
                "  joint_pos_rel      : %s\n"
                "  joint_vel          : %s\n"
                "  %-19s: %s\n"
                "  cube_pos (initial) : %s  (target: ~[x, y, 0.01])\n"
                "  bowl_pos           : %s  (target: ~[x, y, 0.13])",
                variant,
                np.round(joint_pos_rel, 3),
                np.round(joint_vel, 3),
                pos_label, np.round(pos_val, 3),
                np.round(self._initial_cube_pos_robot, 3),
                np.round(bowl_pos, 3),
            )
        return torch.from_numpy(vec).unsqueeze(0).to(self.device)

    def _parse_output(self, output) -> dict[str, Any]:
        arr = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self._last_action = arr.copy()

        # Absolute joint targets: target = default + action * scale (use_default_offset=True)
        arm_target = (self._config.default_joint_pos[:5] + arr[:5] * self._config.arm_action_scale).astype(np.float64)

        # Seed _prev_arm_target to the actual current joint positions on the first call
        # so the rate limiter is active from step 1.  Seeding to arm_target instead would
        # make delta=0 on step 1, bypassing the rate limit for the initial (often large) move.
        if self._prev_arm_target is None:
            d = self._env.get_wrapper_attr("sim").data
            self._prev_arm_target = d.qpos[self._joint_qpos_adr[:5]].copy().astype(np.float64)

        # Smooth toward new target (reduces over-convergence artefacts at 10 substeps),
        # then hard-clip the delta to ±MAX_JOINT_DELTA_RAD.
        arm_target = self.ARM_SMOOTH_ALPHA * arm_target + (1.0 - self.ARM_SMOOTH_ALPHA) * self._prev_arm_target
        delta = arm_target - self._prev_arm_target
        delta = np.clip(delta, -self.MAX_JOINT_DELTA_RAD, self.MAX_JOINT_DELTA_RAD)
        arm_target = self._prev_arm_target + delta
        self._prev_arm_target = arm_target.copy()

        # Log if any arm joint velocity exceeds the IsaacLab training limit.
        qvel = self._env.get_wrapper_attr("sim").data.qvel[self._joint_qvel_adr[:5]]
        if np.any(np.abs(qvel) > self.VEL_LIMIT):
            logger.warning("Joint velocity limit exceeded: %s rad/s", np.round(qvel, 3))

        # Gripper: normalized to [0,1] for the continuous gripper env (binary_gripper=False).
        # BinaryJointPositionAction: threshold raw output at 0 → open (+0.5 rad) or close (−0.1 rad).
        # JointPositionAction: target = default + output * scale, then normalize to [0, 1].
        if self._config.binary_gripper:
            gripper_joint_target = 0.5 if arr[5] > 0.0 else -0.1
        else:
            gripper_joint_target = float(
                self._config.default_joint_pos[5] + arr[5] * self._config.gripper_action_scale
            )
        gripper_normalized = float(np.clip(
            (gripper_joint_target - self._GRIPPER_MIN) / (self._GRIPPER_MAX - self._GRIPPER_MIN),
            0.0, 1.0,
        ))
        return {"robot": {"joints": arm_target, "gripper": np.array([gripper_normalized], dtype=np.float32)}}

    def predict(self, obs: dict[str, Any]) -> dict[str, Any]:
        import torch
        if self._module is None:
            raise RuntimeError("Call load() before predict().")
        # Clamp joint velocities to IsaacLab training range (velocity_limit_sim=1.5 rad/s).
        # MuJoCo has no built-in velocity cap; without this, force-saturated joints
        # accumulate velocity without bound, sending observations out-of-distribution.
        if self._joint_qvel_adr is not None:
            d = self._env.get_wrapper_attr("sim").data
            d.qvel[self._joint_qvel_adr] = np.clip(
                d.qvel[self._joint_qvel_adr], -self.VEL_LIMIT, self.VEL_LIMIT
            )
        inp = self._build_input(obs)
        with torch.no_grad():
            out = self._module(inp)
        return self._parse_output(out)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

_ZERO_ACTION_CARTESIAN: dict[str, Any] = {
    "robot": {
        "tquat":   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "gripper": np.array([0.0]),
    }
}

# When the env is in JOINTS mode with RelativeTo.NONE the "joints" entry is an
# absolute target. Use the IsaacLab training default so warm-up keeps the arm at
# the same pose the policy will hand back when last_action is zero.
# Gripper default: 0.2 rad normalized to [0,1] → (0.2 - (-0.17453)) / (1.74533 - (-0.17453)) ≈ 0.193
_GRIPPER_NORM_DEFAULT = (0.2 - (-0.17453292519943295)) / (1.7453292519943295 - (-0.17453292519943295))
_ZERO_ACTION_JOINTS_ABS: dict[str, Any] = {
    "robot": {
        "joints":  IsaacLabJointPolicy.DEFAULT_JOINT_POS[:5].astype(np.float64),
        "gripper": np.array([_GRIPPER_NORM_DEFAULT], dtype=np.float32),
    }
}

_ZERO_ACTION = _ZERO_ACTION_CARTESIAN


def make_env(
    eval_key: str,
    *,
    headless: bool = True,
    realtime: bool = False,
    joint_control: bool = False,
    policy_cfg: IsaacLabPolicyConfig | None = None,
    cube_color: str | None = None,
    cube_colors: list[str] | None = None,
    bowl_xyz: list[float] | None = None,
) -> gym.Env:
    """Build a gymnasium-wrapped SO101 eval environment.

    Parameters
    ----------
    eval_key:      "eval1", "eval2", or "eval3"
    headless:      Run without GUI (faster; set False to watch the rollout).
    realtime:      Throttle simulation to real-time speed.
    joint_control: Use ControlMode.JOINTS instead of Cartesian (for IsaacLab checkpoints).
    policy_cfg:    Auto-detected IsaacLabPolicyConfig; sets q_home and cube spawn range.
    cube_color:    (Eval 1 only) Pin the cube color, e.g. "red". None = random.
    cube_colors:   (Eval 2/3) Pin the cube color list. None = random.
    bowl_xyz:      Override bowl position [x, y, z] in robot frame (meters).
    """
    from rcs.envs.base import ControlMode, RelativeTo

    bowl_pose = (
        rcs.common.Pose(translation=np.array(bowl_xyz, dtype=float))
        if bowl_xyz is not None
        else None
    )

    if eval_key == "eval1":
        scene = SO101Eval1(
            cube_color=cube_color,
            bowl_pose=bowl_pose,
            **({"cube_x_center": policy_cfg.cube_x_center,
                "cube_x_width":  policy_cfg.cube_x_width,
                "cube_y_width":  policy_cfg.cube_y_width} if policy_cfg is not None else {}),
        )
        gym_id = "rcs/so101_eval1"
    elif eval_key == "eval2":
        scene = SO101Eval2(cube_colors=cube_colors, bowl_pose=bowl_pose)
        gym_id = "rcs/so101_eval2"
    elif eval_key == "eval3":
        scene = SO101Eval3(cube_colors=cube_colors, bowl_pose=bowl_pose)
        gym_id = "rcs/so101_eval3"
    else:
        raise ValueError(f"Unknown eval key: {eval_key!r}")

    cfg = scene.config()
    cfg.headless = headless
    if joint_control:
        # IsaacLab policies emit absolute joint targets (default + action*scale).
        # Disable RelativeActionSpace so targets reach the PD controller as-is,
        # and start the arm at the policy's init pose so last_action=0 keeps it put.
        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.NONE
        q_home = (
            policy_cfg.default_joint_pos[:5].astype(np.float64)
            if policy_cfg is not None
            else IsaacLabJointPolicy.DEFAULT_JOINT_POS[:5].astype(np.float64)
        )
        for robot_cfg in cfg.robot_cfgs.values():
            robot_cfg.q_home = q_home
        # Always use continuous gripper mode: binary mode drives the joint to ±limit
        # (0 or 1.745 rad), which is 6+σ out-of-distribution and prevents grasping.
        cfg.wrapper_cfg.binary_gripper = False
        # IsaacLab trained at sim.dt=0.01 * decimation=2 → 50 Hz policy with only a
        # couple of physics steps between actions. step_until_convergence drives the
        # joints all the way to (saturated) targets each step, which is unrealistic
        # and crashes the policy out of its training distribution. Match IL: 50 Hz
        # async, two physics substeps per policy step.
        cfg.sim_cfg = SimConfig(
            realtime=realtime,
            async_control=True,
            frequency=50.0,
            max_convergence_steps=5000,
        )
    else:
        cfg.sim_cfg = SimConfig(
            realtime=realtime,
            async_control=False,
            max_convergence_steps=5000,
        )
    return gym.make(gym_id, cfg=cfg, disable_env_checker=True)


# ---------------------------------------------------------------------------
# Rollout logic
# ---------------------------------------------------------------------------

def _hold_action(obs: dict[str, Any]) -> dict[str, Any]:
    """Build a 'hold current pose' action from the current observation."""
    robot_obs = obs["robot"]
    action: dict[str, Any] = {"robot": {}}
    if "joints" in robot_obs:
        action["robot"]["joints"] = np.asarray(robot_obs["joints"], dtype=np.float64).copy()
    elif "tquat" in robot_obs:
        action["robot"]["tquat"] = np.asarray(robot_obs["tquat"], dtype=np.float64).copy()
    # Open gripper during warm-up
    action["robot"]["gripper"] = np.array([1.0], dtype=np.float32)
    return action


def run_rollout(
    env: gym.Env,
    policy: Policy,
    max_steps: int = 500,
    warm_up_steps: int = 5,
    zero_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one episode and return a result dict.

    Parameters
    ----------
    env:          A ready-made gymnasium environment.
    policy:       Policy instance; predict() is called each step.
    max_steps:    Hard step limit per episode.
    warm_up_steps:Steps at the start where a zero-action is sent so the arm
                  settles at the home pose before the policy takes over.
    zero_action:  Action dict to send during warm-up. Defaults to Cartesian zero.

    Returns
    -------
    dict with keys: success (bool), total_reward (float), steps (int),
                    terminated (bool), truncated (bool).
    """
    if zero_action is None:
        zero_action = _ZERO_ACTION_CARTESIAN
    obs, info = env.reset()

    # Give policy a chance to capture reset state (e.g. initial cube position)
    if hasattr(policy, "on_reset"):
        policy.on_reset(obs, env)

    total_reward = 0.0

    # Let the arm settle at the reset pose before the policy starts acting
    for _ in range(warm_up_steps):
        obs, r, terminated, truncated, info = env.step(zero_action)
        total_reward += float(r)
        if terminated or truncated:
            break

    # Freeze initial cube position AFTER warm-up so the policy sees the cube
    # at the same spot the env physics has settled it to.
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()

    for step in range(max_steps):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            break

    success = bool(info.get("success", False))
    return {
        "success":      success,
        "total_reward": total_reward,
        "steps":        step + 1,
        "terminated":   terminated,
        "truncated":    truncated,
    }


def run_rollouts(
    env: gym.Env,
    policy: Policy,
    n_rollouts: int,
    max_steps: int = 500,
    zero_action: dict[str, Any] | None = None,
) -> None:
    """Run *n_rollouts* episodes and print a summary to stdout."""
    results = []
    for i in range(n_rollouts):
        result = run_rollout(env, policy, max_steps=max_steps, zero_action=zero_action)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        logger.info(
            "Rollout %d/%d  %s  reward=%.3f  steps=%d",
            i + 1,
            n_rollouts,
            status,
            result["total_reward"],
            result["steps"],
        )

    successes = sum(r["success"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    logger.info(
        "\n=== Summary: %d/%d rollouts successful  avg_reward=%.3f ===",
        successes,
        n_rollouts,
        avg_reward,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained policy in the SO101 MuJoCo eval environments."
    )

    # Eval selection (mutually exclusive)
    eval_grp = p.add_mutually_exclusive_group(required=True)
    eval_grp.add_argument("--eval1", action="store_true", help="Eval 1: single cube → bowl")
    eval_grp.add_argument("--eval2", action="store_true", help="Eval 2: targeted pick in clutter")
    eval_grp.add_argument("--eval3", action="store_true", help="Eval 3: sequential multi-step")

    # Policy
    policy_grp = p.add_mutually_exclusive_group(required=True)
    policy_grp.add_argument(
        "--policy", metavar="PATH",
        help="Path to a TorchScript (.pt) checkpoint.",
    )
    policy_grp.add_argument(
        "--random", action="store_true",
        help="Use a random-action policy (baseline/dry-run, no checkpoint needed).",
    )
    p.add_argument(
        "--isaaclab", action="store_true",
        help="Treat the checkpoint as an IsaacLab RSL-RL policy (27-dim obs, 6-dim joint output).",
    )

    # Rollout settings
    p.add_argument("--n-rollouts", type=int, default=5, metavar="N",
                   help="Number of episodes to run (default: 5).")
    p.add_argument("--max-steps", type=int, default=None, metavar="N",
                   help="Maximum steps per episode. Defaults to the policy's episode length "
                        "(400 for 8 s policies, 250 for 5 s policies); fallback 500 otherwise.")
    p.add_argument("--headless", action="store_true", default=False,
                   help="Disable GUI (faster; required on servers).")
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Enable GUI visualization (default).")
    p.add_argument("--realtime", action="store_true",
                   help="Throttle simulation to real-time speed.")
    p.add_argument("--device", default="cpu",
                   help="Torch device for TorchJITPolicy (default: cpu).")
    p.add_argument("--no-smoothing", action="store_true",
                   help="Diagnostic: disable arm-target smoothing (ALPHA=1.0, no delta clip). "
                        "Use to test whether IL→MuJoCo PD smoothing is throttling the policy.")

    # Environment overrides
    p.add_argument("--target-color", metavar="COLOR",
                   help="(Eval 1) Pin cube color: red|blue|green|yellow|orange|purple.")
    p.add_argument("--target-colors", nargs="+", metavar="COLOR",
                   help="(Eval 2/3) Pin cube colors (space-separated list).")
    p.add_argument("--bowl-xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
                   help="Bowl position in shared/robot frame (meters), e.g. 0.35 0.20 0.003. "
                        "Sets both the env bowl placement and the policy bowl obs (default: 0.35 0.20 0.003).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    isaaclab = getattr(args, "isaaclab", False)
    eval_key = next(k for k in ("eval1", "eval2", "eval3") if getattr(args, k))

    # Auto-detect per-checkpoint config from the checkpoint filename.
    policy_cfg: IsaacLabPolicyConfig | None = None
    if isaaclab and args.policy:
        policy_cfg = detect_policy_config(args.policy)

    logger.info(
        "Eval: %s  rollouts: %d  headless: %s  isaaclab: %s  policy: %s",
        eval_key, args.n_rollouts, args.headless, isaaclab,
        policy_cfg.name if policy_cfg is not None else "n/a",
    )

    env = make_env(
        eval_key,
        headless=args.headless,
        realtime=args.realtime,
        joint_control=isaaclab,
        policy_cfg=policy_cfg,
        cube_color=args.target_color,
        cube_colors=args.target_colors,
        bowl_xyz=args.bowl_xyz,
    )

    if args.random:
        policy: Policy = RandomPolicy(env.action_space)
        zero_action = _ZERO_ACTION_CARTESIAN
    elif isaaclab:
        policy = IsaacLabJointPolicy(config=policy_cfg, device=args.device)
        policy.load(args.policy)
        policy.set_env(env)
        if args.no_smoothing:
            policy.ARM_SMOOTH_ALPHA = 1.0
            policy.MAX_JOINT_DELTA_RAD = float(np.pi)
            logger.info(
                "Action smoothing DISABLED (diagnostic): ALPHA=%.2f, MAX_DELTA=%.3f rad",
                policy.ARM_SMOOTH_ALPHA, policy.MAX_JOINT_DELTA_RAD,
            )
        zero_action = policy.make_zero_action()
    else:
        policy = TorchJITPolicy(device=args.device)
        policy.load(args.policy)
        zero_action = _ZERO_ACTION_CARTESIAN

    max_steps: int
    if args.max_steps is not None:
        max_steps = args.max_steps
    elif policy_cfg is not None:
        max_steps = policy_cfg.max_steps
    else:
        max_steps = 500

    try:
        run_rollouts(env, policy, n_rollouts=args.n_rollouts, max_steps=max_steps,
                     zero_action=zero_action)
    finally:
        env.close()


if __name__ == "__main__":
    main()
