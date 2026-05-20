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
from abc import ABC, abstractmethod
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
        print(self._module.code)
        self._module.eval()
        print(list(self._module.named_parameters()))
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


# IsaacLab training defaults (JointPositionAction with use_default_offset=True)
_Q_DEFAULT_ARM     = np.array([0.0, -1.4, 0.4, 1.4, -1.57], dtype=np.float32)
_Q_DEFAULT_GRIPPER = np.array([0.2],                          dtype=np.float32)
_ARM_SCALE         = 0.5
_CUBE_JOINT        = "PickTask_box_joint"
_HEIGHT_OFFSET     = 0.12  # IsaacLab adds this to bowl z in the obs


class SO101JointPolicy(TorchJITPolicy):
    """IsaacLab RSL-RL joint-position policy for the SO101 pick-and-place tasks.

    Builds the 27-dim observation expected by the trained policy::

        joint_pos_rel(6) | joint_vel(6) | object_pos(3) |
        initial_object_pos(3) | bowl_pos(3) | last_action(6)

    Applies IsaacLab action scaling on the network output:
        arm_targets  = q_default_arm + 0.5 * raw[:5]
        gripper      = 1.0 if raw[5] > 0 else 0.0

    Parameters
    ----------
    env:
        The gymnasium environment (needed for sim access inside predict()).
    bowl_xyz:
        Bowl position [x, y, z] in the shared/robot frame (meters).
        Matches the ``--bowl-xyz`` CLI argument.
    device:
        Torch device string, e.g. "cpu" or "cuda:0".
    """

    def __init__(self, env: gym.Env, bowl_xyz: list[float], device: str = "cpu"):
        super().__init__(device=device)
        self._env = env
        bowl = np.asarray(bowl_xyz, dtype=np.float32)
        # IsaacLab adds height_offset to bowl z for the obs
        self._bowl_pos_obs = np.array(
            [bowl[0], bowl[1], bowl[2] + _HEIGHT_OFFSET], dtype=np.float32
        )
        self._initial_object_pos = np.zeros(3, dtype=np.float32)
        self._last_raw_action    = np.zeros(6, dtype=np.float32)

    def on_reset(self, obs: dict[str, Any], env: gym.Env) -> None:
        """Capture initial cube position and clear last-action buffer.

        Must be called once per episode, right after env.reset() returns.
        """
        sim = env.get_wrapper_attr("sim")
        self._initial_object_pos = np.array(
            sim.data.joint(_CUBE_JOINT).qpos[:3], dtype=np.float32
        )
        self._last_raw_action = np.zeros(6, dtype=np.float32)

    def _build_input(self, obs: dict[str, Any]):
        import torch

        sim = self._env.get_wrapper_attr("sim")

        # --- joint_pos_rel (6) ---
        arm_joints  = np.asarray(obs["robot"]["joints"], dtype=np.float32)   # (5,)
        gripper_raw = np.float32(sim.data.joint("robot6").qpos[0])            # scalar
        joint_pos   = np.concatenate([arm_joints, [gripper_raw]])             # (6,)
        joint_pos_rel = joint_pos - np.concatenate(
            [_Q_DEFAULT_ARM, _Q_DEFAULT_GRIPPER]
        )

        # --- joint_vel (6) ---
        joint_vel = np.asarray(obs["robot"]["joint_vel"], dtype=np.float32)  # (6,)

        # --- object_pos (3) ---
        object_pos = np.array(
            sim.data.joint(_CUBE_JOINT).qpos[:3], dtype=np.float32
        )

        vec = np.concatenate([
            joint_pos_rel,             # 6
            joint_vel,                 # 6
            object_pos,                # 3
            self._initial_object_pos,  # 3
            self._bowl_pos_obs,        # 3
            self._last_raw_action,     # 6
        ])  # total: 27

        return torch.from_numpy(vec).unsqueeze(0).to(self.device)  # (1, 27)

    def _parse_output(self, output) -> dict[str, Any]:
        raw = output.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (6,)
        self._last_raw_action = raw.copy()

        arm_targets = (_Q_DEFAULT_ARM + _ARM_SCALE * raw[:5]).astype(np.float64)
        gripper     = np.array([1.0 if raw[5] > 0.0 else 0.0], dtype=np.float32)

        return {
            "robot": {
                "joints":  arm_targets,
                "gripper": gripper,
            }
        }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env(
    eval_key: str,
    *,
    headless: bool = True,
    realtime: bool = False,
    cube_color: str | None = None,
    cube_colors: list[str] | None = None,
    bowl_xyz: list[float] | None = None,
) -> gym.Env:
    """Build a gymnasium-wrapped SO101 eval environment.

    Parameters
    ----------
    eval_key:   "eval1", "eval2", or "eval3"
    headless:   Run without GUI (faster; set False to watch the rollout).
    realtime:   Throttle simulation to real-time speed.
    cube_color: (Eval 1 only) Pin the cube color, e.g. "red". None = random.
    cube_colors:(Eval 2/3) Pin the cube color list. None = random.
    bowl_xyz:   Override bowl position [x, y, z] in robot frame (meters).
    """
    bowl_pose = (
        rcs.common.Pose(translation=np.array(bowl_xyz, dtype=float))
        if bowl_xyz is not None
        else None
    )

    if eval_key == "eval1":
        scene = SO101Eval1(cube_color=cube_color, bowl_pose=bowl_pose)
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
    cfg.sim_cfg = SimConfig(
        realtime=realtime,
        async_control=False,
        frequency=50.0,
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
) -> dict[str, Any]:
    """Execute one episode and return a result dict.

    Parameters
    ----------
    env:          A ready-made gymnasium environment.
    policy:       Policy instance; predict() is called each step.
    max_steps:    Hard step limit per episode.
    warm_up_steps:Steps at the start where the arm holds its reset pose so the
                  simulation settles before the policy takes over.

    Returns
    -------
    dict with keys: success (bool), total_reward (float), steps (int),
                    terminated (bool), truncated (bool).
    """
    obs, info = env.reset()

    # Give policy a chance to capture reset state (e.g. initial cube position)
    if hasattr(policy, "on_reset"):
        policy.on_reset(obs, env)

    total_reward = 0.0

    # Let the arm settle at the reset pose before the policy starts acting
    for _ in range(warm_up_steps):
        obs, r, terminated, truncated, info = env.step(_hold_action(obs))
        total_reward += float(r)
        if terminated or truncated:
            break

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
) -> None:
    """Run *n_rollouts* episodes and print a summary to stdout."""
    results = []
    for i in range(n_rollouts):
        result = run_rollout(env, policy, max_steps=max_steps)
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

    # Rollout settings
    p.add_argument("--n-rollouts", type=int, default=5, metavar="N",
                   help="Number of episodes to run (default: 5).")
    p.add_argument("--max-steps", type=int, default=500, metavar="N",
                   help="Maximum steps per episode (default: 500).")
    p.add_argument("--headless", action="store_true", default=False,
                   help="Disable GUI (faster; required on servers).")
    p.add_argument("--no-headless", dest="headless", action="store_false",
                   help="Enable GUI visualization (default).")
    p.add_argument("--realtime", action="store_true",
                   help="Throttle simulation to real-time speed.")
    p.add_argument("--device", default="cpu",
                   help="Torch device for TorchJITPolicy (default: cpu).")

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

    eval_key = next(k for k in ("eval1", "eval2", "eval3") if getattr(args, k))
    logger.info("Eval: %s  rollouts: %d  headless: %s", eval_key, args.n_rollouts, args.headless)

    env = make_env(
        eval_key,
        headless=args.headless,
        realtime=args.realtime,
        cube_color=args.target_color,
        cube_colors=args.target_colors,
        bowl_xyz=args.bowl_xyz,
    )

    # Default bowl position matches IsaacLab training setup
    bowl_xyz: list[float] = args.bowl_xyz if args.bowl_xyz is not None else [0.35, 0.20, 0.003]

    if args.random:
        policy: Policy = RandomPolicy(env.action_space)
    else:
        policy = SO101JointPolicy(env=env, bowl_xyz=bowl_xyz, device=args.device)
        policy.load(args.policy)

    try:
        run_rollouts(env, policy, n_rollouts=args.n_rollouts, max_steps=args.max_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
