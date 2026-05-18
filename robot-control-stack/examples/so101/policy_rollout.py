"""Run a trained policy in the SO101 MuJoCo eval environments.

This script provides:
  - A Policy abstract base class that you subclass for your own checkpoint format.
  - A ready-to-use TorchJITPolicy for TorchScript (.pt) checkpoints.
  - A rollout loop for Eval 1/2/3 that collects per-episode success/reward.

Usage
-----
# Run 5 rollouts of Eval 1 with a TorchScript policy (headless):
    python policy_rollout.py --eval1 --policy path/to/policy.pt \\
        --n-rollouts 5 --headless

# Eval 2 with a fixed target color:
    python policy_rollout.py --eval2 --policy path/to/policy.pt \\
        --target-color blue

# Eval 3 with a custom bowl position (x y z in meters, robot frame):
    python policy_rollout.py --eval3 --policy path/to/policy.pt \\
        --bowl-xyz 0.35 0.15 0.003

# Dry-run with a random-action policy (no checkpoint needed):
    python policy_rollout.py --eval1 --random --n-rollouts 3

Subclassing Policy
------------------
Implement the two abstract methods:

    class MyPolicy(Policy):
        def load(self, path: str) -> None:
            self.model = ...  # load your checkpoint

        def predict(self, obs: dict) -> dict:
            # obs["robot"] contains: tquat (7,), joints (5,), xyzrpy (6,),
            #                        gripper (1,), frames (camera dict)
            arm_action = ...  # shape (7,)  [dx, dy, dz, qx, qy, qz, qw]
            gripper_cmd = ... # shape (1,)  0.0=close  1.0=open
            return {"robot": {"tquat": arm_action, "gripper": gripper_cmd}}

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


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

_ZERO_ACTION: dict[str, Any] = {
    "robot": {
        "tquat":   np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "gripper": np.array([0.0]),
    }
}


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
        max_convergence_steps=5000,
    )
    return gym.make(gym_id, cfg=cfg, disable_env_checker=True)


# ---------------------------------------------------------------------------
# Rollout logic
# ---------------------------------------------------------------------------

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
    warm_up_steps:Steps at the start where a zero-action is sent so the arm
                  settles at the home pose before the policy takes over.

    Returns
    -------
    dict with keys: success (bool), total_reward (float), steps (int),
                    terminated (bool), truncated (bool).
    """
    obs, info = env.reset()
    total_reward = 0.0

    # Let the arm settle at home before the policy starts acting
    for _ in range(warm_up_steps):
        obs, r, terminated, truncated, info = env.step(_ZERO_ACTION)
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
    p.add_argument("--headless", action="store_true",
                   help="Disable GUI (faster; required on servers).")
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
                   help="Override bowl position in robot frame (meters), e.g. 0.35 0.20 0.003.")

    # TorchJITPolicy obs keys
    p.add_argument("--obs-keys", nargs="+",
                   default=["tquat", "joints", "gripper"],
                   metavar="KEY",
                   help="Proprioceptive obs keys fed to TorchJITPolicy (default: tquat joints gripper).")

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

    if args.random:
        policy: Policy = RandomPolicy(env.action_space)
    else:
        policy = TorchJITPolicy(obs_keys=args.obs_keys, device=args.device)
        policy.load(args.policy)

    try:
        run_rollouts(env, policy, n_rollouts=args.n_rollouts, max_steps=args.max_steps)
    finally:
        env.close()


if __name__ == "__main__":
    main()
