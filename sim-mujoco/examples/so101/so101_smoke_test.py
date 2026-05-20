"""Smoke test: spawn SO101 eval environments and hold at home position.

Usage:
    python so101_smoke_test.py --eval1
    python so101_smoke_test.py --eval2
    python so101_smoke_test.py --eval3
"""

import argparse

import numpy as np
import gymnasium as gym
from rcs._core.sim import SimConfig
from rcs.envs.configs import SO101Eval1, SO101Eval2, SO101Eval3

import rcs.envs.configs  # noqa: F401 - registers gym environments


_EVAL_REGISTRY = {
    "eval1": ("rcs/so101_eval1", SO101Eval1),
    "eval2": ("rcs/so101_eval2", SO101Eval2),
    "eval3": ("rcs/so101_eval3", SO101Eval3),
}

# Zero relative cartesian action: no translation, identity quaternion, gripper closed
_ZERO_ACTION = {"robot": {"tquat": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), "gripper": np.array([0.0])}}


def make_env(eval_key: str) -> gym.Env:
    gym_id, scene_cls = _EVAL_REGISTRY[eval_key]
    scene = scene_cls()
    cfg = scene.config()
    cfg.sim_cfg = SimConfig(realtime=True, async_control=False, max_convergence_steps=5000)
    return gym.make(gym_id, cfg=cfg, disable_env_checker=True)


def main():
    parser = argparse.ArgumentParser(description="SO101 eval environment smoke test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval1", action="store_true", help="Single cube + bowl")
    group.add_argument("--eval2", action="store_true", help="Two adjacent cubes + bowl")
    group.add_argument("--eval3", action="store_true", help="Four cubes in 2×2 grid + bowl")
    args = parser.parse_args()

    eval_key = next(k for k in ("eval1", "eval2", "eval3") if getattr(args, k))
    print(f"Launching {eval_key} environment...")

    env = make_env(eval_key)
    obs, info = env.reset()

    print("Environment ready.")
    print("Action space :", env.action_space)
    print("Observation space:", env.observation_space)

    sim = env.get_wrapper_attr("sim")
    step = 0
    while sim._gui_process is not None and sim._gui_process.is_alive():
        obs, reward, terminated, truncated, info = env.step(_ZERO_ACTION)
        step += 1
        if terminated or truncated:
            print(f"Episode ended at step {step}. Resetting.")
            obs, info = env.reset()
            step = 0

    env.close()


if __name__ == "__main__":
    main()
