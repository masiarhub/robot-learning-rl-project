"""One-shot probe: reset the SO101Eval1 env and dump the observation structure.

Run:
    python _probe_obs_keys.py
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym

from rcs._core.sim import SimConfig
from rcs.envs.configs import SO101Eval1
import rcs.envs.configs  # noqa: F401 – registers gym ids


def _summarize(value):
    if isinstance(value, np.ndarray):
        return f"ndarray shape={value.shape} dtype={value.dtype}"
    return f"{type(value).__name__}: {value!r}"


def main() -> None:
    scene = SO101Eval1()
    cfg = scene.config()
    cfg.headless = True
    cfg.sim_cfg = SimConfig(
        realtime=False, async_control=False, frequency=50.0, max_convergence_steps=5000
    )
    env = gym.make("rcs/so101_eval1", cfg=cfg, disable_env_checker=True)

    obs, info = env.reset()

    print("=== top-level obs keys ===")
    print(list(obs.keys()))

    if "robot" in obs:
        print("\n=== obs['robot'] keys ===")
        for k, v in obs["robot"].items():
            print(f"  {k}: {_summarize(v)}")
    else:
        print("\nWARNING: no 'robot' key in obs")

    has_joint_vel = isinstance(obs.get("robot"), dict) and "joint_vel" in obs["robot"]
    print(f"\nobs['robot']['joint_vel'] present? -> {has_joint_vel}")
    print(f"obs['robot']['joints'] after reset = {np.asarray(obs['robot']['joints'])}")
    print(f"expected _Q_DEFAULT_ARM           = [0.0, -1.4, 0.4, 1.4, -1.57]")

    env.close()


if __name__ == "__main__":
    main()
