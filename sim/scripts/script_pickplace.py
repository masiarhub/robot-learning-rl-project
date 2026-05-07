"""
4-waypoint scripted controller for SO101PlaceBowlCube-v1.

Useful for:
  - Sanity-checking the env (geometry, friction, reward shape) before kicking
    off a 30-minute SAC run.
  - Generating synthetic demonstrations to seed the SAC replay buffer or to
    pre-train a BC policy.

Strategy: open-loop, joint-target deltas chosen heuristically via privileged
state (item / bowl / TCP poses). Not a real teleop — just enough to validate
that the "pick up cube, drop it in bowl" semantics work in sim.

Run:
    python -m sim.scripts.script_pickplace
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import envs  # noqa: F401  (registers SO101* envs)


def _waypoint_action(env, phase: str):
    """Pick a delta-joint action that nudges the TCP toward a waypoint.

    This is intentionally crude: a cleaner version would use the env's IK,
    but for a sanity-check we just bias the wrist-flex/elbow-flex joints
    based on tcp-to-target offset.
    """
    base = env.unwrapped
    tcp = base.agent.tcp_pos[0].cpu().numpy()
    item = base.item.pose.p[0].cpu().numpy()
    bowl = base.bowl.pose.p[0].cpu().numpy()

    if phase == "above_item":
        target = item.copy(); target[2] = item[2] + 0.05
    elif phase == "down_item":
        target = item.copy(); target[2] = item[2] + 0.005
    elif phase == "above_bowl":
        target = bowl.copy(); target[2] = bowl[2] + 0.10
    else:  # "drop"
        target = bowl.copy(); target[2] = bowl[2] + 0.04

    delta = target - tcp
    # 6-dof delta-joint action; map xy/z roughly to shoulder_pan / shoulder_lift / elbow_flex
    a = np.zeros(6, dtype=np.float32)
    a[0] = np.clip(delta[1] * 5.0, -1, 1)   # pan ~ y
    a[1] = np.clip(-delta[2] * 5.0, -1, 1)  # lift ~ -z
    a[2] = np.clip(delta[0] * 5.0, -1, 1)   # elbow ~ x
    a[5] = +1.0 if phase in ("above_item", "above_bowl", "drop") else -1.0  # gripper
    return a


def main():
    env = gym.make("SO101PlaceBowlCube-v1", num_envs=1, obs_mode="state",
                   domain_randomization=False)
    obs, _ = env.reset(seed=0)

    schedule = (
        ("above_item", 12),
        ("down_item", 8),
        ("above_bowl", 12),
        ("drop", 18),
    )

    successes = 0
    n_episodes = 4
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for phase, steps in schedule:
            for _ in range(steps):
                a = _waypoint_action(env, phase)
                a_batch = torch.tensor(a).unsqueeze(0)
                obs, r, term, trunc, info = env.step(a_batch)
        info_eval = env.unwrapped.evaluate()
        success = bool(info_eval["success"][0])
        successes += int(success)
        print(f"[ep {ep}] success={success}")
    print(f"Scripted controller: {successes}/{n_episodes} successes")


if __name__ == "__main__":
    main()
