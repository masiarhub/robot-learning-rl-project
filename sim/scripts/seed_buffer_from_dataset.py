"""
Seed the SAC replay buffer with teleop demonstrations recorded via
`lerobot-record` (LeRobotDataset format).

Each episode in the dataset is replayed inside the simulator using
control_mode="pd_joint_pos" so we get back valid (obs, action, reward,
next_obs, done) tuples that match what `train_sim.py` produces.

Outputs a `.pt` file you can load via:
    rb_data = torch.load("seed_buffer.pt")
    rb.extend(rb_data)

Spec note: Eval 2 / Eval 3 explicitly encourage using teleoperation data for
training efficiency. Use this with `--checkpoint=...` and a fresh wandb run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

import envs  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True, help="LeRobotDataset directory or HF repo id")
    p.add_argument("--env_id", default="SO101PlaceBowlCube-v1")
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--out", default="seed_buffer.pt")
    args = p.parse_args()

    # Lazy import; not strictly needed if the user only runs train_sim.py.
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as e:
        raise SystemExit(
            "lerobot must be installed to read LeRobotDataset. "
            "Activate the `lerobot` conda env or `pip install lerobot[feetech]`."
        ) from e

    ds = LeRobotDataset(args.dataset_path) if Path(args.dataset_path).exists() else LeRobotDataset.from_hub(args.dataset_path)
    print(f"[seed] loaded dataset with {len(ds)} frames")

    env = gym.make(args.env_id, num_envs=args.num_envs, obs_mode="rgb+segmentation+state",
                   control_mode="pd_joint_pos")

    transitions = []
    obs, _ = env.reset(seed=0)
    for episode_idx in range(ds.num_episodes):
        ep_frames = ds.get_episode(episode_idx) if hasattr(ds, "get_episode") else []
        # Replay action stream
        for f in ep_frames:
            action = torch.as_tensor(f["action"], dtype=torch.float32)
            if action.dim() == 1:
                action = action.unsqueeze(0).expand(args.num_envs, -1)
            next_obs, reward, term, trunc, info = env.step(action)
            transitions.append(
                TensorDict(
                    observations={"rgb": obs.get("rgb", torch.zeros(0)), "state": obs.get("state", torch.zeros(0))},
                    actions=action,
                    rewards=reward,
                    next_observations={"rgb": next_obs.get("rgb", torch.zeros(0)),
                                       "state": next_obs.get("state", torch.zeros(0))},
                    dones=(term | trunc),
                    batch_size=[args.num_envs],
                )
            )
            obs = next_obs
        obs, _ = env.reset()

    torch.save(transitions, args.out)
    print(f"[seed] saved {len(transitions)} transitions to {args.out}")


if __name__ == "__main__":
    main()
