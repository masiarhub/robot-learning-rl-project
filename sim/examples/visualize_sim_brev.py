#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================
# HEADLESS GPU SETUP
# =========================
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["DISPLAY"] = ""
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import argparse
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

import numpy as np
import gymnasium as gym
import cv2
import torch

import envs
import mani_skill.envs
from mani_skill.utils.visualization.misc import tile_images


# =========================
# CONFIG
# =========================

HEADLESS_CONFIG = {
    "tasks": [
        "SO101ReachCube-v1",
        "SO101LiftCube-v1",
        "SO101PlaceCube-v1",
        "SO101StackCube-v1",
    ],
    "num_envs": 1,
    "sim_backend": "auto",

    # 🔥 WICHTIG: nur RGB Observations, kein Renderer
    "obs_mode": "rgb",
    "render_mode": None,

    "domain_randomization": False,
    "steps_per_task": 30,
    "seed": 1,
    "window_size": 512,
}


# =========================
# ENV FACTORY
# =========================

def make_env(task: str, config: dict):

    env_kwargs = dict(
        obs_mode=config["obs_mode"],
        render_mode=config["render_mode"],
        num_envs=config["num_envs"],
        sim_backend=config["sim_backend"],
        domain_randomization=config["domain_randomization"],
    )

    env = gym.make(task, **env_kwargs)
    env.reset(seed=config["seed"])

    return env


# =========================
# RUNNER (HEADLESS VIDEO)
# =========================

def run_headless(config: dict):

    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)

    for task in config["tasks"]:

        print(f"\n[INFO] Running task: {task}")

        env = make_env(task, config)
        obs, _ = env.reset()

        action_shape = env.action_space.shape
        steps = config["steps_per_task"]

        video_path = os.path.join(output_dir, f"{task}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            video_path,
            fourcc,
            30,
            (config["window_size"], config["window_size"])
        )

        for step in range(steps):

            # simple dummy action
            action = np.zeros(action_shape)
            action[..., -1] = 1 if step < 15 else -1

            obs, reward, terminated, truncated, info = env.step(action)

            done = (terminated | truncated).any()

            # =========================
            # 🔥 IMPORTANT FIX
            # NO env.render()
            # ONLY OBS RGB
            # =========================
            rgb = obs["rgb"]

            # format fix: (N,H,W,3)
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu()

            if rgb.shape[-1] != 3:
                rgb = rgb[..., :3]

            # tile envs (even if 1 env)
            rgb = tile_images(rgb, nrows=1)
            rgb = rgb.numpy().astype(np.uint8)

            # resize
            rgb = cv2.resize(rgb, (config["window_size"], config["window_size"]))

            # BGR for OpenCV
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            writer.write(rgb)

            print(f"Step {step+1}/{steps} | reward={float(np.mean(reward)):.4f} | done={done}", end="\r")

            if done:
                env.reset()

        writer.release()
        env.close()

        print(f"\n[INFO] Saved video: {video_path}")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.headless:
        run_headless(HEADLESS_CONFIG)
    else:
        print("❌ This version is headless-only. Use --headless")