"""Debug script: drive arm to a target joint pose and print EE frame position each step.

Usage (from isaac_so_arm101/):
    python src/isaac_so_arm101/scripts/debug/ee_frame_debug.py --num_envs 1

Edit TARGET_JOINT_POS below to change the target pose.
Every PRINT_EVERY steps prints:
  - gripper_link world position
  - end_effector FrameTransformer world position
  - live offset between them (should match OffsetCfg(pos=[0.01, 0.0, -0.09]))
  - gripper joint angle
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="EE frame position debug.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_so_arm101.tasks  # noqa: F401
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation
from isaaclab_tasks.utils import parse_env_cfg

# ── Target pose (rad) — edit freely ──────────────────────────────────────────
TARGET_JOINT_POS = {
                "shoulder_pan": 0.0,
                "shoulder_lift": -0.4,
                "elbow_flex": -0.3,
                "wrist_flex": 1.57,
                "wrist_roll": -1.57,
                "gripper": 0.2,
}


ARM_SCALE     = 2.5
GRIPPER_SCALE = 2.5
PRINT_EVERY   = 20
TASK          = "Isaac-SO-ARM101-Task-One-Teacher-Play-v0"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(TASK, cfg=env_cfg)
    raw_env = env.unwrapped
    device = raw_env.device

    robot: Articulation = raw_env.scene["robot"]

    env.reset()

    # Read default joint positions from the robot (populated after first reset).
    # With use_default_offset=True, action = (target - default) / scale.
    default_q = robot.data.default_joint_pos[0]  # [num_joints]

    arm_names     = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    arm_ids, _    = robot.find_joints(arm_names)
    gripper_ids, _ = robot.find_joints(["gripper"])

    target_arm     = torch.tensor([TARGET_JOINT_POS[n] for n in arm_names], device=device)
    target_gripper = torch.tensor([TARGET_JOINT_POS["gripper"]], device=device)

    action_arm     = (target_arm     - default_q[arm_ids])     / ARM_SCALE
    action_gripper = (target_gripper - default_q[gripper_ids]) / GRIPPER_SCALE

    # Action order in the env: arm_action (5) then gripper_action (1) — alphabetical by cfg name.
    actions = torch.cat([action_arm, action_gripper]).unsqueeze(0).expand(raw_env.num_envs, -1)

    print(f"[INFO] Driving to target joint pos with action vector: {actions[0].tolist()}")

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)
            step += 1

            if step % PRINT_EVERY != 0:
                continue

            ee_frame: FrameTransformer = raw_env.scene["ee_frame"]
            ee_pos_w      = ee_frame.data.target_pos_w[0, 0, :].cpu()

            body_idx       = robot.find_bodies(["gripper_link"])[0][0]
            gripper_pos_w  = robot.data.body_pos_w[0, body_idx, :].cpu()

            gripper_jid    = robot.find_joints(["gripper"])[0][0]
            gripper_rad    = robot.data.joint_pos[0, gripper_jid].item()
            actual_q       = robot.data.joint_pos[0, arm_ids].cpu()

            offset = ee_pos_w - gripper_pos_w

            print(
                f"[step {step:5d}] "
                f"gripper_link=({gripper_pos_w[0]:.4f}, {gripper_pos_w[1]:.4f}, {gripper_pos_w[2]:.4f})  "
                f"ee=({ee_pos_w[0]:.4f}, {ee_pos_w[1]:.4f}, {ee_pos_w[2]:.4f})  "
                f"offset=({offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f})  "
                f"gripper={gripper_rad:+.3f}rad"
            )
            print(
                f"         arm_q  actual=[" +
                ", ".join(f"{v:.3f}" for v in actual_q.tolist()) + "]"
            )
            print(
                f"         arm_q  target=[" +
                ", ".join(f"{v:.3f}" for v in target_arm.tolist()) + "]"
            )
            print(
                f"         arm_q   error=[" +
                ", ".join(f"{v:+.3f}" for v in (target_arm.cpu() - actual_q).tolist()) + "]"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
