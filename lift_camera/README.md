# SO-ARM101 LiftCamera — Deployment Guide

Deploy the RSL-RL PPO policy trained in Isaac Lab onto the real SO-ARM101 robot.

## Files

| File | Description |
|---|---|
| `deploy_script.py` | Policy runner — builds observations, runs inference, sends actions |
| `model_2999.pt` | Trained checkpoint (lift-camera policy, iteration 2999) |
| `requirements.txt` | Pinned Python dependencies |
| `setup.sh` | One-shot environment setup script |

---

## Prerequisites

- Python 3.12 (required by LeRobot 0.5+)
- [uv](https://docs.astral.sh/uv/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- git
- SO-ARM101 connected via USB (`/dev/ttyACM0` by default)
- Wrist camera mounted on the gripper link and connected via USB

> **Before the first run:** calibrate the robot if not already done:
> ```bash
> lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_so101
> ```

---

## Setup

Run once on the deployment laptop:

```bash
cd Docs/Policies/lift_camera
bash setup.sh
```

This creates `venv_deploy/` in the same folder, installs all dependencies (auto-detects CUDA vs CPU), and runs a smoke test. Takes ~2 minutes on first run (LeRobot clones from GitHub).

---

## Running the Policy

Activate the environment first:

```bash
source Docs/Policies/lift_camera/venv_deploy/bin/activate
cd Docs/Policies/lift_camera
```

### 1. Smoke test (no robot needed)

Verifies the checkpoint loads and observation shapes are correct:

```bash
python deploy_script.py --checkpoint model_2999.pt --mock
```

### 2. First real run

Use `--max_delta_deg 1.0` on the first run — limits joint movement to 1°/step so you can catch unexpected behaviour safely:

```bash
python deploy_script.py \
    --checkpoint model_2999.pt \
    --robot_port /dev/ttyACM0 \
    --robot_id my_so101 \
    --bowl_pos 0.30 0.10 0.00 \
    --max_delta_deg 1.0 \
    --num_episodes 3
```

Once motion looks correct, increase to the normal limit:

```bash
python deploy_script.py \
    --checkpoint model_2999.pt \
    --robot_port /dev/ttyACM0 \
    --robot_id follower_arm \
    --bowl_pos 0.30 0.10 0.00 \
    --num_episodes 1
```

### 3. Verify camera view

Save one frame per second to `./camera_debug/` to confirm the wrist camera is mounted and oriented correctly:

```bash
python deploy_script.py --checkpoint model_2999.pt --mock --save_camera_frames
```

Expected view: looking slightly downward from the gripper, cube and table visible in the lower half of the frame.

---

## Key Parameters

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint file |
| `--robot_port` | `/dev/ttyACM0` | Serial port of the robot |
| `--robot_id` | `so101_follower` | Robot ID used during calibration |
| `--bowl_pos X Y Z` | `0.30 0.10 0.00` | Bowl center in robot root frame (metres) |
| `--max_delta_deg` | `3.0` | Max joint movement per step (deg) — use `1.0` for first run |
| `--num_episodes` | `5` | Number of rollouts before stopping |
| `--episode_duration` | `5.0` | Seconds per episode (matches training) |
| `--reset_duration` | `15.0` | Pause between episodes to reset the scene |
| `--camera_type` | `usb` | `usb` (OpenCV) or `realsense` (Intel RealSense) |
| `--camera_device` | `0` | USB camera device index (`/dev/video0`) |
| `--mock` | off | Run without a real robot (inference only) |
| `--save_camera_frames` | off | Save camera frames to `./camera_debug/` |

---

## Measuring `--bowl_pos`

The bowl position must be given in the **robot root frame** (metres): x = forward, y = left, z = up, origin at the robot base.

Measure with a ruler from the centre of the robot base to the centre of the bowl, in the robot's coordinate directions. Typical values: `0.25–0.40 0.0 0.00`.

---

## Policy Details

| Property | Value |
|---|---|
| Actor obs | 536-dim: `joint_pos(6) + joint_vel(6) + gripper_link_pos(3) + bowl_pos(3) + ResNet18(512) + last_action(6)` |
| Action | 6-dim: 5 arm joints (JointPosition, scale=0.5) + 1 binary gripper |
| Control rate | 50 Hz (sim.dt=0.01s, decimation=2) |
| Camera | 72×128 px RGB → frozen ResNet18 → 512-dim feature |
| Default joint pos | `[0.0, -0.6, -0.6, 1.57, -1.57, 0.0]` rad |
