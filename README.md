# Robot Learning RL Project

Group Project 3: Singulation - Reinforcement Learning

---

## Table of Contents

- [Overview](#overview)
- [Local Setup — Windows](#local-setup--windows)
  - [0. Robot IDs & Camera Setup](#0-robot-ids--camera-setup)
  - [1. Load Calibration Files](#1-load-calibration-files)
  - [2. Teleoperate (Leader → Follower)](#2-teleoperate-leader--follower)
  - [3. Re-calibrate (If Needed)](#3-re-calibrate-if-needed)
  - [4. Record a Dataset](#4-record-a-dataset)
    - [4.1 Login to Hugging Face](#41-login-to-hugging-face-once)
    - [4.2 Record Episodes](#42-record-episodes)
    - [4.3 Record with Scene Position Logging](#43-record-with-scene-position-logging)
    - [4.4 Record Without Uploading to Hub](#44-record-without-uploading-to-hub)
    - [4.5 Resume Interrupted Recording](#45-resume-interrupted-recording)
    - [4.6 Upload Local Dataset Manually](#46-upload-local-dataset-manually)
    - [4.7 Merge Datasets](#47-merge-datasets)
  - [5. Visualize & Inspect Dataset](#5-visualize--inspect-dataset)
  - [6. Train a Policy (ACT)](#6-train-a-policy-act)
  - [6b. Fine-tune from a HuggingFace Checkpoint](#6b-fine-tune-from-a-huggingface-checkpoint)
  - [7. Evaluate / Run Inference on Robot](#7-evaluate--run-inference-on-robot)
  - [7b. Remote Inference: Policy on Server, Robot on Laptop](#7b-remote-inference-policy-on-server-robot-on-laptop)
  - [8. Useful Diagnostics](#8-useful-diagnostics)
  - [9. Tips for Good Data Collection](#9-tips-for-good-data-collection)
- [Quick Server Setup Scripts](#quick-server-setup-scripts)
  - [Run Order on Fresh Instance](#run-order-on-fresh-instance)
  - [1. Bootstrap the Box](#1-bootstrap-the-box)
  - [2. Install LeRobot](#2-install-lerobot)
  - [3. Activate Environment in Any New Shell](#3-activate-environment-in-any-new-shell)
  - [4. Run Training / Inference](#4-run-training--inference)
  - [5. Decommission Before Shutting Down](#5-decommission-before-shutting-down)
  - [.env Reference](#env-reference)
- [Brev GPU Training Guide](#brev-gpu-training-guide)
  - [Instance Details](#instance-details)
  - [1. Connect to the Instance](#1-connect-to-the-instance)
  - [2. First-time Setup on Fresh Instance](#2-first-time-setup-on-fresh-instance)
  - [3. Resume After Disconnect](#3-resume-after-disconnect)
  - [4. Check Training Progress](#4-check-training-progress)
  - [5. Upload Trained Model Manually](#5-upload-trained-model-manually)
  - [6. Stop the Instance](#6-stop-the-instance)
  - [7. Useful tmux Cheatsheet](#7-useful-tmux-cheatsheet)
- [Server Setup & Training](#server-setup--training)
  - [1. Go to the Project](#1-go-to-the-project)
  - [2. Activate the LeRobot Environment](#2-activate-the-lerobot-environment)
  - [3. Video Backend Setup](#3-video-backend-setup)
  - [4. Optional: Use tmux for Persistent Sessions](#4-optional-use-tmux-for-persistent-sessions)
  - [5. Train ACT Policy on Linux](#5-train-act-policy-on-linux)
  - [6. H100 Performance Tuning](#6-h100-performance-tuning)
  - [7. Check GPU Usage](#7-check-gpu-usage)
  - [8. Find the Latest Checkpoint](#8-find-the-latest-checkpoint)
  - [9. Resume Training](#9-resume-training)
  - [10. Upload Trained Policy to Hugging Face](#10-upload-trained-policy-to-hugging-face)
  - [11. Fine-tune from Existing Hugging Face Policy](#11-fine-tune-from-existing-hugging-face-policy)
  - [12. Async Inference (Policy Server)](#12-async-inference-policy-server)

---

## Overview

This project combines Behavior Cloning (BC) and Reinforcement Learning (RL) using the LeRobot framework to train robotic manipulation policies for SO-101 arms.

### Key Technologies
- **Framework:** [HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- **RL Algorithm:** SAC (Soft Actor-Critic) for sample efficiency
- **Policy Architecture:** ResNet-18 visual encoder + MLP actor
- **Input:** RGB image from wrist camera (84×84) + gripper state + goal (x,y,z coordinates)
- **Output:** Robot action (Δx, Δy, Δz, gripper)
- **Sim-to-Real:** Domain randomization across block positions, colors, textures, lighting, camera noise

---

## Local Setup — Windows

All commands assume the `lerobot` conda environment is active. Run this in every new PowerShell session:

```powershell
& "C:\Users\pcwag\miniforge3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression
conda activate lerobot
```

If you only need one command without activating the environment:

```powershell
& "C:\Users\pcwag\miniforge3\Scripts\conda.exe" run -n lerobot lerobot-teleoperate --help
```

### 0. Robot IDs & Camera Setup

Every user's hardware setup is different. Before running any commands, you **must** identify your own COM ports and camera indices. The values shown below are examples only.

**Step 1: Find your COM ports**

Plug in both robot arms (follower and leader) via USB. Then list active COM ports:

```powershell
# List all active COM ports
Get-PnpDevice -Class Ports | Where-Object Status -eq OK | Select-Object FriendlyName, Status

# Or use this alternative:
[System.IO.Ports.SerialPort]::GetPortNames()

# Interactive identification (recommended)
# Plug/unplug each arm and run this to see which port changes
lerobot-find-port
```

Note the COM port assigned to each arm. Common values are `COM3`, `COM4`, `COM5`, `COM6`, etc.

**Step 2: Find your camera index**

Plug in the robot camera (USB), then scan for available camera indices:

```powershell
python -c "
import cv2
for i in range(8):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Index {i}: OK  ({w}x{h})')
        cap.release()
    else:
        print(f'Index {i}: not found')
"
```

To identify which index is the robot camera (vs built-in webcam), run with camera **unplugged**, note the indices, plug it **in**, and run again—the new index is your robot camera.

**Step 3: Verify camera index (optional)**

Live preview of a specific camera index (press `q` to close):

```powershell
python -c "
import cv2
cap = cv2.VideoCapture(1)  # change 1 to YOUR camera index
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
"
```

**Your Setup (customize these values):**

| Device   | `--*.id` value  | COM Port | Notes |
|----------|-----------------|----------|-------|
| Follower | `follower_arm`  | `COM?`   | **Your value** — find with `lerobot-find-port` |
| Leader   | `leader_arm`    | `COM?`   | **Your value** — find with `lerobot-find-port` |

| Camera      | Index | Resolution | Notes |
|-------------|-------|------------|-------|
| Robot front | `?`   | 1280×720   | **Your value** — find with camera index script above |

**Replace all `COM?` and camera `?` with your actual values in the commands below.**

### 1. Load Calibration Files

Copy calibration files to the LeRobot cache. Robot IDs in filenames must match `--*.id` in commands:

```powershell
$calib = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration"

New-Item -ItemType Directory -Force "$calib\robots\so_follower"
New-Item -ItemType Directory -Force "$calib\teleoperators\so_leader"

Copy-Item ".\Docs\Calibration\follower_arm.json" `
    "$calib\robots\so_follower\follower_arm.json"

Copy-Item ".\Docs\Calibration\leader_arm.json" `
    "$calib\teleoperators\so_leader\leader_arm.json"
```

Verify:

```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\lerobot\calibration" -Recurse
```

**Use your actual COM ports and camera index from § 0 in all commands below.**

### 2. Teleoperate (Leader → Follower)

Test the setup without recording. **Replace `COM5`, `COM7`, and camera index `1` with your values from § 0:**

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm
```

With camera:

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true
```

### 3. Re-calibrate (If Needed)

Replace `COM5` and `COM7` with your actual COM ports from § 0:

```powershell
# Follower arm
lerobot-calibrate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm

# Leader arm
lerobot-calibrate `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm
```

Back up the calibration files:

```powershell
$calib = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration"
Copy-Item "$calib\robots\so_follower\follower_arm.json" ".\Docs\Calibration\follower_arm.json" -Force
Copy-Item "$calib\teleoperators\so_leader\leader_arm.json" ".\Docs\Calibration\leader_arm.json" -Force
```

### 4. Record a Dataset

#### 4.1 Login to Hugging Face (Once)

```powershell
hf auth login
# Paste write-access token from https://huggingface.co/settings/tokens
```

Set `HF_USER` for the session:

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

#### 4.2 Record Episodes

Adjust task description and episode count as needed. Dataset uploads automatically on exit (Esc). **Replace `COM5`, `COM7`, and camera index `1` with your values:**

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=30 `
    --dataset.single_task="Singlecubepick" `
    --dataset.push_to_hub=true `
    --dataset.private=true `
    --dataset.log_cube_position=true `
    --dataset.log_bin_position=true
```

Requires `pynput` for keyboard shortcuts:

```powershell
pip install pynput
```

**Recording Workflow (One Episode at a Time):**

1. **Recording Phase:** Teleoperate robot to perform task
   - Press `→` when done (saves, enters reset phase)
   - Press `←` to discard and re-record
2. **Reset Phase:** Move everything back to start
   - Press `→` again to skip countdown and start next episode
   - Press `←` to re-record previous episode

**Keyboard Controls:**

| Key | Phase | Action |
|-----|-------|--------|
| `→` | Recording | End episode, enter reset phase |
| `→` | Reset | Skip reset, save and start next |
| `←` | Recording | Discard, re-record same number |
| `←` | Reset | Discard previous, re-record it |
| `Esc` | Either | Stop, encode videos, upload dataset |

#### 4.3 Record with Scene Position Logging

Prompt for position labels (free-form text like `left`, `center-far`) before each episode. **Replace `COM5`, `COM7`, and camera index `1` with your values:**

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=50 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.push_to_hub=true `
    --dataset.private=true `
    --dataset.log_cube_position=true `
    --dataset.log_bin_position=true
```

Positions are stored as per-frame top-level features in the dataset and appear in the HuggingFace viewer and batch dicts (`batch["cube_position"]`, `batch["bin_position"]`).

#### 4.4 Record Without Uploading to Hub

**Replace `COM5` and `COM7` with your actual COM ports:**

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=50 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.push_to_hub=false
```

Dataset saved locally with timestamp:

```
%USERPROFILE%\.cache\huggingface\lerobot\pcwagner\so101_pickplace_YYYYMMDD_HHMMSS\
```

#### 4.5 Resume Interrupted Recording

`--dataset.num_episodes` is the number of **additional** episodes to record. **Replace `COM5` and `COM7` with your actual COM ports:**

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=10 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --resume=true
```

#### 4.6 Upload Local Dataset Manually

```powershell
# Find most recent dataset folder
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\lerobot\pcwagner" |
    Where-Object { $_.Name -like "so101_pickplace*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 FullName
```

Upload using that path:

```powershell
hf upload RobotLearningProject/so101_pickplace `
    "$env:USERPROFILE\.cache\huggingface\lerobot\pcwagner\so101_pickplace_XXXXXXXXXXXXXXXX" `
    --repo-type dataset
```

After uploading, tag the dataset with its codebase version (required by lerobot-train):

```python
from huggingface_hub import HfApi
hub_api = HfApi()
hub_api.create_tag("RobotLearningProject/so101_pickplace", tag="v3.0", repo_type="dataset")
```

#### 4.7 Merge Datasets

Merge Hub datasets and push:

```powershell
lerobot-edit-dataset `
    --new_repo_id=RobotLearningProject/so101_pickplace_merged `
    --operation.type=merge `
    --operation.repo_ids "['RobotLearningProject/so101_pickplace', 'RobotLearningProject/rollout_so101_dagger_remote']" `
    --push_to_hub=true
```

Merge local folders without downloading:

```powershell
lerobot-edit-dataset `
    --new_repo_id=RobotLearningProject/so101_pickplace_merged `
    --new_root="$env:USERPROFILE\.cache\huggingface\lerobot\RobotLearningProject\so101_pickplace_merged" `
    --operation.type=merge `
    --operation.repo_ids "['so101_pickplace_a', 'so101_pickplace_b']" `
    --operation.roots "['C:/Users/pcwag/.cache/huggingface/lerobot/pcwagner/so101_pickplace_a', 'C:/Users/pcwag/.cache/huggingface/lerobot/pcwagner/so101_pickplace_b']" `
    --push_to_hub=true
```

Inspect merged dataset:

```powershell
lerobot-edit-dataset `
    --repo_id=RobotLearningProject/so101_pickplace_merged `
    --operation.type=info `
    --operation.show_features=true

lerobot-dataset-viz --dataset.repo_id=RobotLearningProject/so101_pickplace_merged
```

Recompute stats after merging (important before training):

```powershell
lerobot-edit-dataset `
    --repo_id=RobotLearningProject/so101_pickplace_merged `
    --new_repo_id=RobotLearningProject/so101_pickplace_merged_stats `
    --operation.type=recompute_stats `
    --operation.num_workers=4 `
    --push_to_hub=true
```

Train on stats-refreshed repo:

```powershell
lerobot-train `
    --dataset.repo_id=RobotLearningProject/so101_pickplace_merged_stats `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_pickplace_merged `
    --job_name=act_so101_pickplace_merged `
    --policy.device=cuda `
    --wandb.enable=false `
    --policy.repo_id=RobotLearningProject/act_so101_pickplace_merged `
    --dataset.video_backend=pyav
```

### 5. Visualize & Inspect Dataset

```powershell
lerobot-dataset-viz --dataset.repo_id=RobotLearningProject/so101_pickplace
```

Replay a recorded episode on the real robot. **Replace `COM5` with your actual COM port:**

```powershell
lerobot-replay `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.episode=0
```

### 6. Train a Policy (ACT)

Run on a machine with GPU. CPU-only training is very slow.

```powershell
lerobot-train `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_pickplace `
    --job_name=act_so101_pickplace `
    --policy.device=cuda `
    --wandb.enable=false `
    --policy.repo_id=RobotLearningProject/act_so101_pickplace `
    --dataset.video_backend=pyav
```

**Notes:**
- `--dataset.video_backend=pyav` is required on Windows (torchcodec FFmpeg DLLs not bundled)
- Change `--policy.device=cuda` to `--policy.device=cpu` if no GPU
- Set `--wandb.enable=true` and run `wandb login` for training plots
- Checkpoints save to `outputs/train/act_so101_pickplace/checkpoints/`
- On Windows, no `last/` symlink — use numbered folder (e.g., `000010`, `020000`)
- Find latest: `Get-ChildItem outputs\train\act_so101_pickplace\checkpoints | Sort-Object Name -Descending | Select-Object -First 1`

**Resume training from checkpoint:**

```powershell
lerobot-train `
    --config_path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model/train_config.json `
    --resume=true
```

**Upload trained policy to Hub:**

```powershell
hf upload RobotLearningProject/act_so101_pickplace `
    outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model
```

### 6b. Fine-tune from a HuggingFace Checkpoint

Use `--policy.path` instead of `--policy.type` to start from an existing model. Policy type and hyperparameters load automatically:

#### From a Previously Trained Policy

```bash
lerobot-train \
    --policy.path=RobotLearningProject/act_so101_pickplace \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_pickplace_ft \
    --job_name=act_so101_pickplace_ft \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace_ft
```

- `--policy.path` accepts HF Hub ID (`user/model`) or local `pretrained_model/` path
- New run saves checkpoints independently
- Use different `--output_dir` and `--policy.repo_id` to avoid overwriting original

#### On Linux/Server

Drop the `--dataset.video_backend=pyav` flag (torchcodec works on Linux):

```bash
lerobot-train \
    --policy.path=RobotLearningProject/act_so101_pickplace \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_pickplace_ft \
    --job_name=act_so101_pickplace_ft \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace_ft
```

#### From Community Checkpoint

```bash
lerobot-train \
    --policy.path=lerobot/act_so100_lego \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_from_lego \
    --job_name=act_so101_from_lego \
    --policy.device=cuda \
    --policy.repo_id=RobotLearningProject/act_so101_from_lego
```

**Note:** Source checkpoint must use same architecture and compatible input/output shapes.

#### Override Hyperparameters

```bash
lerobot-train \
    --policy.path=RobotLearningProject/act_so101_pickplace \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_pickplace_ft \
    --policy.device=cuda \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace_ft \
    --steps=50000 \
    --batch_size=16
```

### 7. Evaluate / Run Inference on Robot

Use `lerobot-rollout` (not `lerobot-record`) to deploy a trained policy. `--policy.path` can be local or HF Hub. **In all commands below, replace `COM5` and camera index `1` with your values from § 0.**

#### 7.1 Quick Autonomous Run (No Recording)

```powershell
lerobot-rollout `
    --strategy.type=base `
    --policy.path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --task="Pick up the object and place it in the bin" `
    --duration=30
```

#### 7.2 Run from HF Hub Model

```powershell
lerobot-rollout `
    --strategy.type=base `
    --policy.path=RobotLearningProject/act_so101_pickplace `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --task="Pick up the object and place it in the bin" `
    --duration=30
```

### 7b. Remote Inference: Policy on Server, Robot on Laptop

When laptop CPU is too slow for 30 fps inference, run policy on Brev GPU and keep robot on laptop over USB. Only observations (images + joint state) and action chunks cross the network; USB stays local.

#### 7b.1 One-time Install

Add async extras to laptop env:

```powershell
cd .\robot_setup\lerobot_src
pip install -e '.[async]'
cd ..\..
```

Sanity check:

```powershell
python -c "import grpc, lerobot.async_inference.robot_client; print('ok')"
```

#### 7b.2 Start the Policy Server

Follow § Server Setup & Training → Async Inference. Leave it running in tmux on Brev.

#### 7b.3 Open SSH Tunnel from Laptop

In a separate PowerShell, kept open during inference:

```powershell
ssh -N -L 8080:localhost:8080 -i "C:\Users\pcwag\.brev\brev.pem" shadeform@38.128.233.202
```

(`<user>@<brev-host>` from Brev dashboard SSH access). `-N` means no shell — the window just forwards. `localhost:8080` on laptop tunnels to policy server.

#### 7b.4 Run Remote Inference (Recommended)

Use for DAgger, Sentry, Highlight, or base rollout. Laptop connects to robot/teleoperator locally; `--inference.type=remote` sends observations to server and reads back action chunks asynchronously. **Replace `COM5`, `COM7`, and camera index `1` with your values:**

**Base rollout, no recording:**

```powershell
lerobot-rollout `
    --strategy.type=base `
    --inference.type=remote `
    --inference.server_address=localhost:8080 `
    --inference.policy_device=cuda `
    --inference.actions_per_chunk=50 `
    --inference.chunk_size_threshold=0.5 `
    --inference.aggregate_fn_name=weighted_average `
    --policy.path=RobotLearningProject/act_so101_merged `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --task="Pick up the object and place it in the bin" `
    --display_data=true `
    --duration=30
```

**DAgger with remote policy inference:**

```powershell
lerobot-rollout `
     --strategy.type=dagger `
     --strategy.num_episodes=10 `
     --strategy.input_device=keyboard `
     --strategy.pre_correction_s=2.0 `
     --strategy.post_correction_s=2.0 `
     --inference.type=remote `
     --inference.server_address=localhost:8080 `
     --inference.policy_device=cuda `
     --inference.actions_per_chunk=50 `
     --inference.chunk_size_threshold=0.5 `
     --inference.aggregate_fn_name=weighted_average `
     --policy.path=RobotLearningProject/act_so101_merged `
     --robot.type=so101_follower `
     --robot.port=COM5 `
     --robot.id=follower_arm `
     --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
     --teleop.type=so101_leader `
     --teleop.port=COM7 `
     --teleop.id=leader_arm `
     --dataset.repo_id=RobotLearningProject/rollout_so101_dagger_remote_newestVersoin `
     --dataset.single_task="Pick up the object and place it in the bin" `
     --dataset.num_episodes=10 `
     --task="Pick up the object and place it in the bin" `
     --fps=10 `
     --strategy.follower_mirror=true `
     --display_data=true
```

With `--strategy.follower_mirror=true`, leader follows follower during autonomous and paused phases. Press correction key to release leader torque and start recording from aligned pose.

**DAgger reset/setup flow:**

- `AUTONOMOUS`: Policy is playing (not recorded in corrections-only mode)
- Press `Space` to pause. Robot holds pose. Reset cube/bin, clear scene
- Press `m` while paused to toggle setup motion (move follower without recording)
- Press `Tab` to enter `CORRECTING / RECORDING`. Drive with leader — frames are saved
- Press `Backspace` during correction to discard and return to paused
- Press `Tab` again to stop correction (becomes pending)
- If bad correction, press `Backspace` while paused to delete pending correction
- If good, continue. Pressing `Space`, starting new correction, pushing, or `Esc`/`Ctrl+C` saves pending correction
- When scene is ready for next eval, press `Space` again. Policy resets and autonomy resumes

Reset logic is manual-but-explicit: after each correction, stay paused, reset objects/arm, then resume. Use `--strategy.follower_mirror=true` so leader stays aligned while paused.

#### 7b.5 Legacy Standalone Robot Client

**Replace `COM5` and camera index `1` with your values:**

```powershell
python -m lerobot.async_inference.robot_client `
    --server_address=localhost:8080 `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 1280, height: 720, fps: 30, fourcc: MJPG, backend: 700}}" `
    --task="Pick up the object and place it in the bin" `
    --policy_type=act `
    --pretrained_name_or_path=RobotLearningProject/act_so101_merged `
    --policy_device=cuda `
    --actions_per_chunk=50 `
    --chunk_size_threshold=0.5 `
    --aggregate_fn_name=weighted_average `
    --display_data=true `
    --debug_visualize_queue_size=true
```

**Notes:**
- `--pretrained_name_or_path` resolved on server (HF Hub ID or server filesystem path)
- **Private repos:** Server must be logged in with HF token with read access. Run `hf auth whoami` on Brev first
- If `config.json not found`: check repo visibility with `curl -s -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/models/<repo>` (401=private, 200=public)
- Camera key and resolution must match training
- Calibration files must exist locally at `%USERPROFILE%\.cache\huggingface\lerobot\calibration\robots\so_follower\follower_arm.json`
- `--policy_device=cuda` runs on server GPU; use `cpu` for testing only
- First request triggers checkpoint download (~few seconds for ACT); subsequent runs reuse cache

#### 7b.6 Notes & Tuning

- For `lerobot-rollout`, `--policy.path` still required on laptop (loads lightweight config, passes model ID to server). Heavy weights load on **server**
- For legacy client, `--pretrained_name_or_path` resolved on **server**

Tune based on robot behavior and logs. Legacy client with `--debug_visualize_queue_size=true` shows queue plot:

- **Queue drains to 0 (robot stutters):** Round-trip too slow. Try `--actions_per_chunk=80` (ACT outputs up to 100), or lower `--chunk_size_threshold=0.3` for earlier refills
- **Queue stays full:** You have headroom. Raise `--chunk_size_threshold=0.7` for more reactive updates (more bandwidth, fresher observations)
- **Bandwidth concerns:** Observations are images — reduce camera resolution (640×480) or fps to cut uplink size. Policy must be trained at same resolution

If connection breaks or client hangs, kill both ends (Ctrl+C client, Ctrl+C SSH tunnel) and restart in order: server (tmux already running) → tunnel → client.

### 8. Useful Diagnostics

```powershell
# List all detected COM ports
Get-PnpDevice -Class Ports | Select-Object FriendlyName, Status

# Find which port belongs to robot (interactive)
lerobot-find-port

# Check installed LeRobot version
python -c "import lerobot; print(lerobot.__version__)"

# Inspect a local dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('RobotLearningProject/so101_pickplace')
print(ds)
print('Episodes:', ds.num_episodes)
print('Frames:', ds.num_frames)
"

# Check calibration file location
python -c "
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
print(HF_LEROBOT_CALIBRATION)
"

# List cameras detected by OpenCV
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera index {i}: OK')
        cap.release()
    else:
        print(f'Camera index {i}: not found')
"
```

### 9. Tips for Good Data Collection

- Record at least **50 episodes** before training; 100+ is better for robust policies
- Keep camera **fixed** throughout all recordings. Any camera movement invalidates dataset
- Reset object to **consistent start position** between episodes
- Perform task **smoothly and consistently** — erratic motions hurt learning
- Cover **multiple object positions** (at least 3–5 different locations)
- Rule of thumb: if you can complete the task using only the camera image, the policy can learn it
- Do not add variation (object type, background, lighting) until policy reliably works on simple case

---

## Brev GPU Training Guide

Training runs on a rented NVIDIA A6000 instance via Brev/Shadeform.

### Instance Details

| Field | Value |
|-------|-------|
| Name | `evil-peach-crayfish` |
| IP | `38.128.233.14` |
| GPU | NVIDIA RTX A6000 (48 GiB) |
| Cost | ~$0.60/hr — **stop when done** |

### 1. Connect to the Instance

Open WSL (Ubuntu) on Windows:

```bash
wsl
```

Connect:

```bash
brev shell evil-peach-crayfish
```

If `brev` is not installed in WSL:

```bash
# Install Brev CLI from your Brev dashboard
brev login
brev shell evil-peach-crayfish
```

### 2. First-time Setup on Fresh Instance

Clone the repo using a GitHub personal access token (password auth disabled):

```bash
git clone https://YOUR_GITHUB_TOKEN@github.com/masiarhub/robot-learning-rl-project
cd robot-learning-rl-project
```

Get a token at https://github.com/settings/tokens (classic, `repo` scope).

Run full setup:

```bash
tmux new -s train
bash robot_setup/brev_setup.sh
```

The script installs conda, LeRobot, logs into HuggingFace, trains, and uploads the model automatically. Detach with `Ctrl+B` then `D`. Training continues after closing terminal.

### 3. Resume After Disconnect

Reconnect:

```bash
wsl
brev shell evil-peach-crayfish
```

Reattach tmux:

```bash
tmux attach -t train
```

List sessions:

```bash
tmux ls
```

### 4. Check Training Progress

Inside tmux you'll see live loss output. Check GPU usage in a second tmux window:

```bash
# Ctrl+B then C to open new window
watch -n 2 nvidia-smi
```

Switch windows with `Ctrl+B` then window number (`0`, `1`, etc.).

### 5. Upload Trained Model Manually

If the script didn't upload automatically (e.g., killed early):

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
echo "Uploading: $CKPT"
hf upload pcwagner/act_so101_pickplace "$CKPT"
```

### 6. Stop the Instance

**Always stop when done** — costs $0.60/hr.

Go to https://brev.dev → your instance → **Stop**.

Or from terminal before disconnecting:

```bash
sudo poweroff
```

### 7. Useful tmux Cheatsheet

| Action | Keys |
|--------|------|
| Detach (keep running) | `Ctrl+B` then `D` |
| New window | `Ctrl+B` then `C` |
| Switch window | `Ctrl+B` then `0`–`9` |
| Split pane horizontally | `Ctrl+B` then `"` |
| Kill current pane | `Ctrl+B` then `X` |
| Reattach session | `tmux attach -t train` |
| List sessions | `tmux ls` |

---

## Server Setup & Training

**Note:** This section documents manual Linux/bash commands for the training server. Most of the setup steps (§ 1-3 below) are already automated by the [Quick Server Setup Scripts](#quick-server-setup-scripts) section. Use this section if you:
- Prefer manual control and understanding each step
- Need reference documentation of what the scripts do
- Are customizing the setup for specific needs
- Want to run advanced training/inference workflows (§ 4-12)

### 1. Go to the Project

```bash
cd ~/robot-learning-rl-project
```

If the repo is somewhere else, `cd` into that path instead.

### 2. Activate the LeRobot Environment

Run in every new server shell before using `lerobot-train`, `hf`, or other commands:

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

Quick check:

```bash
which python
python -c "import lerobot; print(lerobot.__version__)"
```

If `conda activate lerobot` fails on a fresh server:

```bash
bash QuicksetupScripts/lerobotSetup.sh
```

### 3. Video Backend Setup

The setup script patches LeRobot to use PyAV by default (avoids TorchCodec/FFmpeg errors on fresh servers).

If packages are missing:

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
conda install -c conda-forge "ffmpeg>=6,<8" -y
pip install 'av>=15.0.0,<16.0.0'
```

Or force PyAV in training command:

```bash
--dataset.video_backend=pyav
```

### 4. Optional: Use tmux for Persistent Sessions

Start a persistent training session that survives SSH disconnects:

```bash
tmux new -s train
```

Detach without stopping:

```
Ctrl+B, then D
```

Reattach later:

```bash
tmux attach -t train
```

List sessions:

```bash
tmux ls
```

### 5. Train ACT Policy on Linux

On Linux, use backslashes for line continuation (not PowerShell backticks):

```bash
lerobot-train \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_pickplace \
    --job_name=act_so101_pickplace \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace
```

**Notes:**
- Do not use PowerShell backticks on Linux
- `--policy.device=cuda` requires NVIDIA GPU
- Use `--dataset.video_backend=pyav` if the command or saved config tries to use TorchCodec
- Run `wandb login` and change `--wandb.enable=false` to `--wandb.enable=true` for W&B integration

### 6. H100 Performance Tuning

ACT is small, so H100 won't be saturated by default `batch_size=8`. Video datasets spend CPU time decoding before GPU trains.

For faster wall-clock training on H100, start with:

```bash
--batch_size=64 \
--num_workers=12 \
--prefetch_factor=4 \
--dataset.video_backend=pyav
```

If VRAM and training are stable, try:

```bash
--batch_size=128 \
--num_workers=16 \
--prefetch_factor=4 \
--dataset.video_backend=pyav
```

Larger batches improve steps/sec and samples/sec but can change learning behavior. More workers help until CPU/video decoding or storage saturates.

### 7. Check GPU Usage

```bash
watch -n 2 nvidia-smi
```

### 8. Find the Latest Checkpoint

```bash
ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1
```

Save to variable:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
echo "$CKPT"
```

### 9. Resume Training

Replace `NNNNNN` with checkpoint folder name (e.g., `020000`):

```bash
lerobot-train \
    --config_path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model/train_config.json \
    --resume=true
```

Or resume from latest automatically:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
lerobot-train \
    --config_path="$CKPT/train_config.json" \
    --resume=true
```

### 10. Upload Trained Policy to Hugging Face

Login first if needed:

```bash
hf auth login
```

Upload latest checkpoint:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
hf upload RobotLearningProject/act_so101_pickplace "$CKPT"
```

### 11. Fine-tune from Existing Hugging Face Policy

Use `--policy.path` instead of `--policy.type`:

```bash
lerobot-train \
    --policy.path=RobotLearningProject/act_so101_pickplace \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_pickplace_ft \
    --job_name=act_so101_pickplace_ft \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace_ft
```

### 12. Async Inference (Policy Server)

Use when laptop CPU inference is too slow. Server runs policy, streams action chunks to `lerobot-rollout --inference.type=remote` or legacy `RobotClient` on laptop, which controls SO-101 over USB locally. Only observations and action chunks cross network; USB stays local. See § Local Setup → 7b for laptop side.

#### 12.1 One-time Install

`QuicksetupScripts/lerobotSetup.sh` installs `async` extra automatically. If your env predates that change:

```bash
cd ~/robot-learning-rl-project/robot_setup/lerobot_src
sudo pip install -e '.[async]' -q
```

Sanity check:

```bash
python -c "import grpc, lerobot.async_inference.policy_server; print('ok')"
```

#### 12.2 Start the Policy Server

```bash
tmux new -s policy-server
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

Server starts empty. Policy selected during first handshake from laptop (`--policy.path` for remote rollout, `--pretrained_name_or_path` for legacy client) — do **not** pass `--policy.path` here. Detach with `Ctrl+B` then `D`. Reattach: `tmux attach -t policy-server`.

#### 12.3 Expose Port 8080 to Laptop

Two options. **SSH tunnel is preferred** — simpler, encrypted, no public exposure.

**Option A — SSH local-forward (recommended):** Nothing to do on server. Laptop forwards `localhost:8080` over existing SSH connection (see § Local Setup → 7b.3).

**Option B — Public Brev port:** Brev dashboard → your instance → Networking → expose port `8080`. Use public URL Brev returns as `server_address` on laptop. gRPC service has **no auth** — only do on trusted network or temporarily.

#### 12.4 One-shot Activation + Training

Use this to paste one block into fresh server shell:

```bash
cd ~/robot-learning-rl-project
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot

lerobot-train \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_pickplace \
    --job_name=act_so101_pickplace \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=RobotLearningProject/act_so101_pickplace
```

---

## Quick Server Setup Scripts

Automate server setup and training on rented GPU instances (Brev, Shadeform).

### Run Order on Fresh Instance

Run these commands top to bottom on a **fresh Brev / Shadeform GPU instance**. Every script is idempotent — safe to re-run.

**Prerequisite:** Open the instance in VS Code via Remote-SSH or Remote-Tunnels first; the rest runs inside that remote shell.

### 1. Bootstrap the Box

Clone repo, install VS Code extensions, monitoring tools:

```bash
mkdir -p ~/QuicksetupScripts
# Drag-drop the four script files into ~/QuicksetupScripts in VS Code
# remote-explorer panel (or scp from laptop). Restore +x — drag-drop loses executable bit:
chmod +x ~/QuicksetupScripts/*.sh
cd ~
bash QuicksetupScripts/brevServerSetup.sh
```

**What it does:**
- Re-launches itself in a `tmux` session called `server-setup` (so SSH drops don't kill it)
- Clones `masiarhub/robot-learning-rl-project` into `~/robot-learning-rl-project` with submodules
- Checks out the `lerobot-setup` branch
- Installs VS Code extensions (Claude, ChatGPT, Git Graph) into remote VS Code Server
- Installs `btop` and `nvtop`

If the GitHub token in `QuicksetupScripts/.env` is missing or invalid, it prompts for one and saves it.

**Detach tmux:** `Ctrl+B` then `D`. **Re-attach:** `tmux attach -t server-setup`.

### 2. Install LeRobot

Set up conda env, dependencies, HF login:

`git clone` preserves +x for executable files, so `chmod` is normally not needed. Included as safety net for mounts that strip permissions (some WSL `/mnt/c` setups, etc.):

```bash
chmod +x ~/robot-learning-rl-project/QuicksetupScripts/*.sh
bash ~/robot-learning-rl-project/QuicksetupScripts/lerobotSetup.sh
```

**What it does:**
- Installs Miniconda if missing (`~/miniconda3`)
- Creates `lerobot` conda env (Python 3.12)
- Installs `ffmpeg>=6,<8` from conda-forge (TorchCodec needs FFmpeg 4–7)
- Applies local LeRobot patches:
  - `datasets/factory.py` — initializes `dataset.meta.stats[key]` before assigning ImageNet stats
  - `datasets/factory.py` — overrides `video_backend='torchcodec'` to `'pyav'`
  - `utils/import_utils.py` — `get_safe_default_codec()` always returns `'pyav'`
- Installs editable `lerobot[dataset,training]` + `av>=15` + `pynput`
- Logs into Hugging Face using `HF_TOKEN` from `.env`

#### 2a. Install Async-Inference Extras (Only if Running Remote Inference)

```bash
cd ~/robot-learning-rl-project/robot_setup/lerobot_src
pip install -e '.[async]' -q
cd ~/robot-learning-rl-project
```

Adds `grpcio` and `matplotlib`. Required to run policy server in § Server Setup & Training § 12.

### 3. Activate Environment in Any New Shell

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

Useful one-liner copy block for fresh tmux panes.

### 4. Run Training / Inference

See:
- § Server Setup & Training § 5 — train ACT
- § Server Setup & Training § 12 — run async policy server

### 5. Decommission Before Shutting Down

```bash
bash ~/robot-learning-rl-project/QuicksetupScripts/brevServerDecomission.sh
```

**What it does:**
- Creates branch `decommissioning-<hostname>-<YYYY-MM-DD>` and pushes to GitHub (excludes `outputs/`)
- Uploads every `outputs/train/*/checkpoints/*/pretrained_model/` to Hugging Face under `${HF_REPO_PREFIX}/<run_name>`
- Prints GitHub branch URL and HF repo URLs at the end

If `HF_REPO_PREFIX` isn't in `.env`, it prompts for one (e.g., `pcwagner` or `RobotLearningProject`).

### .env Reference

`QuicksetupScripts/.env` (auto-created from `.env.example` on first run, `chmod 600`). Don't commit.

```env
GITHUB_TOKEN=ghp_...        # repo scope
HF_TOKEN=hf_...             # write access
HF_REPO_PREFIX=pcwagner     # username or org for uploaded checkpoints
```

The setup scripts validate each token against the live API before continuing and re-prompt if invalid.
