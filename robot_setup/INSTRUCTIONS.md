# LeRobot — Data Collection & Deployment Guide (Windows)

All commands assume the `lerobot` conda environment is active. Run this in every new terminal session:

```powershell
C:\Users\pcwag\miniforge3\Scripts\conda.exe activate lerobot
# or if conda is on PATH:
conda activate lerobot
```

---

## 0. Robot IDs used in this project

| Device   | `--*.id` value  | COM port (check Device Manager if unsure) |
|----------|-----------------|-------------------------------------------|
| Follower | `follower_arm`  | `COM5` (CH343, verify with `lerobot-find-port`) |
| Leader   | `leader_arm`    | `COM7` (CH343, verify with `lerobot-find-port`) |

> **Finding ports:** Plug/unplug the arm and run `lerobot-find-port` — it walks you through identification interactively.
> Or in PowerShell: `Get-PnpDevice -Class Ports | Select-Object FriendlyName, Status`

---

## 1. Load calibration files

Calibration files live in `docs/Calibration/`. Copy them once to the lerobot cache — the IDs in the filenames must match `--*.id` in every command.

```powershell
$calib = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration"

New-Item -ItemType Directory -Force "$calib\robots\so101_follower"
New-Item -ItemType Directory -Force "$calib\teleoperators\so101_leader"

Copy-Item "..\docs\Calibration\follower_arm.json" `
    "$calib\robots\so101_follower\follower_arm.json"

Copy-Item "..\docs\Calibration\leader_arm.json" `
    "$calib\teleoperators\so101_leader\leader_arm.json"
```

Verify the files are there:

```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\lerobot\calibration" -Recurse
```

---

## 2. Teleoperate (leader → follower, no recording)

Use this to verify the setup is working before recording any data.

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm
```

With live visualization (requires rerun):

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true
```

With a camera attached (adjust `index_or_path` to your webcam index, usually 0 or 1):

```powershell
lerobot-teleoperate `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true
```

---

## 3. Re-calibrate (if needed)

If the robot behaves incorrectly or calibration files are lost, re-run calibration for each arm separately.

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

After calibration, back up the generated files:

```powershell
$calib = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration"
Copy-Item "$calib\robots\so101_follower\follower_arm.json" "..\docs\Calibration\follower_arm.json" -Force
Copy-Item "$calib\teleoperators\so101_leader\leader_arm.json" "..\docs\Calibration\leader_arm.json" -Force
```

---

## 4. Record a dataset

### 4.1 Login to Hugging Face (once)

```powershell
hf auth login
# paste a write-access token from https://huggingface.co/settings/tokens
```

Get your username:

```powershell
hf auth whoami
# note the username, e.g. "pcwag" — used as HF_USER below
```

### 4.2 Record episodes

Replace `pcwag` with your HF username and adjust the task description and episode count.

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true `
    --dataset.repo_id=pcwag/so101_pickplace `
    --dataset.num_episodes=50 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.streaming_encoding=true `
    --dataset.encoder_threads=2
```

**Keyboard controls during recording:**

| Key | Action |
|-----|--------|
| `→` (right arrow) | End current episode early / skip reset period |
| `←` (left arrow) | Cancel current episode and re-record it |
| `Esc` | Stop session, encode videos, upload dataset |

### 4.3 Record without uploading to Hub

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --dataset.repo_id=pcwag/so101_pickplace `
    --dataset.num_episodes=50 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.push_to_hub=false
```

Dataset is saved locally at:

```
%USERPROFILE%\.cache\huggingface\lerobot\pcwag\so101_pickplace\
```

### 4.4 Resume a interrupted recording

`--dataset.num_episodes` is the number of **additional** episodes to record, not the total.

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --dataset.repo_id=pcwag/so101_pickplace `
    --dataset.num_episodes=10 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --resume=true
```

### 4.5 Upload a local dataset manually

```powershell
hf upload pcwag/so101_pickplace `
    "$env:USERPROFILE\.cache\huggingface\lerobot\pcwag\so101_pickplace" `
    --repo-type dataset
```

---

## 5. Visualize and inspect a dataset

```powershell
lerobot-dataset-viz --dataset.repo_id=pcwag/so101_pickplace
```

Replay a recorded episode on the real robot:

```powershell
lerobot-replay `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --dataset.repo_id=pcwag/so101_pickplace `
    --dataset.episode=0
```

---

## 6. Train a policy (ACT)

Run on a machine with a GPU. If training locally on CPU only, expect very long training times.

```powershell
lerobot-train `
    --dataset.repo_id=pcwag/so101_pickplace `
    --policy.type=act `
    --output_dir=outputs/train/act_so101_pickplace `
    --job_name=act_so101_pickplace `
    --policy.device=cuda `
    --wandb.enable=false `
    --policy.repo_id=pcwag/act_so101_pickplace
```

- Change `--policy.device=cuda` to `--policy.device=cpu` if no GPU is available (slow).
- Set `--wandb.enable=true` and run `wandb login` first to get training plots.
- Checkpoints are saved to `outputs/train/act_so101_pickplace/checkpoints/`.

Resume training from last checkpoint:

```powershell
lerobot-train `
    --config_path=outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model/train_config.json `
    --resume=true
```

Upload the trained policy to Hub:

```powershell
hf upload pcwag/act_so101_pickplace `
    outputs/train/act_so101_pickplace/checkpoints/last/pretrained_model
```

---

## 7. Evaluate / run inference on the robot

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
    --dataset.repo_id=pcwag/eval_act_so101_pickplace `
    --dataset.num_episodes=10 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.push_to_hub=false `
    --policy.path=pcwag/act_so101_pickplace
```

To also allow manual teleoperation between episodes, add:

```powershell
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
```

---

## 8. Useful diagnostics

```powershell
# List all detected COM ports
Get-PnpDevice -Class Ports | Select-Object FriendlyName, Status

# Find which port belongs to the robot interactively
lerobot-find-port

# Check installed lerobot version
python -c "import lerobot; print(lerobot.__version__)"

# Inspect a local dataset
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('pcwag/so101_pickplace')
print(ds)
print('Episodes:', ds.num_episodes)
print('Frames:', ds.num_frames)
"

# Check where calibration files are stored
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

---

## 9. Tips for good data collection

- Record at least **50 episodes** before training; 100+ is better for robust policies.
- Keep the camera **fixed** throughout all recordings. Any camera movement invalidates the dataset.
- Reset the object to a **consistent start position** between episodes.
- Perform the task **smoothly and consistently** — erratic motions hurt policy learning.
- Cover **multiple object positions** (at least 3–5 different locations).
- The rule of thumb: if you can complete the task yourself using only the camera image, the policy can learn it.
- Do not add variation (object type, background, lighting) until the policy reliably works on the simple case.
