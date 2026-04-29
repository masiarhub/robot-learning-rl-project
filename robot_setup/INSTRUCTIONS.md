# LeRobot — Data Collection & Deployment Guide (Windows)

All commands assume the `lerobot` conda environment is active. Run this in every new PowerShell session:

```powershell
& "C:\Users\pcwag\miniforge3\Scripts\conda.exe" shell.powershell hook | Out-String | Invoke-Expression
conda activate lerobot
```

If you only need to run one command and do not want to activate the environment, use `conda run`:

```powershell
& "C:\Users\pcwag\miniforge3\Scripts\conda.exe" run -n lerobot lerobot-teleoperate --help
```

---

## 0. Robot IDs used in this project

| Device   | `--*.id` value  | COM port (check Device Manager if unsure) |
|----------|-----------------|-------------------------------------------|
| Follower | `follower_arm`  | `COM5` (CH343, verify with `lerobot-find-port`) |
| Leader   | `leader_arm`    | `COM7` (CH343, verify with `lerobot-find-port`) |

> **Finding ports:** Plug/unplug the arm and run `lerobot-find-port` — it walks you through identification interactively.
> To list currently active COM ports in PowerShell:
>
> ```powershell
> Get-PnpDevice -Class Ports | Where-Object Status -eq OK | Select-Object FriendlyName, Status
> ```
>
> You can also use:
>
> ```powershell
> [System.IO.Ports.SerialPort]::GetPortNames()
> ```

---

## 0b. Find the correct camera index

Run this script to see which indices have a camera attached. **Plug in the robot camera first**, then run:

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

To identify **which index is the robot camera** (vs laptop webcam): run the script once with the camera unplugged, note the indices, then plug it in and run again — the new index that appears is the robot camera.

You can also preview a specific index live:

```powershell
python -c "
import cv2
cap = cv2.VideoCapture(1)  # change 1 to the index you want to test
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
"
```

Press `q` to close the preview window.

**Current setup** (update this as hardware changes):

| Camera       | Index | Resolution |
|--------------|-------|------------|
| Robot front  | `1`   | 640×480    |

---

## 1. Load calibration files

Calibration files live in `Docs/Calibration/`. Copy them once to the lerobot cache from the repo root — the IDs in the filenames must match `--*.id` in every command.

```powershell
$calib = "$env:USERPROFILE\.cache\huggingface\lerobot\calibration"

New-Item -ItemType Directory -Force "$calib\robots\so_follower"
New-Item -ItemType Directory -Force "$calib\teleoperators\so_leader"

Copy-Item ".\Docs\Calibration\follower_arm.json" `
    "$calib\robots\so_follower\follower_arm.json"

Copy-Item ".\Docs\Calibration\leader_arm.json" `
    "$calib\teleoperators\so_leader\leader_arm.json"
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
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
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
Copy-Item "$calib\robots\so_follower\follower_arm.json" ".\Docs\Calibration\follower_arm.json" -Force
Copy-Item "$calib\teleoperators\so_leader\leader_arm.json" ".\Docs\Calibration\leader_arm.json" -Force
```

---

## 4. Record a dataset

### 4.1 Login to Hugging Face (once)

```powershell
hf auth login
# paste a write-access token from https://huggingface.co/settings/tokens
```

Set `HF_USER` so subsequent commands pick it up automatically (run once per session):

```bash
HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

### 4.2 Record episodes

Adjust the task description and episode count as needed. The dataset uploads to HuggingFace automatically when the session ends (Esc).

```powershell
lerobot-record `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --teleop.type=so101_leader `
    --teleop.port=COM7 `
    --teleop.id=leader_arm `
    --display_data=true `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=50 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --dataset.push_to_hub=true
```

**Prerequisite:** keyboard shortcuts require `pynput` — install it once if missing:
```powershell
pip install pynput
```

**Recording workflow — one episode at a time:**

Each episode has two phases:

1. **Recording phase** — teleoperate the robot to perform the task.
   - Press `→` when the episode is done (saves it, enters reset phase).
   - Press `←` to throw away the episode and immediately re-record it (episode number stays the same).

2. **Reset phase** — move everything back to the start position.
   - Press `→` again to skip the countdown and jump straight to the next episode.
   - Press `←` here too to re-record the episode you just finished instead of moving on.

Pressing `→` twice in quick succession (once to end recording, once to skip reset) is the normal fast flow. Use `←` any time an episode went wrong.

**Keyboard controls summary:**

| Key | Phase | Action |
|-----|-------|--------|
| `→` | Recording | End episode, enter reset phase |
| `→` | Reset | Skip reset, save episode and start next |
| `←` | Recording | Discard episode, re-record same number |
| `←` | Reset | Discard episode just recorded, re-record it |
| `Esc` | Either | Stop session, encode videos, upload dataset |

### 4.3 Record without uploading to Hub

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

Dataset is saved locally at (with a timestamp suffix added by lerobot):

```
%USERPROFILE%\.cache\huggingface\lerobot\pcwagner\so101_pickplace_YYYYMMDD_HHMMSS\
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
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.num_episodes=10 `
    --dataset.single_task="Pick up the object and place it in the bin" `
    --resume=true
```

### 4.5 Upload a local dataset manually

lerobot saves datasets with a timestamp suffix locally (e.g., `so101_pickplace_20260429_101958`). Find the most recent folder and upload it:

```powershell
# Find the most recent dataset folder
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\lerobot\pcwagner" |
    Where-Object { $_.Name -like "so101_pickplace*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 FullName
```

Then upload using that path (the HF repo ID has no timestamp):

```powershell
hf upload RobotLearningProject/so101_pickplace `
    "$env:USERPROFILE\.cache\huggingface\lerobot\pcwagner\so101_pickplace_XXXXXXXXXXXXXXXX" `
    --repo-type dataset
```

After uploading, **tag the dataset with its codebase version** — lerobot-train requires this tag to load the dataset:

```python
from huggingface_hub import HfApi
hub_api = HfApi()
hub_api.create_tag("RobotLearningProject/so101_pickplace", tag="v3.0", repo_type="dataset")
```

The version (`v3.0`) comes from `codebase_version` in the dataset's `meta/info.json`. When using `--dataset.push_to_hub=true`, lerobot tags the dataset automatically — this manual step is only needed after a manual `hf upload`.

---

## 5. Visualize and inspect a dataset

```powershell
lerobot-dataset-viz --dataset.repo_id=RobotLearningProject/so101_pickplace
```

Replay a recorded episode on the real robot:

```powershell
lerobot-replay `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --dataset.repo_id=RobotLearningProject/so101_pickplace `
    --dataset.episode=0
```

---

## 6. Train a policy (ACT)

Run on a machine with a GPU. If training locally on CPU only, expect very long training times.

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

- `--dataset.video_backend=pyav` is required on Windows — the default `torchcodec` backend requires FFmpeg DLLs that are not bundled with the conda install. `pyav` works out of the box.
- Change `--policy.device=cuda` to `--policy.device=cpu` if no GPU is available (slow).
- Set `--wandb.enable=true` and run `wandb login` first to get training plots.
- Checkpoints are saved to `outputs/train/act_so101_pickplace/checkpoints/`. On Windows there is no `last/` symlink — use the numbered folder (e.g. `000010`, `020000`). Find the latest with: `Get-ChildItem outputs\train\act_so101_pickplace\checkpoints | Sort-Object Name -Descending | Select-Object -First 1`

Resume training from last checkpoint:

```powershell
lerobot-train `
    --config_path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model/train_config.json `
    --resume=true
```

Upload the trained policy to Hub:

```powershell
hf upload RobotLearningProject/act_so101_pickplace `
    outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model
```

---

## 6b. Fine-tune from a HuggingFace checkpoint

Use `--policy.path` instead of `--policy.type` to start training from an existing policy on the Hub rather than from scratch. The policy type, architecture, and hyperparameters are loaded from the checkpoint automatically — you do not need `--policy.type`.

### From a previously trained policy (e.g. continuing on new data)

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

- `--policy.path` accepts a HF Hub model ID (`user/model`) or a local `pretrained_model/` directory path.
- The new run saves its own checkpoints independently — the source checkpoint on the Hub is not modified.
- Use a different `--output_dir` and `--policy.repo_id` from the original run to avoid overwriting it.

### On Linux/server (no `--dataset.video_backend` flag needed)

`torchcodec` (the default) works on Linux. Drop the `--dataset.video_backend=pyav` flag that is required on Windows.

### From a community / third-party checkpoint on the Hub

Same pattern — just point `--policy.path` at any compatible model:

```bash
lerobot-train \
    --policy.path=lerobot/act_so100_lego \
    --dataset.repo_id=RobotLearningProject/so101_pickplace \
    --output_dir=outputs/train/act_so101_from_lego \
    --job_name=act_so101_from_lego \
    --policy.device=cuda \
    --policy.repo_id=RobotLearningProject/act_so101_from_lego
```

> **Note:** the source checkpoint must use the same policy architecture and have compatible input/output feature shapes as your dataset. Mismatches will raise an error at startup before any training begins.

### Overriding hyperparameters from a checkpoint

You can append any `--flag=value` to override individual settings from the loaded config:

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

---

## 7. Evaluate / run inference on the robot

Use `lerobot-rollout` (not `lerobot-record`) to deploy a trained policy. The `--policy.path` can be a local checkpoint or a HF Hub model ID.

### 7.1 Quick autonomous run (no recording)

```powershell
lerobot-rollout `
    --strategy.type=base `
    --policy.path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --task="Pick up the object and place it in the bin" `
    --duration=30
```

`--duration` is in seconds. Adjust as needed.

### 7.2 Run from HF Hub model

```powershell
lerobot-rollout `
    --strategy.type=base `
    --policy.path=RobotLearningProject/act_so101_pickplace `
    --robot.type=so101_follower `
    --robot.port=COM5 `
    --robot.id=follower_arm `
    --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" `
    --task="Pick up the object and place it in the bin" `
    --duration=30
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
ds = LeRobotDataset('RobotLearningProject/so101_pickplace')  # loads from HF Hub
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
