# Server Instructions

Linux/bash commands for the training server. Use these instead of the Windows PowerShell commands in `robot_setup/INSTRUCTIONS.md`.

## 1. Go to the project

```bash
cd ~/robot-learning-rl-project
```

If the repo is somewhere else, `cd` into that repo path instead.

## 2. Activate the LeRobot environment

Run this in every new server shell before using `lerobot-train`, `hf`, or other project commands:

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

If `conda activate lerobot` fails on a fresh server, run setup first:

```bash
bash QuicksetupScripts/lerobotSetup.sh
```

## 3. Install FFmpeg for TorchCodec video decoding

`torchcodec` needs FFmpeg shared libraries such as `libavutil.so`. The setup script installs these automatically, but if you see an error like `Could not load libtorchcodec_core*.so` or `libavutil.so.*: cannot open shared object file`, run:

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
conda install -c conda-forge ffmpeg -y
```

If you need a quick fallback, force the PyAV video backend in your training command:

```bash
--dataset.video_backend=pyav
```

## 4. Optional: use tmux so training survives disconnects

Start a persistent training session:

```bash
tmux new -s train
```

Detach without stopping training:

```text
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

## 5. Train ACT policy on Linux

On Linux, use backslashes for line continuation:

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

Notes:

- Do not use PowerShell backticks on Linux.
- `--policy.device=cuda` requires an NVIDIA GPU.
- Linux can use the default TorchCodec backend once FFmpeg is installed. Use `--dataset.video_backend=pyav` only as a fallback if TorchCodec cannot load FFmpeg libraries.
- To enable Weights & Biases, run `wandb login` and change `--wandb.enable=false` to `--wandb.enable=true`.

## 6. Check GPU usage

```bash
watch -n 2 nvidia-smi
```

## 7. Find the latest checkpoint

```bash
ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1
```

Save it to a variable:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
echo "$CKPT"
```

## 8. Resume training

Replace `NNNNNN` with the checkpoint folder name, for example `020000`:

```bash
lerobot-train \
    --config_path=outputs/train/act_so101_pickplace/checkpoints/NNNNNN/pretrained_model/train_config.json \
    --resume=true
```

Or resume from the latest checkpoint automatically:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
lerobot-train \
    --config_path="$CKPT/train_config.json" \
    --resume=true
```

## 9. Upload trained policy to Hugging Face

Login first if needed:

```bash
hf auth login
```

Upload the latest checkpoint:

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
hf upload RobotLearningProject/act_so101_pickplace "$CKPT"
```

## 10. Fine-tune from an existing Hugging Face policy

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

## 11. Run async inference (policy server)

Use this when laptop CPU inference is too slow. The server runs the policy and streams action chunks to a `RobotClient` on the laptop, which talks to the SO-101 over USB locally. Laptop side is in [robot_setup/INSTRUCTIONS.md](INSTRUCTIONS.md) § 7b.

### 11.1 One-time install

`QuicksetupScripts/lerobotSetup.sh` now installs the `async` extra automatically. If your env predates that change:

```bash
cd ~/robot-learning-rl-project/robot_setup/lerobot_src
pip install -e '.[async]' -q
```

Sanity check:

```bash
python -c "import grpc, lerobot.async_inference.policy_server; print('ok')"
```

### 11.2 Start the policy server

```bash
tmux new -s policy-server     # so it survives SSH disconnect
conda activate lerobot

python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

The server starts empty — the policy is selected during the first handshake with the client, so you do **not** pass `--policy.path` here. Detach tmux with `Ctrl+B` then `D`. Re-attach with `tmux attach -t policy-server`.

### 11.3 Expose port 8080 to the laptop

Two options. **SSH tunnel is preferred** — simpler, encrypted, no public exposure.

**Option A — SSH local-forward (recommended).** Nothing to do on the server. The laptop forwards `localhost:8080` over the existing SSH connection (see laptop instructions § 7b.3).

**Option B — Public Brev port.** Brev dashboard → your instance → Networking → expose port `8080`. Use the public URL Brev returns as `server_address` on the laptop. The gRPC service has **no auth** — only do this on a trusted network or temporarily.

## 12. One-shot activation plus training

Use this if you want to paste one block into a fresh server shell:

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
