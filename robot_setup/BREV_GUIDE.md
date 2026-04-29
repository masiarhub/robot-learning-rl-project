# Brev GPU Training Guide

Training runs on a rented NVIDIA A6000 instance via Brev/Shadeform. This guide covers everything from connecting to running training.

---

## Instance details

| Field        | Value                        |
|--------------|------------------------------|
| Name         | `evil-peach-crayfish`        |
| IP           | `38.128.233.14`              |
| GPU          | NVIDIA RTX A6000 (48 GiB)   |
| Cost         | ~$0.60/hr — **stop when done** |

---

## 1. Connect to the instance

Open WSL (Ubuntu) on your Windows machine:

```bash
wsl
```

Then connect:

```bash
brev shell evil-peach-crayfish
```

If `brev` is not installed in WSL:

```bash
# Install Brev CLI (copy the exact command from your Brev dashboard)
# Then login:
brev login
brev shell evil-peach-crayfish
```

---

## 2. First-time setup on a fresh instance

Clone the repo (use a GitHub personal access token — password auth is disabled):

```bash
git clone https://YOUR_GITHUB_TOKEN@github.com/masiarhub/robot-learning-rl-project
cd robot-learning-rl-project
```

Get a token at https://github.com/settings/tokens (classic, `repo` scope).

Then run the full setup script — it installs conda, lerobot, logs into HuggingFace, trains, and uploads the model automatically:

```bash
tmux new -s train
bash robot_setup/brev_setup.sh
```

Detach from tmux with `Ctrl+B` then `D`. The training keeps running after you close the terminal.

---

## 3. Resuming after disconnect

Reconnect to the instance:

```bash
wsl
brev shell evil-peach-crayfish
```

Reattach to the running training session:

```bash
tmux attach -t train
```

List all tmux sessions if you forgot the name:

```bash
tmux ls
```

---

## 4. Check training progress

Inside the tmux session you'll see live loss output. To also check GPU usage:

```bash
# Open a second tmux window: Ctrl+B then C
watch -n 2 nvidia-smi
```

Switch between tmux windows with `Ctrl+B` then the window number (`0`, `1`, etc.).

---

## 5. Upload the trained model manually

If the script didn't upload automatically (e.g. you killed it early):

```bash
CKPT=$(ls -d outputs/train/act_so101_pickplace/checkpoints/*/pretrained_model | sort | tail -1)
echo "Uploading: $CKPT"
hf upload pcwagner/act_so101_pickplace "$CKPT"
```

---

## 6. Stop the instance

**Always stop the instance when training is done** — it costs $0.60/hr while running.

Go to https://brev.dev → your instance → **Stop**.

Or from the terminal before disconnecting:

```bash
sudo poweroff
```

---

## 7. Useful tmux cheatsheet

| Action                  | Keys                  |
|-------------------------|-----------------------|
| Detach (keep running)   | `Ctrl+B` then `D`     |
| New window              | `Ctrl+B` then `C`     |
| Switch window           | `Ctrl+B` then `0`–`9` |
| Split pane horizontally | `Ctrl+B` then `"`     |
| Kill current pane       | `Ctrl+B` then `X`     |
| Reattach session        | `tmux attach -t train` |
| List sessions           | `tmux ls`             |
