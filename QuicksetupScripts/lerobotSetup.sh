#!/usr/bin/env bash
# Step 2 - LeRobot setup: conda env, dependencies, and HuggingFace login.
# Run after QuicksetupScripts/brevServerSetup.sh.
# Usage: bash QuicksetupScripts/lerobotSetup.sh

set -euo pipefail

REPO_DIR="$HOME/robot-learning-rl-project"
SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

# Switch to the correct branch before initialising submodules — the submodule
# config (.gitmodules) may only exist or differ on this branch.
git -C "$REPO_DIR" checkout lerobot-setup
git -C "$REPO_DIR" submodule update --init --recursive



# ── 1. Conda ──────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc"
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. Conda env.
# Accept Anaconda ToS for default channels (required since Miniconda 24.x).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

if ! conda env list | grep -q "^lerobot"; then
    echo "Creating lerobot conda environment..."
    conda create -n lerobot python=3.12 -y
fi

conda activate lerobot

# 3. Install FFmpeg runtime libraries for TorchCodec video decoding.
echo "Installing FFmpeg runtime libraries..."
conda install -c conda-forge ffmpeg -y

# 4. Install lerobot.
echo "Installing lerobot..."
if [ ! -d "$REPO_DIR/robot_setup/lerobot_src" ]; then
    echo "Missing LeRobot source at: $REPO_DIR/robot_setup/lerobot_src"
    echo "Run QuicksetupScripts/brevServerSetup.sh first, or clone submodules manually."
    exit 1
fi
cd "$REPO_DIR/robot_setup/lerobot_src"
pip install -e '.[dataset,training,async]' -q
cd "$REPO_DIR"
pip install pynput -q

# 5. HuggingFace login.
ensure_hf_token
if ! hf auth whoami &>/dev/null; then
    echo ""
    echo "Logging in to HuggingFace from QuicksetupScripts/.env..."
    hf auth login --token "$HF_TOKEN"
    echo "  ok HuggingFace login successful."
else
    echo "HuggingFace: already logged in, skipping."
fi

echo ""
echo "LeRobot setup complete."
