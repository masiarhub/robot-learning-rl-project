#!/bin/bash
# Step 2 — LeRobot setup: conda env, dependencies, and HuggingFace login.
# Run after setup_1_server.sh.
# Usage: bash setup_2_lerobot.sh

set -e

REPO_DIR="$HOME/robot-learning-rl-project"

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

# ── 2. Conda env ──────────────────────────────────────────────────────────────
# Accept Anaconda ToS for default channels (required since Miniconda 24.x).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

if ! conda env list | grep -q "^lerobot"; then
    echo "Creating lerobot conda environment..."
    conda create -n lerobot python=3.12 -y
fi

conda activate lerobot

# ── 3. Install lerobot ────────────────────────────────────────────────────────
echo "Installing lerobot..."
cd "$REPO_DIR/robot_setup/lerobot_src"
pip install -e '.[dataset,training]' -q
cd "$REPO_DIR"
pip install pynput -q

# ── 4. HuggingFace login ──────────────────────────────────────────────────────
if ! hf auth whoami &>/dev/null; then
    echo ""
    echo "Logging in to HuggingFace (paste your write-access token from https://huggingface.co/settings/tokens)..."
    MAX_RETRIES=3
    for attempt in $(seq 1 $MAX_RETRIES); do
        if hf auth login; then
            echo "  ✓ HuggingFace login successful."
            break
        fi
        if [ "$attempt" -eq "$MAX_RETRIES" ]; then
            echo ""
            echo "  ✗ HuggingFace login failed after $MAX_RETRIES attempts."
            echo "    Try running manually: HF_DEBUG=1 hf auth login"
            echo "    Or set your token directly: hf auth login --token <your-token>"
            exit 1
        fi
        echo "  Login failed (attempt $attempt/$MAX_RETRIES), retrying..."
    done
else
    echo "HuggingFace: already logged in, skipping."
fi

echo ""
echo "LeRobot setup complete."
