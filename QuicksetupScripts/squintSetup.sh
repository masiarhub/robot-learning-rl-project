#!/usr/bin/env bash
# Step 2 - Squint setup: conda env from environment.yaml
# Run after QuicksetupScripts/brevServerSetup.sh.
# Usage: bash QuicksetupScripts/squintSetup.sh

set -euo pipefail

REPO_DIR="$HOME/robot-learning-rl-project"
SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

# Switch to squint branch
git -C "$REPO_DIR" checkout squint
git -C "$REPO_DIR" submodule update --init --recursive

# ── 1. Conda bootstrap ───────────────────────────────────────────────────────
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

# ── 2. Create Squint environment from YAML ───────────────────────────────────
echo "Creating Squint conda environment from environment.yaml..."

if conda env list | grep -q "^squint"; then
    echo "Squint environment already exists, skipping creation."
else
    conda env create -f "$REPO_DIR/sim/environment.yaml"
fi

# ── 3. Persist conda init ─────────────────────────────────────────────────────
"$HOME/miniconda3/bin/conda" init bash
echo "source ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc


# ── 4. Activate environment ──────────────────────────────────────────────────
conda activate squint

echo ""
echo "Squint environment ready."
echo "Activate manually with: conda activate squint"

