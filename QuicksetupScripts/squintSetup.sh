
#!/usr/bin/env bash
# Step 2 - Squint setup: conda env from environment.yaml

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
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
fi


# ── 2. Make conda available IN THIS SCRIPT ────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"


# ── 3. Ensure conda works (safety check) ──────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found after installation"
    exit 1
fi


# ── 4. Initialize shell for future sessions (idempotent) ──────────────────────
if ! grep -q "conda.sh" "$HOME/.bashrc"; then
    "$HOME/miniconda3/bin/conda" init bash
fi


# ── 5. Create Squint environment from YAML ───────────────────────────────────
echo "Creating Squint conda environment from environment.yaml..."

if conda env list | grep -q "^squint"; then
    echo "Squint environment already exists, skipping creation."
else
    conda env create -f "$REPO_DIR/sim/environment.yaml"
fi


# ── 6. Activate environment (IMPORTANT: must use eval hook) ──────────────────
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate squint


echo ""
echo "Squint environment ready."
echo "Activate manually with: conda activate squint"

