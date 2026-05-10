#!/usr/bin/env bash
# Decommission script - run before shutting down a GPU instance.
# 1. Pushes repo to GitHub decommission branch.
# 2. Uploads Squint trained checkpoints to HuggingFace.
# Usage: bash QuicksetupScripts/brevServerDecomission.sh

set -euo pipefail

REPO_DIR="$HOME/robot-learning-rl-project"
RUNS_DIR="$REPO_DIR/runs"
SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

SERVER_NAME=$(hostname)
DATE=$(date +%Y-%m-%d)
BRANCH="decommissioning-${SERVER_NAME}-${DATE}"

echo "======================================================="
echo " Squint Decommission: $SERVER_NAME ($DATE)"
echo " Git branch : $BRANCH"
echo " Runs       : $RUNS_DIR"
echo "======================================================="
echo ""

# ─────────────────────────────────────────────
# 1. Git push (snapshot of code)
# ─────────────────────────────────────────────

cd "$REPO_DIR"

ensure_github_token "pushing decommission branch"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/masiarhub/robot-learning-rl-project.git"

git checkout -B "$BRANCH"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Committing changes..."
    git add -A
    git commit -m "squint decommission: $SERVER_NAME $DATE"
fi

echo "Pushing branch..."
git push -u origin "$BRANCH"

git remote set-url origin "https://github.com/masiarhub/robot-learning-rl-project.git"

echo "  ok Git branch pushed: $BRANCH"

# ─────────────────────────────────────────────
# 2. Hugging Face upload (Squint models)
# ─────────────────────────────────────────────

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

conda activate base 2>/dev/null || true

if ! command -v hf &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

ensure_hf_token
ensure_hf_repo_prefix

hf_login() {
    echo ""
    echo "HuggingFace login..."
    hf auth login --token "$HF_TOKEN"
}

hf_login

# Find all Squint checkpoints
mapfile -t CKPT_FILES < <(find "$RUNS_DIR" -type f -name "ckpt.pt" 2>/dev/null)

if [ ${#CKPT_FILES[@]} -eq 0 ]; then
    echo "  ! No ckpt.pt files found in $RUNS_DIR"
else
    echo ""
    echo "Found ${#CKPT_FILES[@]} checkpoints. Uploading..."

    UPLOADED_REPOS=()

    for CKPT in "${CKPT_FILES[@]}"; do

        RUN_DIR=$(dirname "$CKPT")
        RUN_NAME=$(basename "$RUN_DIR")

        REPO_ID="${HF_REPO_PREFIX}/${RUN_NAME}"

        echo ""
        echo "Uploading run: $RUN_NAME"
        echo " -> $REPO_ID"

        # retry once if auth fails
        if ! hf upload "$REPO_ID" "$RUN_DIR" . --type model; then
            echo "  retrying login..."
            hf_login
            hf upload "$REPO_ID" "$RUN_DIR" . --type model
        fi

        UPLOADED_REPOS+=("https://huggingface.co/$REPO_ID")
        echo "  ok uploaded"
    done
fi

# ─────────────────────────────────────────────
# 3. Summary
# ─────────────────────────────────────────────

echo ""
echo "======================================================="
echo " Decommission complete"
echo "======================================================="
echo "Git branch:"
echo "  https://github.com/masiarhub/robot-learning-rl-project/tree/$BRANCH"

echo ""
echo "Hugging Face models:"
if [ ${#UPLOADED_REPOS[@]} -gt 0 ]; then
    printf '  %s\n' "${UPLOADED_REPOS[@]}"
else
    echo "  none uploaded"
fi