#!/usr/bin/env bash
# Decommission script - run before shutting down a GPU instance.
# 1. Pushes repo to a decommissioning branch on GitHub.
# 2. Uploads trained policies to HuggingFace.
# Usage: bash QuicksetupScripts/brevServerDecomission.sh

set -euo pipefail

REPO_DIR="$HOME/robot-learning-rl-project"
POLICIES_DIR="$REPO_DIR/outputs"
SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

SERVER_NAME=$(hostname)
DATE=$(date +%Y-%m-%d)
BRANCH="decommissioning-${SERVER_NAME}-${DATE}"

echo "======================================================="
echo " Decommissioning: $SERVER_NAME  ($DATE)"
echo " Git branch : $BRANCH"
echo " Policies   : $POLICIES_DIR"
echo "======================================================="
echo ""

# 1. Git - commit any uncommitted changes and push.
cd "$REPO_DIR"

# Store the token for this session only.
ensure_github_token "pushing decommission branch"
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/masiarhub/robot-learning-rl-project.git"

git checkout -B "$BRANCH"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Committing uncommitted changes..."
    # Exclude outputs/ - large model files go to HuggingFace, not git.
    git add -A -- ':!outputs/'
    git commit -m "decommission: $SERVER_NAME $DATE"
fi

echo "Pushing $BRANCH to origin..."
git push -u origin "$BRANCH"
echo "  ok Branch pushed: $BRANCH"

# Restore origin URL without token.
git remote set-url origin "https://github.com/masiarhub/robot-learning-rl-project.git"

# 2. HuggingFace - upload trained policies.
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true

# Try lerobot env first; fall back to base which always has pip.
if conda activate lerobot 2>/dev/null; then
    echo "Using lerobot conda env for HuggingFace upload."
else
    echo "lerobot env not found - using base env."
    conda activate base 2>/dev/null || true
fi

# Ensure hf CLI is available regardless of which env is active.
if ! command -v hf &>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

ensure_hf_token

hf_login() {
    echo ""
    echo "HuggingFace login from QuicksetupScripts/.env..."
    hf auth login --token "$HF_TOKEN"
}

# Verify the token actually works against the server, not just locally.
if ! hf auth whoami 2>&1 | grep -qv "401\|Unauthorized\|Invalid"; then
    hf_login
fi

ensure_hf_repo_prefix

# Find every pretrained_model dir - one per training run / checkpoint.
mapfile -t MODEL_DIRS < <(find "$POLICIES_DIR" -type d -name "pretrained_model" 2>/dev/null)
UPLOADED_REPOS=()

if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
    echo "  ! No pretrained_model directories found in $POLICIES_DIR - skipping upload."
else
    for MODEL_DIR in "${MODEL_DIRS[@]}"; do
        # Derive a repo name from the run folder name, e.g. act_so101_pickplace_firstTry.
        RUN_NAME=$(echo "$MODEL_DIR" | sed 's|.*/train/||; s|/checkpoints/.*||')
        REPO_ID="${HF_REPO_PREFIX}/${RUN_NAME}"
        echo "Uploading $RUN_NAME -> $REPO_ID ..."
        if ! hf upload "$REPO_ID" "$MODEL_DIR" . --type model; then
            echo "  Upload failed - re-logging in and retrying..."
            hf_login
            hf upload "$REPO_ID" "$MODEL_DIR" . --type model
        fi
        UPLOADED_REPOS+=("https://huggingface.co/$REPO_ID")
        echo "  ok https://huggingface.co/$REPO_ID"
    done
fi

echo ""
echo "Decommissioning complete."
echo "  Git branch : https://github.com/masiarhub/robot-learning-rl-project/tree/$BRANCH"
if [ ${#UPLOADED_REPOS[@]} -gt 0 ]; then
    echo "  HF repos   :"
    printf '    %s\n' "${UPLOADED_REPOS[@]}"
else
    echo "  HF repos   : none uploaded"
fi
