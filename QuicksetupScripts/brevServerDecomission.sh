#!/bin/bash
# Decommission script — run before shutting down a GPU instance.
# 1. Pushes repo to a decommissioning branch on GitHub.
# 2. Uploads trained policies to HuggingFace.
# Usage: bash decommission.sh

set -e

REPO_DIR="$HOME/robot-learning-rl-project"
POLICIES_DIR="$REPO_DIR/outputs"

SERVER_NAME=$(hostname)
DATE=$(date +%Y-%m-%d)
BRANCH="decommissioning-${SERVER_NAME}-${DATE}"

echo "═══════════════════════════════════════════════════════"
echo " Decommissioning: $SERVER_NAME  ($DATE)"
echo " Git branch : $BRANCH"
echo " Policies   : $POLICIES_DIR"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── 1. Git — commit any uncommitted changes and push ─────────────────────────
cd "$REPO_DIR"

# Store the token for this session only.
echo ""
echo "GitHub token (needs 'repo' scope) for push:"
read -r -s -p "GitHub token: " GITHUB_TOKEN
echo
git remote set-url origin "https://${GITHUB_TOKEN}@github.com/masiarhub/robot-learning-rl-project.git"
unset GITHUB_TOKEN

git checkout -B "$BRANCH"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Committing uncommitted changes..."
    git add -A
    git commit -m "decommission: $SERVER_NAME $DATE"
fi

echo "Pushing $BRANCH to origin..."
git push -u origin "$BRANCH"
echo "  ✓ Branch pushed: $BRANCH"

# Restore origin URL without token.
git remote set-url origin "https://github.com/masiarhub/robot-learning-rl-project.git"

# ── 2. HuggingFace — upload trained policies ──────────────────────────────────
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate lerobot 2>/dev/null || true

if ! hf auth whoami &>/dev/null; then
    echo ""
    echo "HuggingFace login required..."
    hf auth login
fi

HF_USER=$(hf auth whoami 2>/dev/null | head -1)
echo ""
echo "Logged in to HuggingFace as: $HF_USER"
read -r -p "HuggingFace repo to upload policies to [${HF_USER}/robot-policies]: " HF_REPO
HF_REPO="${HF_REPO:-${HF_USER}/robot-policies}"

if [ ! -d "$POLICIES_DIR" ] || [ -z "$(ls -A "$POLICIES_DIR" 2>/dev/null)" ]; then
    echo "  ! No policies found at $POLICIES_DIR — skipping upload."
else
    echo "Uploading policies from $POLICIES_DIR to $HF_REPO..."
    huggingface-cli upload "$HF_REPO" "$POLICIES_DIR" \
        "decommissioning/${SERVER_NAME}-${DATE}" \
        --repo-type model \
        --commit-message "decommission: $SERVER_NAME $DATE"
    echo "  ✓ Policies uploaded to: https://huggingface.co/$HF_REPO"
fi

echo ""
echo "Decommissioning complete."
echo "  Git branch : https://github.com/masiarhub/robot-learning-rl-project/tree/$BRANCH"
echo "  HF repo    : https://huggingface.co/$HF_REPO"