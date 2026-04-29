#!/bin/bash
# Step 1 — Server setup: clone repo and install VS Code extensions.
# Run once per fresh instance inside the connected VS Code window.
# Usage: bash setup_1_server.sh

set -e

REPO="masiarhub/robot-learning-rl-project"
REPO_DIR="$HOME/$(basename "$REPO")"

# ── 0. Re-launch inside tmux so the session survives SSH disconnect ───────────
if [ -z "$TMUX" ]; then
    if ! command -v tmux &>/dev/null; then
        sudo apt-get install -y -q tmux
    fi
    SESSION="server-setup"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Attaching to existing tmux session '$SESSION'..."
        exec tmux attach-session -t "$SESSION"
    fi
    echo "Relaunching inside tmux session '$SESSION'..."
    exec tmux new-session -s "$SESSION" "bash $(realpath "$0") $*; exec bash"
fi

# ── 1. Clone repo with submodules ─────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " GitHub token for cloning (needs 'repo' scope only)"
    echo " Generate at: github.com/settings/tokens/new"
    echo "═══════════════════════════════════════════════════════"
    read -r -s -p "GitHub token: " GITHUB_TOKEN
    echo
    git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$REPO_DIR"
    unset GITHUB_TOKEN
fi

# Switch to the correct branch before initialising submodules — the submodule
# config (.gitmodules) may only exist or differ on this branch.
git -C "$REPO_DIR" checkout lerobot-setup
git -C "$REPO_DIR" submodule update --init --recursive

# ── 2. Install VS Code extensions ─────────────────────────────────────────────
VSCODE_EXTENSIONS=(
    "openai.chatgpt"
    "anthropic.claude-code"
)

echo ""
echo "Installing VS Code extensions..."
for EXT in "${VSCODE_EXTENSIONS[@]}"; do
    code --install-extension "$EXT" --force \
        && echo "  ✓ $EXT" \
        || echo "  ! $EXT (install failed — install manually in VS Code)"
done

echo ""
echo "Server setup complete. Repo is at: $REPO_DIR"
echo "Next: run bash setup_2_lerobot.sh"
