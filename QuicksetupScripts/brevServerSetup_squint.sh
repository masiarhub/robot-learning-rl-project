#!/usr/bin/env bash
# Step 1 - Server setup: clone repo and install VS Code extensions.
# Run once per fresh instance inside the connected VS Code window.
# Usage: bash QuicksetupScripts/brevServerSetup.sh

set -euo pipefail

REPO="masiarhub/robot-learning-rl-project"
REPO_DIR="$HOME/$(basename "$REPO")"
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

git config --global user.email "pcwagner2000@gmail.com"
git config --global user.name "Paul Wagner"

# 0. Re-launch inside tmux so the session survives SSH disconnect.
if [ -z "${TMUX:-}" ]; then
    if ! command -v tmux &>/dev/null; then
        sudo apt-get update -y -q
        sudo apt-get install -y -q tmux
    fi
    SESSION="server-setup"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Attaching to existing tmux session '$SESSION'..."
        exec tmux attach-session -t "$SESSION"
    fi
    echo "Relaunching inside tmux session '$SESSION'..."
    exec tmux new-session -s "$SESSION" "bash '$SCRIPT_PATH' $*; exec bash"
fi

# 1. Clone repo with submodules.
if [ ! -d "$REPO_DIR/.git" ]; then
    ensure_github_token "cloning" "$REPO"
    git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$REPO_DIR"
fi

# Switch to Squint branch before initialising submodules
git -C "$REPO_DIR" checkout squint
git -C "$REPO_DIR" submodule update --init --recursive

# 2. Install VS Code extensions.
VSCODE_EXTENSIONS=(
    "openai.chatgpt"
    "anthropic.claude-code"
    "mhutchie.git-graph"
)

find_vscode_cli() {
    local cli
    cli=$(ls "$HOME/.vscode-server/bin/"*/bin/code 2>/dev/null | tail -1)
    [ -n "$cli" ] && { echo "$cli"; return; }

    cli=$(ls "$HOME/.vscode-server/cli/servers/"*/server/bin/remote-cli/code 2>/dev/null | tail -1)
    [ -n "$cli" ] && { echo "$cli"; return; }

    command -v code 2>/dev/null && return
    return 1
}

echo ""
echo "Installing VS Code extensions..."
VSCODE_CLI=$(find_vscode_cli) || {
    echo "  x VS Code Server CLI not found."
    echo "    Open this folder in VS Code via Remote-SSH first, then re-run the script."
    exit 1
}

echo "  Using CLI: $VSCODE_CLI"

for EXT in "${VSCODE_EXTENSIONS[@]}"; do
    "$VSCODE_CLI" --install-extension "$EXT" --force \
        && echo "  ok $EXT" \
        || echo "  ! $EXT (install failed - install manually in VS Code)"
done

# 3. Install system monitoring tools and dependencies.
echo ""
echo "Installing btop, nvtop and dependencies..."
sudo apt-get update -y -q
sudo apt-get install -y -q btop nvtop libgl1 \
    && echo "  ok btop + nvtop + libgl1" \
    || echo "  ! install failed"
echo ""
echo "Server setup complete. Repo is at: $REPO_DIR"
echo "Next: run bash QuicksetupScripts/squintSetup.sh"