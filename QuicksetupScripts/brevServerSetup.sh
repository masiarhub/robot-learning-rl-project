#!/usr/bin/env bash
# Step 1 - Server setup: clone repo and install VS Code extensions.
# Run once per fresh instance inside the connected VS Code window.
# Usage: bash QuicksetupScripts/brevServerSetup.sh

set -euo pipefail

REPO="masiarhub/robot-learning-rl-project"
REPO_DIR="$HOME/$(basename "$REPO")"
SCRIPT_PATH=$(realpath "$0")

# 0. Re-launch inside tmux so the session survives SSH disconnect.
if [ -z "$TMUX" ]; then
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
    echo ""
    echo "======================================================="
    echo " GitHub token for cloning (needs 'repo' scope only)"
    echo " Generate at: github.com/settings/tokens/new"
    echo "======================================================="
    read -r -s -p "GitHub token: " GITHUB_TOKEN
    echo
    git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$REPO_DIR"
    unset GITHUB_TOKEN
fi

# Switch to the correct branch before initialising submodules. The submodule
# config (.gitmodules) may only exist or differ on this branch.
git -C "$REPO_DIR" checkout lerobot-setup
git -C "$REPO_DIR" submodule update --init --recursive

# 2. Install VS Code extensions.
VSCODE_EXTENSIONS=(
    "openai.chatgpt"
    "anthropic.claude-code"
    "mhutchie.git-graph"
)

# Find the VS Code Server CLI. It lives under ~/.vscode-server after VS Code
# connects via Remote SSH. Try both the legacy layout and the newer tunnel layout.
find_vscode_cli() {
    # Legacy: ~/.vscode-server/bin/<commit>/bin/code
    local cli
    cli=$(ls "$HOME/.vscode-server/bin/"*/bin/code 2>/dev/null | tail -1)
    [ -n "$cli" ] && { echo "$cli"; return; }
    # Newer tunnel layout: ~/.vscode-server/cli/servers/Stable-<commit>/server/bin/remote-cli/code
    cli=$(ls "$HOME/.vscode-server/cli/servers/"*/server/bin/remote-cli/code 2>/dev/null | tail -1)
    [ -n "$cli" ] && { echo "$cli"; return; }
    # Fall back to whatever is in PATH
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

# 3. Install system monitoring tools.
echo ""
echo "Installing btop and nvtop..."
sudo apt-get update -y -q
sudo apt-get install -y -q btop nvtop \
    && echo "  ok btop + nvtop" \
    || echo "  ! btop/nvtop install failed"

echo ""
echo "Server setup complete. Repo is at: $REPO_DIR"
echo "Next: run bash QuicksetupScripts/lerobotSetup.sh"
