#!/bin/bash
# Full server setup and training script for Brev/Shadeform GPU instances.
# Run once per fresh instance: bash robot_setup/lerobot_brev_setup.sh
# After it finishes a VS Code tunnel URL is printed — open it in local VS Code.

set -e

# ── 0. Re-launch inside tmux so the tunnel survives SSH disconnect ────────────
if [ -z "$TMUX" ]; then
    if ! command -v tmux &>/dev/null; then
        sudo apt-get install -y -q tmux
    fi
    SESSION="lerobot-setup"
    # If the session already exists, just attach to it.
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Attaching to existing tmux session '$SESSION'..."
        exec tmux attach-session -t "$SESSION"
    fi
    echo "Relaunching inside tmux session '$SESSION'..."
    exec tmux new-session -s "$SESSION" "bash $(realpath "$0") $*; exec bash"
fi

# ── 1. Conda ────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc"
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# ── 2. Conda env ─────────────────────────────────────────────────────────────
if ! conda env list | grep -q "^lerobot"; then
    echo "Creating lerobot conda environment..."
    conda create -n lerobot python=3.10 -y
fi

conda activate lerobot

# ── 3. Install lerobot ───────────────────────────────────────────────────────
echo "Installing lerobot..."
cd robot_setup/lerobot_src
pip install -e '.[dataset,train]' -q
cd ../..
pip install pynput -q

# ── 4. HuggingFace login ─────────────────────────────────────────────────────
echo ""
echo "Logging in to HuggingFace (paste your write-access token)..."
hf auth login

# ── 5. System monitoring tools ───────────────────────────────────────────────
echo "Installing nvtop and btop..."
sudo apt-get install -y -q nvtop btop

# ── 6. Node.js ───────────────────────────────────────────────────────────────
# Needed for Claude Code CLI. Use the system package manager to avoid
# conda-nodejs conflicts with native addons.
if ! command -v node &>/dev/null; then
    echo "Installing Node.js (LTS)..."
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - -q
    sudo apt-get install -y -q nodejs
fi

# ── 7. GitHub CLI (gh) ───────────────────────────────────────────────────────
# Required to authenticate GitHub Copilot inside the VS Code server.
if ! command -v gh &>/dev/null; then
    echo "Installing GitHub CLI..."
    (type -p wget >/dev/null || sudo apt-get install -y wget) \
        && sudo mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) \
        && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && sudo cat "$out" | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null \
        && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
            | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null \
        && sudo apt-get update -q \
        && sudo apt-get install -y -q gh
fi

# ── 8. VS Code CLI ───────────────────────────────────────────────────────────
if ! command -v code &>/dev/null; then
    echo "Installing VS Code CLI..."
    curl -sLk \
        "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" \
        -o /tmp/vscode_cli.tar.gz
    mkdir -p "$HOME/.local/bin"
    tar -xf /tmp/vscode_cli.tar.gz -C "$HOME/.local/bin"
    rm /tmp/vscode_cli.tar.gz
    export PATH="$HOME/.local/bin:$PATH"
    # Persist for future shells
    grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" \
        || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

# ── 9. Claude Code CLI ───────────────────────────────────────────────────────
if ! command -v claude &>/dev/null; then
    echo "Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code -q
fi

# ── 10. Authenticate GitHub (Copilot) ────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " GitHub login — required for GitHub Copilot in VS Code"
echo "═══════════════════════════════════════════════════════"
gh auth login --web -h github.com

# ── 11. Authenticate Claude Code ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " Claude Code login"
echo "═══════════════════════════════════════════════════════"
claude auth login

# ── 12. Start VS Code tunnel & install extensions ────────────────────────────
# The tunnel downloads the VS Code server on first run, then stays alive.
# Open the printed URL in your local VS Code (Remote Tunnels extension).
TUNNEL_NAME="gpu-server"
VSCODE_EXTENSIONS=(
    "GitHub.copilot"
    "GitHub.copilot-chat"
    "anthropic.claude-code"
    "mhutchie.git-graph"
)

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Starting VS Code tunnel  (name: $TUNNEL_NAME)"
echo " 1. Follow the GitHub device-auth prompt below."
echo " 2. Once connected, open VS Code locally and go to:"
echo "    Remote Explorer → Tunnels → $TUNNEL_NAME"
echo "═══════════════════════════════════════════════════════"

# Accept the server license non-interactively, then hand control to the tunnel.
# The tunnel command blocks; Ctrl-C stops it (use tmux/screen for persistence).
code tunnel --accept-server-license-terms --name "$TUNNEL_NAME" &
TUNNEL_PID=$!

# Wait for the vscode-server binary to appear (downloaded on first connect).
echo "Waiting for VS Code server to initialise..."
SERVER_BIN=""
for i in $(seq 1 60); do
    SERVER_BIN=$(find "$HOME/.vscode/cli/servers" -name "code-server" -type f 2>/dev/null | head -1)
    [ -n "$SERVER_BIN" ] && break
    sleep 2
done

if [ -n "$SERVER_BIN" ]; then
    echo "Installing VS Code extensions on server..."
    for EXT in "${VSCODE_EXTENSIONS[@]}"; do
        "$SERVER_BIN" --install-extension "$EXT" --force 2>/dev/null \
            && echo "  ✓ $EXT" \
            || echo "  ! $EXT (install failed — install manually in VS Code)"
    done
else
    echo "Server binary not found yet — extensions will install automatically"
    echo "when you first open VS Code and connect to the tunnel."
fi

echo ""
echo "Setup complete. The tunnel is running in the background (PID $TUNNEL_PID)."
echo "To keep it alive after logout: run this script inside tmux or screen."
wait $TUNNEL_PID