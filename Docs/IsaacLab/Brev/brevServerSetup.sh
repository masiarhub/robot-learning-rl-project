#!/usr/bin/env bash
# Step 1 - Server setup: clone repo, install VS Code extensions, set up Isaac Lab.
# Run once per fresh instance inside the connected VS Code window.
# Usage: bash Docs/IsaacLab/Brev/brevServerSetup.sh

set -euo pipefail

REPO="masiarhub/robot-learning-rl-project"
BASE_DIR="$HOME/robot-learning"
REPO_DIR="$BASE_DIR/$(basename "$REPO")"
ISAACLAB_VERSION="v2.3.0"
ISAACLAB_DIR="$BASE_DIR/IsaacLab"
CONTAINER_PROJECT="/workspace/robot-learning/$(basename "$REPO")"
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_PATH")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

# Apply git identity from .env
if [ -n "${GIT_AUTHOR_NAME:-}" ]; then
    git config --global user.name "${GIT_AUTHOR_NAME}"
fi
if [ -n "${GIT_AUTHOR_EMAIL:-}" ]; then
    git config --global user.email "${GIT_AUTHOR_EMAIL}"
fi

# 0. Re-launch inside tmux so the session survives SSH disconnect.
if [ -z "${TMUX:-}" ]; then
    if ! command -v tmux &>/dev/null; then
        sudo apt-get update -y -q
        sudo apt-get install -y -q tmux
    fi
    SESSION="isaac_setup"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Attaching to existing tmux session '$SESSION'..."
        exec tmux attach-session -t "$SESSION"
    fi
    echo "Relaunching inside tmux session '$SESSION'..."
    exec tmux new-session -s "$SESSION" "bash '$SCRIPT_PATH' $*; exec bash"
fi

# 1. Create base directory and clone repos.
mkdir -p "$BASE_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    ensure_github_token "cloning" "$REPO"
    git clone "https://${GITHUB_TOKEN}@github.com/${REPO}.git" "$REPO_DIR"
    # Strip token from remote URL immediately
    git -C "$REPO_DIR" remote set-url origin "https://github.com/${REPO}.git"
fi

# Switch to the correct branch before initialising submodules.
git -C "$REPO_DIR" checkout rl_isaac_lab
git -C "$REPO_DIR" submodule update --init --recursive

# 2. Clone Isaac Lab at the pinned version.
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
    echo ""
    echo "Cloning Isaac Lab ${ISAACLAB_VERSION}..."
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
    git -C "$ISAACLAB_DIR" checkout "$ISAACLAB_VERSION"
    echo "  ok Isaac Lab ${ISAACLAB_VERSION}"
else
    echo "  -- Isaac Lab already cloned, skipping."
fi

# 3. Replace the stock Dockerfile.base with our custom one.
cp "$SCRIPT_DIR/Dockerfile.base" "$ISAACLAB_DIR/docker/Dockerfile.base"
echo "  ok Dockerfile.base replaced."

# 4. Create override file: bind-mount repo and pass secrets into the container.
cat > "$ISAACLAB_DIR/docker/docker-compose.override.yaml" << EOF
services:
  isaac-lab-base:
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY:-}
    volumes:
      - type: bind
        source: ${REPO_DIR}
        target: ${CONTAINER_PROJECT}
EOF
echo "  ok docker-compose.override.yaml created."

echo ""
echo "Building and starting Isaac Lab container (first run takes ~15 min)..."
cd "$ISAACLAB_DIR"

# 5. Build and start the Isaac Lab Docker container.
./docker/container.py start --file docker-compose.override.yaml
echo "  ok Isaac Lab container running."

# 6. Configure the running container for the bind-mounted repo.
# (bind mount is only available at runtime, not during docker build)
docker exec isaac-lab-base \
    git config --global --add safe.directory "${CONTAINER_PROJECT}"
echo "  ok git safe.directory set."

docker exec isaac-lab-base \
    bash -c "\${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e '${CONTAINER_PROJECT}/isaac_so_arm101' --quiet" \
    && echo "  ok isaac_so_arm101 installed (editable)." \
    || echo "  ! isaac_so_arm101 install failed — run manually inside the container."

# 8. Install VS Code extensions.
VSCODE_EXTENSIONS=(
    "openai.chatgpt"
    "anthropic.claude-code"
    "mhutchie.git-graph"
)

find_vscode_cli() {
    local cli
    # Legacy layout: ~/.vscode-server/bin/<commit>/bin/code
    cli=$(ls "$HOME/.vscode-server/bin/"*/bin/code 2>/dev/null | tail -1)
    [ -n "$cli" ] && { echo "$cli"; return; }
    # Newer tunnel layout
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
        || echo "  ! $EXT (install failed — install manually in VS Code)"
done

# 9. Install system monitoring tools (btop, nvtop).
echo ""
echo "Installing btop and nvtop..."
sudo apt-get update -y -q
sudo apt-get install -y -q btop nvtop \
    && echo "  ok btop + nvtop" \
    || echo "  ! btop/nvtop install failed"

echo ""
echo "=========================================="
echo "  Server setup complete."
echo "  Repo:      $REPO_DIR"
echo "  Isaac Lab: $ISAACLAB_DIR (${ISAACLAB_VERSION})"
echo "  Container: running — enter with:"
echo "    $ISAACLAB_DIR/docker/container.py enter base"
echo "=========================================="
echo "Next: enter the container with the command above and run your training scripts."