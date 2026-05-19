#!/usr/bin/env bash
# One-shot bootstrap for a fresh Linux/GPU machine.
#
# Download this file onto the new machine, run:
#   bash bootstrap_lerobot_act.sh
#
# It installs basic system packages, clones this project, initializes the
# LeRobot submodule, creates the conda env, installs LeRobot with ACT support,
# logs into Hugging Face, and runs a small smoke test.

set -euo pipefail

REPO="${REPO:-masiarhub/robot-learning-rl-project}"
BRANCH="${BRANCH:-lerobot-setup}"
REPO_DIR="${REPO_DIR:-$HOME/robot-learning-rl-project}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
TMUX_SESSION="${TMUX_SESSION:-lerobot-act-setup}"
export REPO_DIR

SCRIPT_PATH="$(realpath "$0")"

usage() {
    cat <<EOF
Usage: bash bootstrap_lerobot_act.sh [--no-tmux] [--repo owner/name] [--branch branch] [--dir path]

Environment overrides:
  GITHUB_TOKEN     GitHub token for private repo/submodule access
  HF_TOKEN         Hugging Face token for downloading/uploading models
  HF_REPO_PREFIX   Hugging Face username/org for uploaded model repos
  REPO             GitHub repo, default: ${REPO}
  BRANCH           Branch to checkout, default: ${BRANCH}
  REPO_DIR         Clone target, default: ${REPO_DIR}
  CONDA_DIR        Miniconda target, default: ${CONDA_DIR}
EOF
}

USE_TMUX=1
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-tmux)
            USE_TMUX=0
            shift
            ;;
        --repo)
            REPO="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --dir)
            REPO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

need_cmd() {
    command -v "$1" >/dev/null 2>&1
}

prompt_secret_if_empty() {
    local var_name="$1"
    local prompt="$2"
    local current_value="${!var_name:-}"

    if [ -z "$current_value" ]; then
        read -r -s -p "$prompt: " current_value
        echo
        export "$var_name=$current_value"
    fi
}

prompt_value_if_empty() {
    local var_name="$1"
    local prompt="$2"
    local current_value="${!var_name:-}"

    if [ -z "$current_value" ]; then
        read -r -p "$prompt: " current_value
        export "$var_name=$current_value"
    fi
}

git_with_optional_token() {
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git -c "http.https://github.com/.extraheader=Authorization: Bearer ${GITHUB_TOKEN}" "$@"
    else
        git "$@"
    fi
}

write_quicksetup_env() {
    local env_dir="$REPO_DIR/QuicksetupScripts"
    local env_file="$env_dir/.env"

    mkdir -p "$env_dir"
    touch "$env_file"
    chmod 600 "$env_file" 2>/dev/null || true

    {
        printf "GITHUB_TOKEN=%q\n" "${GITHUB_TOKEN:-}"
        printf "HF_TOKEN=%q\n" "${HF_TOKEN:-}"
        printf "HF_REPO_PREFIX=%q\n" "${HF_REPO_PREFIX:-}"
    } > "$env_file"
}

echo ""
echo "== LeRobot ACT one-shot setup =="
echo "Repo:    $REPO"
echo "Branch:  $BRANCH"
echo "Target:  $REPO_DIR"
echo ""

if [ "$(uname -s)" != "Linux" ]; then
    echo "This bootstrap is intended for Ubuntu/Linux GPU machines."
    echo "Detected: $(uname -s)"
    exit 1
fi

if [ "$USE_TMUX" -eq 1 ] && [ -z "${TMUX:-}" ]; then
    if ! need_cmd tmux; then
        if need_cmd sudo; then
            sudo apt-get update -y -q
            sudo apt-get install -y -q tmux
        else
            echo "tmux is missing and sudo is unavailable. Re-run with --no-tmux or install tmux."
            exit 1
        fi
    fi

    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Attaching to existing tmux session '$TMUX_SESSION'..."
        exec tmux attach-session -t "$TMUX_SESSION"
    fi

    echo "Relaunching inside tmux session '$TMUX_SESSION' so disconnects do not kill setup..."
    exec tmux new-session -s "$TMUX_SESSION" "bash '$SCRIPT_PATH' --no-tmux --repo '$REPO' --branch '$BRANCH' --dir '$REPO_DIR'; exec bash"
fi

echo ""
echo "Installing system packages..."
sudo apt-get update -y -q
sudo apt-get install -y -q \
    bash \
    btop \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgl1 \
    nvtop \
    tmux \
    wget

if ! need_cmd nvidia-smi; then
    echo ""
    echo "Warning: nvidia-smi was not found."
    echo "Install/enable the NVIDIA driver on this machine before GPU training."
else
    echo ""
    echo "Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
fi

echo ""
echo "Access tokens. Press Enter to leave GitHub blank if the repo/submodules are public."
prompt_secret_if_empty "GITHUB_TOKEN" "GitHub token"
prompt_secret_if_empty "HF_TOKEN" "Hugging Face token"
prompt_value_if_empty "HF_REPO_PREFIX" "Hugging Face username/org prefix"

echo ""
echo "Cloning/updating project..."
if [ ! -d "$REPO_DIR/.git" ]; then
    mkdir -p "$(dirname "$REPO_DIR")"
    git_with_optional_token clone "https://github.com/${REPO}.git" "$REPO_DIR"
else
    echo "Repo already exists, fetching latest refs..."
    git_with_optional_token -C "$REPO_DIR" fetch --all --prune
fi

git_with_optional_token -C "$REPO_DIR" checkout "$BRANCH"
git_with_optional_token -C "$REPO_DIR" pull --ff-only origin "$BRANCH" || true
git -C "$REPO_DIR" remote set-url origin "https://github.com/${REPO}.git"

write_quicksetup_env

echo ""
echo "Initializing LeRobot submodule..."
git_with_optional_token -C "$REPO_DIR" submodule sync --recursive
git_with_optional_token -C "$REPO_DIR" submodule update --init --recursive

echo ""
echo "Running canonical LeRobot setup..."
bash "$REPO_DIR/QuicksetupScripts/lerobotSetup.sh"

echo ""
echo "Running smoke test..."
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate lerobot
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
lerobot-train --help >/dev/null

echo ""
echo "Setup complete."
echo "Next shell:"
echo "  export PATH=\"\$HOME/miniconda3/bin:\$PATH\""
echo "  source \"\$HOME/miniconda3/etc/profile.d/conda.sh\""
echo "  conda activate lerobot"
echo "Project folder:"
echo "  $REPO_DIR"
