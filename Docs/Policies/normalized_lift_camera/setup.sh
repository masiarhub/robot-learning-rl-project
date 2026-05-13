#!/usr/bin/env bash
# Deploy-environment setup for SO-ARM101 LiftCamera policy.
# Run once on the deployment laptop before the first rollout.
#
# Requirements:
#   - Python 3.10 or 3.11
#   - uv  (https://docs.astral.sh/uv/ — install with: curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - git
#   - USB or RealSense wrist camera
#   - SO-ARM101 connected via USB serial

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv_deploy"

# ── 0. Check uv is available ───────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── 1. Create venv ─────────────────────────────────────────────────────────
echo "[1/4] Creating virtual environment at $VENV_DIR ..."
uv venv "$VENV_DIR" --python 3.12

# ── 2. Install PyTorch ─────────────────────────────────────────────────────
echo "[2/4] Installing PyTorch ..."
if nvidia-smi &>/dev/null; then
    echo "  GPU detected — installing CUDA build."
    uv pip install torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --python "$VENV_DIR"
else
    echo "  No GPU detected — installing CPU build."
    uv pip install torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cpu \
        --python "$VENV_DIR"
fi

# ── 3. Install remaining dependencies ─────────────────────────────────────
echo "[3/4] Installing dependencies ..."
uv pip install \
    opencv-python \
    --python "$VENV_DIR"

uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git" \
    --python "$VENV_DIR"

# ── 4. Smoke test ──────────────────────────────────────────────────────────
echo "[4/4] Verifying installation ..."
"$VENV_DIR/bin/python" - <<'EOF'
import torch, torchvision, cv2
import lerobot
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
import torchvision.models as tv_models
tv_models.resnet18(weights=None)
print(f"  torch       {torch.__version__}")
print(f"  torchvision {torchvision.__version__}")
print(f"  opencv      {cv2.__version__}")
print(f"  lerobot     {lerobot.__version__}")
print("  All imports OK.")
EOF

echo ""
echo "Setup complete. To activate:"
echo "  source $VENV_DIR/bin/activate"
