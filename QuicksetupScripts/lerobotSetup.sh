#!/usr/bin/env bash
# Step 2 - LeRobot setup: conda env, dependencies, and HuggingFace login.
# Run after QuicksetupScripts/brevServerSetup.sh.
# Usage: bash QuicksetupScripts/lerobotSetup.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/robot-learning-rl-project}"
SCRIPT_DIR=$(cd "$(dirname "$(realpath "$0")")" && pwd)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_env.sh"
load_quicksetup_env

# Switch to the correct branch before initialising submodules — the submodule
# config (.gitmodules) may only exist or differ on this branch.
git -C "$REPO_DIR" checkout lerobot-setup
git -C "$REPO_DIR" submodule update --init --recursive

apply_lerobot_local_patches() {
    local factory_py="$REPO_DIR/robot_setup/lerobot_src/src/lerobot/datasets/factory.py"
    local import_utils_py="$REPO_DIR/robot_setup/lerobot_src/src/lerobot/utils/import_utils.py"

    echo "Applying local LeRobot compatibility patches..."
    python - "$factory_py" "$import_utils_py" <<'PY'
from pathlib import Path
import sys

factory_path = Path(sys.argv[1])
import_utils_path = Path(sys.argv[2])

factory_text = factory_path.read_text()
factory_old = """    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
"""
factory_new = """    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
"""

if "dataset.meta.stats.setdefault(key, {})" not in factory_text:
    if factory_old not in factory_text:
        raise SystemExit(f"Could not apply ImageNet stats patch; expected block not found in {factory_path}")
    factory_text = factory_text.replace(factory_old, factory_new)

backend_old = """    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if isinstance(cfg.dataset.repo_id, str):
"""
backend_new = """    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    if cfg.dataset.video_backend == "torchcodec":
        logging.warning("Overriding dataset.video_backend='torchcodec' to 'pyav'.")
        cfg.dataset.video_backend = "pyav"

    if isinstance(cfg.dataset.repo_id, str):
"""

if "Overriding dataset.video_backend='torchcodec' to 'pyav'" not in factory_text:
    if backend_old not in factory_text:
        raise SystemExit(f"Could not apply pyav backend override patch; expected block not found in {factory_path}")
    factory_text = factory_text.replace(backend_old, backend_new)

factory_path.write_text(factory_text)

import_utils_text = import_utils_path.read_text()
codec_old = """def get_safe_default_codec():
    logger = logging.getLogger(__name__)
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logger.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"
"""
codec_new = """def get_safe_default_codec():
    return "pyav"
"""

if codec_new not in import_utils_text:
    if codec_old not in import_utils_text:
        raise SystemExit(f"Could not apply pyav default patch; expected block not found in {import_utils_path}")
    import_utils_path.write_text(import_utils_text.replace(codec_old, codec_new))
PY
}

# ── 1. Conda ──────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null && [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    source "$HOME/.bashrc"
fi

export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. Conda env.
# Accept Anaconda ToS for default channels (required since Miniconda 24.x).
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   2>/dev/null || true

if ! conda env list | grep -q "^lerobot"; then
    echo "Creating lerobot conda environment..."
    conda create -n lerobot python=3.12 -y
fi

conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
conda env config vars set LD_LIBRARY_PATH="$CONDA_PREFIX/lib" >/dev/null

# 3. Install FFmpeg runtime libraries for TorchCodec video decoding.
# TorchCodec currently probes FFmpeg 4-7 shared libraries; avoid newer FFmpeg
# majors that provide libavutil.so.60+ and fail at runtime.
echo "Installing TorchCodec-compatible FFmpeg runtime libraries..."
conda install -c conda-forge "ffmpeg>=6,<8" -y

# 4. Install lerobot.
echo "Installing lerobot..."
if [ ! -d "$REPO_DIR/robot_setup/lerobot_src" ]; then
    echo "Missing LeRobot source at: $REPO_DIR/robot_setup/lerobot_src"
    echo "Run QuicksetupScripts/brevServerSetup.sh first, or clone submodules manually."
    exit 1
fi
apply_lerobot_local_patches
cd "$REPO_DIR/robot_setup/lerobot_src"
pip install -e '.[dataset,training,async]' -q
pip install 'av>=15.0.0,<16.0.0' -q
cd "$REPO_DIR"
pip install pynput -q

# 5. HuggingFace login.
ensure_hf_token
if ! hf auth whoami &>/dev/null; then
    echo ""
    echo "Logging in to HuggingFace from QuicksetupScripts/.env..."
    hf auth login --token "$HF_TOKEN"
    echo "  ok HuggingFace login successful."
else
    echo "HuggingFace: already logged in, skipping."
fi

echo ""
echo "LeRobot setup complete."
