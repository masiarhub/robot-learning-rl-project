#!/usr/bin/env bash
# Eval 1 — Task 1: Pick & Place (VisualCoord PPO)
# Runs the trained VisualCoord policy: picks a randomly-colored cube and places it into the bowl.
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Activate virtual environment
# ---------------------------------------------------------------------------
if [ -z "$VIRTUAL_ENV" ]; then
  VENV="$REPO_ROOT/../env_isaaclab"
  if [ ! -f "$VENV/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $VENV"
    echo "Please follow Docs/IsaacLab/INSTALL.md to set up the environment first, then re-run this script."
    exit 1
  fi
  source "$VENV/bin/activate"
fi

# ---------------------------------------------------------------------------
# 2. Install the isaac_so_arm101 extension (editable, idempotent)
# ---------------------------------------------------------------------------
echo "Installing isaac_so_arm101 extension..."
pip install -e "$REPO_ROOT/isaac_so_arm101" --quiet

# ---------------------------------------------------------------------------
# 3. Run evaluation
# ---------------------------------------------------------------------------
echo ""
echo "=== Eval 1: Task One VisualCoord — Pick & Place ==="
echo "Checkpoint: Results/task_1/task_1_visual_part_one/visual_general_model_4999.pt"
echo ""

cd "$REPO_ROOT"
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-One-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_1/task_1_visual_part_one/visual_general_model_4999.pt
