#!/usr/bin/env bash
# Eval 2 — Task 2: Color-Conditioned Pick & Place with Distractor (VisualCoord PPO)
# Runs the Task-1 VisualCoord policy in a 2-cube scene (1 target + 1 distractor).
# The policy uses the color one-hot to pick the correct cube and ignore the other.
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
echo "=== Eval 2: Task Two Part 1 — Pick & Place with 1 Distractor ==="
echo "Checkpoint: Results/task_2/task_2_visual_part_one/visual_general_model_4999.pt"
echo ""

cd "$REPO_ROOT"
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Two-PartOne-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_2/task_2_visual_part_one/visual_general_model_4999.pt
