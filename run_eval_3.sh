#!/usr/bin/env bash
# Eval 3 — Task 3: Singulation (VisualCoord PPO)
# Runs four sequential evaluations covering all Task 3 parts plus the bonus.
# Close the simulation window to advance to the next part.
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

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 3a. Part 1 — 4-cube singulation (primary Task 3 result)
# ---------------------------------------------------------------------------
echo ""
echo "=== Eval 3 Part 1: Task Three VisualCoord — 4-Cube Singulation ==="
echo "Checkpoint: Results/task_3/task_3_visual_part_one/model_5600.pt"
echo "(Close the window to continue to Part 2)"
echo ""
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_one/model_5600.pt

# ---------------------------------------------------------------------------
# 3b. Part 2 — Task-1 policy picking sequentially from 3-cube scene
# ---------------------------------------------------------------------------
echo ""
echo "=== Eval 3 Part 2: Sequential Pick from 3-Cube Scene ==="
echo "Checkpoint: Results/task_3/task_3_visual_part_two/visual_general_model_4999.pt"
echo "(Close the window to continue to Part 3)"
echo ""
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-PartTwo-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_two/visual_general_model_4999.pt

# ---------------------------------------------------------------------------
# 3c. Part 3 — Task-1 policy with 1 distractor in a 2-cube scene
# ---------------------------------------------------------------------------
echo ""
echo "=== Eval 3 Part 3: Pick & Place with Distractor ==="
echo "Checkpoint: Results/task_3/task_3_visual_part_three/visual_general_model_4999.pt"
echo "(Close the window to continue to Bonus)"
echo ""
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-PartThree-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_three/visual_general_model_4999.pt

# ---------------------------------------------------------------------------
# 3d. Bonus — vertical stack singulation
# ---------------------------------------------------------------------------
echo ""
echo "=== Eval 3 Bonus: Stack Singulation (3 stacked cubes) ==="
echo "Checkpoint: Results/task_3/task_3_visual_bonus/model_500.pt"
echo ""
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Bonus-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_bonus/model_500.pt
