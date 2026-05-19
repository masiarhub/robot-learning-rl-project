# Singulation with the SO-ARM101 - `isaac_so_arm101`

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docsthon.org/3/whatsnew/3.11.html)

## Environment

All commands below are run from `isaac_so_arm101/` with the venv active:

```bash
source ~/robot_learning/env_isaaclab/bin/activate
```

To disable WANDB for this terminal session:
```bash
export WANDB_MODE=disabled
```
---

## Task 1 — Pick and Place (cube → bowl)

Task 1 has three training pipelines.  Pick the one that fits your goal:

```
Pipeline A (recommended): Teacher → Distillation → Post-Train
Pipeline B (alternative): Direct camera PPO from scratch
Pipeline C (optional):    Initial-cube-state policy (no camera, no current cube pos)
```

### Pipeline A — Teacher → Distillation → Post-Train

#### Phase 1a — Train the teacher (full privileged state, no camera)

The teacher actor sees: joint positions, joint velocities, EE position (FK), current
cube position, initial cube position (frozen at reset), bowl target position, and last action.

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-Teacher-v0 \
    --num_envs 4096 --headless --video
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-One-Teacher-Play-v0 \
    --num_envs 50 --video \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

Resume a run:
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-Teacher-v0 \
    --num_envs 4096 --headless --video \
    --resume --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

#### Phase 2 — Distillation (teacher → camera student)

The student actor sees: joint positions, joint velocities, gripper link position,
bowl target position, and a frozen ResNet18 encoding of the wrist RGB image (512 dims).
The teacher receives the same 30-dim privileged observation it was trained with.

> **Important:** `--load_run` must point to a Phase 1a teacher checkpoint.
> The teacher input dimensions (30) must match the checkpoint; do not mix with
> Phase 1b (initial-cube-state) checkpoints.

**Train distillation:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-Distill-v0 \
    --num_envs 128 --headless --video --enable_cameras \
    --load_run <TEACHER_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

**Evaluate the distilled student:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-One-Distill-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <DISTILL_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

#### Phase 3 — Post-training RL fine-tune of the distilled student

Loads the distilled student checkpoint and continues training with PPO on the
same camera-based observation space, with stronger domain randomisation.

> **Important:** `--load_run` must point to a Phase 2 distillation checkpoint.

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-PostTrain-v0 \
    --num_envs 128 --video --headless --enable_cameras \
    --resume --load_run <DISTILL_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-One-PostTrain-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <POST_TRAIN_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

### Pipeline B — Direct camera PPO (no teacher, trained from scratch)

Actor sees camera + proprioception; critic sees privileged state (cube position).
No teacher or distillation involved.

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-CamPPO-v0 \
    --num_envs 128 --headless --video --enable_cameras
```

first create root/robot-learning, then:
```bash
echo "Yes" | nohup ./isaaclab.sh -p /workspace/robot-learning/robot-learning-rl-project/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-Task-One-CamPPO-v0 --num_envs 256 --headless --video --enable_cameras --max_iterations 10000 > output.txt 2>&1 &
```

```bash
docker cp isaac-lab-base:/workspace/isaaclab/logs/rsl_rl/task_1_cam_ppo/2026-05-19_08-42-03/videos/train ~/videos
```

```bash
ps aux | grep ./isaaclab.sh
kill <porcess-id>
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-One-CamPPO-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```


---

### Pipeline C — Initial-cube-state policy (no camera)

Actor sees: joint positions, joint velocities, EE position (FK), initial cube position
(frozen at reset, not current), bowl target position, and last action.  No camera,
no current cube tracking.

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-v0 \
    --num_envs 4096 --headless --video
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-One-Play-v0 \
    --num_envs 50 --video \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

## Task 2 — Color-conditioned Pick and Place (cube → bowl)

Same robot and scene as Task 1, but the target bowl colour is randomised each episode.
The three pipelines are structurally identical to Task 1; pick the one that fits your goal:

```
Pipeline A (recommended): Teacher → Distillation → Post-Train
Pipeline B (alternative): Direct camera PPO from scratch
Pipeline C (optional):    Initial-cube-state policy (no camera, no current cube pos)
```

### Pipeline A — Teacher → Distillation → Post-Train

#### Phase 1a — Train the teacher (full privileged state, no camera)

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-Teacher-v0 \
    --num_envs 4096 --headless --video
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-Two-Teacher-Play-v0 \
    --num_envs 50 --video \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

Resume a run:
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-Teacher-v0 \
    --num_envs 4096 --headless --video \
    --resume --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

#### Phase 2 — Distillation (teacher → camera student)

> **Important:** `--load_run` must point to a Phase 1a teacher checkpoint.

**Train distillation:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-Distill-v0 \
    --num_envs 256 --headless --video --enable_cameras \
    --load_run <TEACHER_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

**Evaluate the distilled student:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-Two-Distill-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <DISTILL_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

#### Phase 3 — Post-training RL fine-tune of the distilled student

> **Important:** `--load_run` must point to a Phase 2 distillation checkpoint.

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-PostTrain-v0 \
    --num_envs 128 --video --headless --enable_cameras \
    --resume --load_run <DISTILL_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-Two-PostTrain-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <POST_TRAIN_RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

### Pipeline B — Direct camera PPO (no teacher, trained from scratch)

Actor sees camera + proprioception; critic sees privileged state (cube position).

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-CamPPO-v0 \
    --num_envs 128 --headless --video --enable_cameras
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-Two-CamPPO-Play-v0 \
    --num_envs 10 --video --enable_cameras \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

### Pipeline C — Initial-cube-state policy (no camera)

**Train:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-Two-v0 \
    --num_envs 4096 --headless --video
```

**Evaluate:**
```bash
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
    --task Isaac-SO-ARM101-Task-Two-Play-v0 \
    --num_envs 50 --video \
    --load_run <RUN_TIMESTAMP> --checkpoint model_<N>.pt
```

---

## Other tasks — Lift and Pick-Place

**Train:**
```bash
# Lift
python src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-Lift-Cube-v0 --num_envs 4096 --headless

# Pick & Place
python src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-PickPlace-v0 --num_envs 4096 --headless --video
```

**Evaluate:**
```bash
# Lift
python src/isaac_so_arm101/scripts/rsl_rl/play.py --task Isaac-SO-ARM101-Lift-Cube-Play-v0 \
    --num_envs 4 --checkpoint logs/rsl_rl/lift/<DATE_TIME>/model_0.pt

# Pick & Place
python src/isaac_so_arm101/scripts/rsl_rl/play.py --task Isaac-SO-ARM101-PickPlace-Play-v0 \
    --num_envs 4 --checkpoint logs/rsl_rl/pick_place/<DATE_TIME>/model_0.pt
```

---

## Monitoring

**TensorBoard:**
```bash
# Task 1
tensorboard --logdir logs/rsl_rl/task_1_teacher_ppo/
tensorboard --logdir logs/rsl_rl/task_1_cam_ppo/
tensorboard --logdir logs/rsl_rl/task_1_distillation/
tensorboard --logdir logs/rsl_rl/task_1_post_train/

# Task 2
tensorboard --logdir logs/rsl_rl/task_2_teacher_ppo/
tensorboard --logdir logs/rsl_rl/task_2_cam_ppo/
tensorboard --logdir logs/rsl_rl/task_2_distillation/
tensorboard --logdir logs/rsl_rl/task_2_post_train/
```

---

## Docs

| File                       | Scope                     |
| -------------------------- | ------------------------- |
| [`README.md`](README.md)   | README - Project overview |
| [`INSTALL.md`](INSTALL.md) | INSTALL Instruction       |
| [`DOCKER.md`](DOCKER.md)   | DOCKER Install            |

## Credits

- Base env ported from [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101/tree/main)
