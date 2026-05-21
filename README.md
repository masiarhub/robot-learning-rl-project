<h1 align="center">Project 3: Reinforcement Learning</h1>
<h3 align="center">SO-ARM101 Singulation via RL in Isaac Lab</h3>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Isaac%20Lab-2.3.0-76b900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/RSL--RL-PPO-orange" />
</p>


<p align="center">
  Three singulation tasks of increasing complexity solved with PPO on the <b>SO-ARM101</b> 6-DoF arm in Isaac Lab.<br>
  Our primary approach — <b>VisualCoord PPO</b> — trains with analytic (u, v) pixel coordinates instead of a real camera sensor,<br>
  bridging the sim-to-real gap with zero rendering overhead and 4096 parallel environments.
</p>


## Results

<table>
  <tr>
    <td align="center"><b>Task 1 &mdash; Pick &amp; Place</b></td>
    <td align="center"><b>Task 2 &mdash; 2 Cube Singulation</b></td>
    <td align="center"><b>Task 3.1 &mdash; 4 Cube Singulation</b></td>
  </tr>
  <tr>
    <td>
      <img src="Docs/Media/cropped_videos/task_1_visual_part_one_portrait_cropped.gif" width="100%" />
    </td>
    <td>
      <img src="Docs/Media/cropped_videos/task_2_vsiual_part_one_portrait_cropped.gif" width="100%" />
    </td>
    <td>
      <img src="Docs/Media/cropped_videos/task_3_visual_part_one_portrait_cropped.gif" width="100%" />
    </td>
  </tr>
  <tr>
    <td align="center"><b>Task 3.2 &mdash; 3 Cube Sequential Pick</b></td>
    <td align="center"><b>Task 3.3 &mdash; 2 Cube Sequential Pick</b></td>
    <td align="center"><b>Task 3 Bonus &mdash; Stack Singulation</b></td>
  </tr>
  <tr>
    <td>
      <img src="Docs/Media/cropped_videos/task_3_visual_part_two_portrait_cropped.gif" width="100%" />
    </td>
    <td>
      <img src="Docs/Media/cropped_videos/task_3_visual_part_three_portrait_cropped.gif" width="100%" />
    </td>
    <td>
      <img src="Docs/Media/cropped_videos/task_3_visual_bonus_portrait_cropped.gif" width="100%" />
    </td>
  </tr>
</table>

---

## Overview

| Task | Scene | Goal | Primary env |
|------|-------|------|-------------|
| **Task 1** | 1 cube (random color), 1 bowl | Pick the cube and place it into the bowl | `Isaac-SO-ARM101-Task-One-VisualCoord-v0` |
| **Task 2** | 2 cubes (1 target + 1 distractor), 1 bowl | Pick the target color, ignore distractor | `Isaac-SO-ARM101-Task-Two-VisualCoord-v0` |
| **Task 3** | 4 cubes (fixed colors), 1 bowl | Sequentially singulate all cubes into the bowl | `Isaac-SO-ARM101-Task-Three-VisualCoord-v0` |
| **Bonus** | 3 cubes stacked vertically | Unstack the pile into a graspable configuration | `Isaac-SO-ARM101-Task-Bonus-VisualCoord-v0` |

---

## Method

We went through two earlier approaches before arriving at our final method. **VisualCoord PPO** is our current and primary approach; Teacher-Student Distillation and Direct Camera PPO are what we tried first.

### VisualCoord PPO *(primary)*

> **Key idea:** replace the wrist camera sensor with an analytic projection of the cube's world position onto the image plane, computed from forward kinematics and the known camera-to-gripper offset.

The actor receives a 3-vector `(u, v, visible)` — normalized pixel coordinates plus a binary in-FOV flag — instead of a raw RGB image. No `TiledCamera` sensor is instantiated, so the environment runs at full state-based speed and supports **4096 parallel envs** on a single GPU.

At deployment the identical `(u, v, visible)` signal is produced by a simple **HSV colour-segmentation** step on the real wrist camera, making the policy directly transferable without any fine-tuning.

```
Actor  (33 dims): joint_pos(6) · joint_vel(6) · ee_pos(3) · bowl_pos(3)
                  · cube_img(3) · color_one_hot(6) · actions(6)
Critic (27 dims): joint_pos(6) · joint_vel(6) · ee_pos(3) · obj_pos(3)
                  · bowl_pos(3) · actions(6)               ← privileged, not deployed
```

A **visibility reward** continuously incentivises the policy to keep the target cube inside the camera FOV, ensuring a reliable `(u, v)` signal throughout the episode. An asymmetric actor-critic separates deployed observations (no cube position) from privileged critic observations (current cube 3-D position), enabling the critic to guide learning without leaking privileged info to the deployed actor.

Domain randomisation covers table friction, gripper friction, and object mass to narrow the sim-to-real gap.

**Key files:**

| File | Description |
|------|-------------|
| `tasks/task_1/task_one_visual_coord_env_cfg.py` | Task 1 VisualCoord env — 1 cube, 6-class color palette |
| `tasks/task_2/task_two_visual_coord_env_cfg.py` | Task 2 VisualCoord env — 2 cubes (target + distractor) |
| `tasks/task_3/task_three_visual_coord_env_cfg.py` | Task 3 VisualCoord env — 4 cubes, 4-class one-hot |
| `tasks/task_3_bonus/task_bonus_visual_coord_env_cfg.py` | Bonus VisualCoord env — 3-cube stacked singulation |
| `tasks/task_1/mdp/` · `tasks/task_2/mdp/` · `tasks/task_3/mdp/` | Reward, observation, and event term implementations |

---

### Teacher-Student Distillation *(earlier approach)*

Our first attempt used a privileged **teacher** policy (full cube position in observations, no camera) trained to convergence with PPO. A **student** policy — equipped with a ResNet-18 wrist-camera encoder — then distilled the teacher via behaviour cloning loss while continuing to receive environment rewards (DAgger-style). We ultimately moved away from this because of the training complexity and the overhead of running a camera sensor across many parallel envs.

```
Phase 1a  Teacher PPO    — privileged full state, 4096 envs, fast convergence
Phase 2   Distillation   — camera student imitates teacher logits + env reward
Phase 3   Post-train RL  — RL fine-tune of the distilled student
```

**Key files:**

| File | Description |
|------|-------------|
| `tasks/task_1/task_one_teacher_env_cfg.py` | Teacher env (full privileged state) |
| `tasks/task_1/task_one_distill_env_cfg.py` | Distillation env (camera student + teacher) |
| `tasks/task_1/task_one_post_train_env_cfg.py` | Post-training RL fine-tune env |
| *(same pattern for `task_2/`, `task_3/`)* | |

---

### Direct Camera PPO *(earlier approach)*

We also tried training end-to-end with PPO directly from raw wrist-camera images using a ResNet-18 encoder, with an asymmetric actor-critic (camera actor, privileged-state critic). While conceptually simpler than distillation, the camera sensor bottleneck limited us to far fewer parallel environments, making training slow and less stable compared to VisualCoord. Additionaly, with more environments the TiledCameraCfg rendered worse frames, making the policy unsuitable for sim-to-real deployment.

**Key files:** `tasks/task_*/task_*_cam_ppo_env_cfg.py`

---

## Repository Structure

```
robot-learning-rl-project/
├── isaac_so_arm101/
│   └── src/isaac_so_arm101/
│       ├── tasks/
│       │   ├── task_1/                          # Task 1 — single cube pick & place
│       │   │   ├── task_one_visual_coord_env_cfg.py   ★ primary
│       │   │   ├── task_one_teacher_env_cfg.py
│       │   │   ├── task_one_distill_env_cfg.py
│       │   │   ├── task_one_cam_ppo_env_cfg.py
│       │   │   ├── task_one_post_train_env_cfg.py
│       │   │   ├── task_one_env_cfg.py                # base config
│       │   │   ├── joint_pos_env_cfg.py               # wraps all variants for gym
│       │   │   ├── mdp/                               # rewards, observations, events
│       │   │   ├── agents/rsl_rl_ppo_cfg.py           # PPO hyperparameters
│       │   │   └── __init__.py                        # gym.register calls
│       │   │
│       │   ├── task_2/                          # Task 2 — target color + distractor
│       │   │   ├── task_two_visual_coord_env_cfg.py   ★ primary
│       │   │   └── ...                                # same structure as task_1
│       │   │
│       │   ├── task_3/                          # Task 3 — 4-cube singulation
│       │   │   ├── task_three_visual_coord_env_cfg.py ★ primary
│       │   │   └── ...
│       │   │
│       │   └── task_3_bonus/                    # Bonus — vertical stack singulation
│       │       └── task_bonus_visual_coord_env_cfg.py ★ primary
│       │
│       ├── robots/
│       │   ├── trs_so100/                       # SO-ARM100 URDF + ArticulationCfg
│       │   └── trs_so101/                       # SO-ARM101 URDF + ArticulationCfg
│       └── scripts/
│           ├── rsl_rl/
│           │   ├── train.py                     # training entry point
│           │   └── play.py                      # policy rollout / evaluation
│           └── list_envs.py
│
├── Results/
│   ├── task_1/                                  # checkpoints and videos
│   ├── task_2/
│   ├── task_3/
│   └── RESULTS.md                               # play commands for each checkpoint
│
├── Docs/
│   ├── Media/                                   # result videos
│   └── IsaacLab/INSTALL.md
└── Sim-to-Real/                                 # sim-to-real transfer notes
```

---

## Installation

**Prerequisites:** CUDA 12.x · Isaac Sim 5.1 · Python 3.11

```bash
# 1. Clone the repo and enter it
git clone <repo-url>
cd robot-learning-rl-project

# 2. Activate the shared virtual environment
source ~/robot_learning/env_isaaclab/bin/activate

# 3. Install the custom extension in editable mode
cd isaac_so_arm101
python -m pip install -e .
```

> Full setup guide including Isaac Lab installation: [`Docs/IsaacLab/INSTALL.md`](Docs/IsaacLab/INSTALL.md)

---

## Training

All commands are run from `isaac_so_arm101/` with the venv active.

### VisualCoord PPO (recommended)

```bash
# Task 1
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-One-VisualCoord-v0 \
  --num_envs 4096 --headless

# Task 2
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-Two-VisualCoord-v0 \
  --num_envs 4096 --headless

# Task 3
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-Three-VisualCoord-v0 \
  --num_envs 4096 --headless
```

### Teacher-Student Distillation

```bash
# Phase 1a — train teacher
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-One-Teacher-v0 --num_envs 4096 --headless

# Phase 2 — distill into camera student
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-One-Distill-v0 --num_envs 64 --headless

# Phase 3 — RL fine-tune (optional)
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
  --task Isaac-SO-ARM101-Task-One-PostTrain-v0 --num_envs 64 --headless
```

---

## Evaluation

```bash
# Task 1 VisualCoord (primary submission checkpoint)
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-One-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint ../Results/task_1/task_1_visual_part_one/visual_general_model_4999.pt

# Task 2 — with 1 distractor
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Two-PartOne-Play-v0 \
  --num_envs 4 \
  --checkpoint ../Results/task_2/task_2_visual_part_one/visual_general_model_4999.pt

# Task 3 — 4-cube singulation
python src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint ../Results/task_3/task_3_visual_part_one/model_5600.pt
```

> All play commands with video recording flags are documented in [`Results/RESULTS.md`](Results/RESULTS.md).

---

## Registered Environments

<details>
<summary>Click to expand full environment table</summary>

| Task ID | Description |
|---------|-------------|
| `Isaac-SO-ARM101-Task-One-VisualCoord-v0` | **Task 1 primary** — analytic (u,v) projection, 4096 envs |
| `Isaac-SO-ARM101-Task-One-VisualCoord-Play-v0` | Task 1 primary (eval) |
| `Isaac-SO-ARM101-Task-One-Teacher-v0` | Task 1 Phase 1a — privileged teacher |
| `Isaac-SO-ARM101-Task-One-Distill-v0` | Task 1 Phase 2 — camera student distillation |
| `Isaac-SO-ARM101-Task-One-PostTrain-v0` | Task 1 Phase 3 — RL fine-tune of student |
| `Isaac-SO-ARM101-Task-One-CamPPO-v0` | Task 1 Alt — direct camera PPO, asymmetric AC |
| `Isaac-SO-ARM101-Task-Two-VisualCoord-v0` | **Task 2 primary** — analytic (u,v), 4096 envs |
| `Isaac-SO-ARM101-Task-Two-PartOne-Play-v0` | Task 2 Part 1 eval — Task-1 policy, 2-cube scene |
| `Isaac-SO-ARM101-Task-Three-VisualCoord-v0` | **Task 3 primary** — analytic (u,v), 4 fixed-color cubes |
| `Isaac-SO-ARM101-Task-Three-VisualCoord-Play-v0` | Task 3 primary (eval) |
| `Isaac-SO-ARM101-Task-Three-PartTwo-Play-v0` | Task 3 Part 2 eval — sequential pick from 3-cube scene |
| `Isaac-SO-ARM101-Task-Three-PartThree-Play-v0` | Task 3 Part 3 eval — pick with distractor |
| `Isaac-SO-ARM101-Task-Three-Teacher-v0` | Task 3 Phase 1a — privileged teacher |
| `Isaac-SO-ARM101-Task-Three-Distill-v0` | Task 3 Phase 2 — camera student distillation |
| `Isaac-SO-ARM101-Task-Three-PostTrain-v0` | Task 3 Phase 3 — RL fine-tune of student |
| `Isaac-SO-ARM101-Task-Three-CamPPO-v0` | Task 3 Alt — direct camera PPO, asymmetric AC |
| `Isaac-SO-ARM101-Task-Bonus-VisualCoord-v0` | **Bonus** — 3-cube vertical stack singulation |
| `Isaac-SO-ARM101-Lift-Cube-v0` / `-Play-v0` | Baseline lift task (SO-ARM101) |
| `Isaac-SO-ARM100-Lift-Cube-v0` / `-Play-v0` | Baseline lift task (SO-ARM100) |
| `Isaac-SO-ARM10*-Reach-v0` / `-Play-v0` | Baseline reach tasks |
| `Isaac-SO-ARM10*-PickPlace-v0` / `-Play-v0` | Pick & place baseline (randomised bowl) |

</details>

---

## Acknowledgements
This project was organized by the Robot Learning course at ETH Zurich. Thanks to the course organizers for their guidance in this project.

The code was developed on top of [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) by Muammer Bay (LycheeAI) and Louis Le Lay, and the [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) framework by NVIDIA. Training uses [RSL-RL](https://github.com/leggedrobotics/rsl_rl) (PPO implementation by Legged Robotics, ETH Zürich).
