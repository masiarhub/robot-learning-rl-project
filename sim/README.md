# `sim/` — ManiSkill3 / SO-101 simulation track

This directory houses the **simulation training track** for Project 3.
It is adapted from the open-source [Squint](https://github.com/aalmuzairee/squint)
repository (MIT-licensed, source commit recorded in `THIRD_PARTY.md`) and adds
three project-specific envs that match the TA-supplied evaluation protocol:

| Eval | Env id | What it does |
|---|---|---|
| 1 | `SO101PlaceBowlCube-v1` / `*Fixed-v1` | Single block → single bowl, goal-conditioned on bowl XY |
| 2 | `SO101TargetedPlace-v1` / `*Fixed-v1` | 2-block clutter, goal-conditioned on (color one-hot, bowl XY) |
| 3 | `SO101MultiBlockSeq-v1` | 4 blocks, 3 fixed bowls, 3-step sequence |

The trainer in `train_sim.py` is the squint visual-SAC trainer (1024 parallel
GPU envs, distributional C51 critic, torch.compile + CudaGraphs); we did not
modify it.

## Quick start

Create the conda env (separate from the project's `lerobot` env to avoid the
known mani-skill / numpy pin conflict):

```bash
conda env create -f environment.yaml
conda activate squint     # name kept from upstream
```

Then run the eval-specific configs:

```bash
# Eval 1 — single pick-and-place (BC-allowed; this is the RL fallback)
python train_sim.py --env_id=SO101PlaceBowlCube-v1 --total_timesteps=1500000

# Eval 2 — targeted pick-and-place (RL required)
python train_sim.py --env_id=SO101TargetedPlace-v1 --total_timesteps=2000000

# Eval 3 — Option B (recommended): reuse the Eval 2 checkpoint via the
# sequential runner; no extra training.
python deploy_sim_eval.py --eval=3 \
    --checkpoint=runs/eval2_targeted/ckpt.pt \
    --bowl_positions 0.32 -0.10  0.32 0.00  0.32 0.10 \
    --sequence 0 1 2
```

## Deployment to the real robot

`deploy_sim.py` is squint's deploy script (kept verbatim; talks to the real
SO-101 via `LeRobotRealAgent` from `deploy_utils/manipulator.py`). The thin
wrapper `deploy_sim_eval.py` routes per-eval kwargs (target color, bowl XY)
into the deploy script.

Before deploying, run the camera-alignment tool once per setup:

```bash
python deploy_utils/tune_camera.py
```

…and copy the printed wrist-camera parameters into
`envs/base_random_env.py` under `WristCameraEnv.WRIST_CAMERA_BASE_*`.

## Sanity checks

```bash
# Visualize all envs (cv2 window)
python examples/visualize_sim.py

# 4-waypoint scripted controller — should solve Eval 1 most of the time
python -m scripts.script_pickplace

# Smoke-test env registration + step
pytest tests/ -q
```

## Why this lives in its own folder

- The lerobot fork (`robot_setup/lerobot_src/`) already contains an SAC
  implementation, but it is the **HILSerl** distributed actor/learner setup
  designed for a single real-robot env with human interventions. It does not
  support GPU-vectorized 1024-env rollouts that we need for Eval 2 and Eval 3.
- The squint trainer is highly tuned (16×16 obs, C51, torch.compile,
  CudaGraphs) and converges in minutes per task — porting it into lerobot's
  policy/processor framework would consume days. We instead keep the two
  trainers as separate tracks and use **lerobot's deploy interface only at
  inference time**, via squint's `Sim2RealEnv` + `LeRobotRealAgent`.

## Files added by this branch

- `envs/place_bowl.py` — Eval 1 env (procedural bowl, goal-conditioned)
- `envs/targeted_pick_place.py` — Eval 2 env (2-block clutter)
- `envs/multi_block_eval.py` — Eval 3 env (4 blocks, 3-bowl sequence)
- `policies/sequential_runner.py` — Option B scheduler for Eval 3
- `deploy_sim_eval.py` — `--eval=1|2|3` deploy router
- `configs/eval{1,2,3}_*.yaml` — training and deploy configs
- `scripts/script_pickplace.py` — 4-waypoint scripted demo
- `scripts/seed_buffer_from_dataset.py` — replay LeRobotDataset → SAC buffer
- `tests/test_envs_register.py` — smoke test
