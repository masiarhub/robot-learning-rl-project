# robot-learning-rl-project
Group Project 3: Singulation — Reinforcement Learning

## Sim2Sim Policy Transfer (IsaacLab → MuJoCo)

This branch (`mujoco`) contains the MuJoCo simulation environment and tooling for running and evaluating IsaacLab-trained policies on the SO101 robot arm.

---

## Installation

```shell
cd sim-mujoco
conda create -n rcs python=3.11
conda activate rcs
conda install conda-forge::glfw
pip install 'pip>=25.1'
pip install --group build_deps
pip install -ve .
```

Build the C++ extension (required for `import rcs`):

```shell
make gcccompile   # or: make clangcompile
```

---

## Policy Rollout

All commands run from `sim-mujoco/examples/so101/`.

### Dry run (no checkpoint)

```shell
python policy_rollout.py --eval1 --random --n-rollouts 3
```

### Standard (no-camera) policies

Checkpoint auto-detection picks the right obs config from the filename. Pass `--no-headless` to watch the rollout in the viewer.

```shell
# Eval 1 — single cube → bowl
python policy_rollout.py --eval1 --policy ../../checkpts/policy_no_camera_20260519.pt

# Eval 1 — pin cube color
python policy_rollout.py --eval1 --policy ../../checkpts/policy_no_camera_20260519.pt --target-color blue

# Eval 1 — override bowl position (x y z in robot frame, metres)
python policy_rollout.py --eval1 --policy ../../checkpts/policy_no_camera_20260519.pt --bowl-xyz 0.35 0.15 0.003

# Eval 2 — targeted pick (two cubes side-by-side)
python policy_rollout.py --eval2 --policy ../../checkpts/policy_no_camera_20260519.pt

# Eval 3 — 2×2 grid of cubes
python policy_rollout.py --eval3 --policy ../../checkpts/policy_no_camera_20260519.pt

# With GUI viewer
python policy_rollout.py --eval1 --policy ../../checkpts/policy_no_camera_20260519.pt --no-headless --realtime
```

Available no-camera checkpoints in `sim-mujoco/checkpts/`:

| Checkpoint | Obs variant | Notes |
|---|---|---|
| `policy_no_camera_20260519.pt` | `no_camera` (27-dim) | Latest; matches IsaacLab training params |
| `policy_no_camera_binary_gripper.pt` | `no_camera` | Binary gripper variant |
| `policy_no_camera_continuous_gripper.pt` | `no_camera` | Continuous gripper |
| `policy_no_camera_only_init_cube_obs.pt` | `only_init_cube` | Uses initial cube pos only |

### Visual policy (wrist D405 camera)

The visual policy (`policy_visual_smooth.pt`) uses wrist camera image coordinates of the cube instead of 3-D ground-truth position. The `--camera` flag is enabled automatically when this checkpoint is detected.

```shell
# Eval 1 — visual policy (camera auto-enabled)
python policy_rollout.py --eval1 --policy ../../checkpts/policy_visual_smooth.pt

# Force camera on/off explicitly
python policy_rollout.py --eval1 --policy ../../checkpts/policy_no_camera_20260519.pt --camera
python policy_rollout.py --eval1 --policy ../../checkpts/policy_visual_smooth.pt --no-camera
```

---

## Camera Smoke Test

Verifies that the D405 wrist camera is mounted correctly and renders a valid image. Run from `sim-mujoco/`:

```shell
# Headless — saves PNG to examples/so101/camera_smoke_test_wrist.png
python examples/so101/camera_smoke_test.py

# With MuJoCo viewer for visual inspection
python examples/so101/camera_smoke_test.py --gui
```

---

## Policy Training Parameters

IsaacLab training configs used to produce each checkpoint live in `sim-mujoco/policy_params_isaaclab/`. Each policy has a paired `agent_<name>.yaml` (RSL-RL runner config) and `env_<name>.yaml` (IsaacLab env config).

---

## Key Source Files

| File | Purpose |
|---|---|
| `examples/so101/policy_rollout.py` | Main rollout script; IsaacLab checkpoint adapter, obs construction, eval loop |
| `examples/so101/camera_smoke_test.py` | Validate wrist camera placement and rendering |
| `python/rcs/envs/configs.py` | Scene configs for `EmptyWorldSO101` and Eval 1/2/3 environments |
| `python/rcs/envs/tasks.py` | Task wrappers: `PickTask`, `JointVelWrapper`, `CubeColorWrapper`, etc. |
| `so101_obs_action_spaces.md` | Full SO101 obs/action space documentation |
