# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a group robotics RL project ("Project 3: Singulation") built on top of the **Robot Control Stack (RCS)** framework. The project trains policies for an SO101 robot arm to pick and place cubes into a bowl, with three evaluation scenarios of increasing difficulty. The primary working directory for simulation and policy rollout code is `robot-control-stack/`.

## Project Goals

- **Eval 1**: Pick a single cube and place it in a bowl (random cube position per episode).
- **Eval 2**: Pick a cube of a specified color from two cubes placed side-by-side.
- **Eval 3**: Pick a target cube from a 2×2 grid of four cubes (requires clutter handling).

The intended training approach is **Behavior Cloning + RL fine-tuning** (SAC recommended) with domain randomization to close the sim-to-real gap.

## Installation (robot-control-stack)

```shell
cd robot-control-stack
conda create -n rcs python=3.11
conda activate rcs
conda install conda-forge::glfw
pip install 'pip>=25.1'
pip install --group build_deps
pip install -ve .
```

RCS requires **Python 3.11** exactly — the `ompl` and `pyrealsense2` dependencies break on 3.12+.

## Build (C++ extension)

```shell
cd robot-control-stack
make gcccompile    # build _core pybind11 extension with GCC
make clangcompile  # alternative: build with Clang
```

The build output lands in `build/`. The compiled `_core` module is required for any `import rcs` call.

## Common Commands

From `robot-control-stack/`:

```shell
# Run all tests
make pytest
# or directly
pytest -vv  # tests are in python/tests/

# Lint Python
make pylint       # runs ruff + mypy
make ruff         # ruff only
make mypy         # mypy only

# Format Python
make pyformat     # isort + black in-place
make pycheckformat  # check without modifying

# Lint / format C++
make cpplint
make cppformat
```

Line length is 120 characters (enforced by ruff, black, and pylint).

## Policy Rollout (SO101 Eval Environments)

```shell
cd robot-control-stack/examples/so101

# Eval 1 — dry run with random policy (no checkpoint needed)
python policy_rollout.py --eval1 --random --n-rollouts 3

# Eval 1 — RSL-RL IsaacLab checkpoint
python policy_rollout.py --eval1 --policy ../../checkpts/policy.pt --n-rollouts 5

# Eval 1 — with GUI
python policy_rollout.py --eval1 --policy ../../checkpts/policy.pt --no-headless

# Eval 2 — fixed target color
python policy_rollout.py --eval2 --policy ../../checkpts/policy.pt --target-color blue

# Eval 3 — custom bowl position (x y z in robot frame, metres)
python policy_rollout.py --eval3 --policy ../../checkpts/policy.pt --bowl-xyz 0.35 0.15 0.003
```

## Architecture

### RCS Wrapper Stack

Environments are composed by stacking `gymnasium.Wrapper` layers, from innermost to outermost:

```
SimEnv (mujoco physics)
  └─ RobotWrapper       (IK, control mode)
       └─ GripperWrapper
            └─ RobotSimWrapper / GripperWrapperSim
                 └─ CameraSetWrapper (optional)
                      └─ RelativeActionSpace (optional)
                           └─ task-specific wrappers (e.g. PickObjSuccessWrapper, RandomSquareObjPos)
                                └─ JointVelWrapper  (SO101 only — injects joint_vel into obs)
                                     └─ CoverWrapper (exposes final unified obs/action space)
```

Each wrapper is added via `SimEnvCreator.create_env()` in `python/rcs/envs/scenes.py`. Task-specific wrappers are added by `Task.add_task_env()` in `python/rcs/envs/tasks.py`.

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `SimEnvCreator` | `envs/scenes.py` | Abstract factory; subclass to define a scene |
| `SimEnvCreatorConfig` | `envs/scenes.py` | Dataclass holding all scene configuration |
| `EmptyWorldSO101` | `envs/configs.py` | Base SO101 scene (joint control, no camera by default) |
| `SO101Eval1/2/3` | `envs/configs.py` | Project evaluation environments |
| `PickTask` / `PickTaskConfig` | `envs/tasks.py` | Eval 1 task: random cube placement + pick reward |
| `JointVelWrapper` | `envs/tasks.py` | Adds `joint_vel` to SO101 observations |
| `CubeColorWrapper` | `envs/tasks.py` | Randomizes cube geom color on reset |
| `MultiCubeColorWrapper` | `envs/tasks.py` | Randomizes multiple cube colors |
| `SO101JointPolicy` | `examples/so101/policy_rollout.py` | RSL-RL IsaacLab checkpoint adapter |

### Gymnasium Registration

Environments are registered in `envs/configs.py` bottom section:
- `rcs/so101`, `rcs/so101_eval1`, `rcs/so101_eval2`, `rcs/so101_eval3`
- `rcs/fr3`, `rcs/duo`, `rcs/ur5e`, `rcs/xarm7`

Instantiate with a custom config by passing `cfg=` to `gym.make()`:
```python
env = gym.make("rcs/so101_eval1", cfg=cfg, disable_env_checker=True)
```

### SO101 Observation / Action Space

Documented in full in `robot-control-stack/so101_obs_action_spaces.md`.

**Key observations** (inside `obs["robot"]`):
- `joints` (5,) float64 — arm joint angles (j1–j5) in radians
- `joint_vel` (6,) float64 — joint velocities (injected by `JointVelWrapper`)
- `tquat` (7,) float64 — TCP pose `[x, y, z, qx, qy, qz, qw]` (xyzw quaternion)
- `gripper` (1,) float32 — last gripper command (0.0=closed, 1.0=open)

**Action format** (joint control mode, used by `SO101JointPolicy`):
```python
{"robot": {"joints": np.ndarray(5,), "gripper": np.ndarray(1,)}}
```

**IsaacLab policy observation vector** (27-dim) expected by `SO101JointPolicy`:
```
joint_pos_rel(6) | joint_vel(6) | object_pos(3) | initial_object_pos(3) | bowl_pos(3) | last_action(6)
```

### Coordinate Frame

All poses are in the **shared base frame** (= SO101 robot-base origin). The robot base sits at MuJoCo world `z = −0.03 m`, so the shared frame is 3 cm above the MuJoCo floor. Bowl and object coordinates from TAs are given in this frame.

### Assets

Assets live under `robot-control-stack/assets/`:
- `robots/so101/` — SO101 MJCF model
- `objects/` — colored cubes (`red_cube`, `blue_cube`, etc.) and `bowl`
- `cameras/` — D405 RealSense camera MJCF fragment

The `RCS_PREFIX` env var overrides the asset root; by default it resolves to the `robot-control-stack/` directory.

### Trained Checkpoint

`robot-control-stack/checkpts/policy.pt` — first IsaacLab RSL-RL TorchScript checkpoint for Eval 1.

## Extensions

Hardware extensions (FR3, xArm7, UR5e, SO101, RealSense, ZED, etc.) live in `robot-control-stack/extensions/` and are installed separately:
```shell
pip install -ve extensions/rcs_so101
```

## Conventions

- Quaternions: **xyzw** order (`[qx, qy, qz, qw]`) throughout RCS.
- All spatial units are **metres** and **radians**.
- `rcs.common.Pose` is the canonical pose type: `translation` (3,) + `quaternion` (4, xyzw).
- Gripper semantics: `0.0` = close/grasp, `1.0` = open (binary, rounded internally).
