# SO101 Policy Rollout

Runs a trained policy in the SO101 MuJoCo simulation for Eval 1/2/3.

## Setup

From the repo root, activate the environment and go to the script directory:

```bash
conda activate rcs
cd sim-mujoco/examples/so101
```

## Running an Isaac Lab checkpoint

Isaac Lab policies (RSL-RL PPO, TorchScript `.pt`) use `--isaaclab`:

```bash
python policy_rollout.py --eval1 \
    --policy ../../checkpts/policy.pt \
    --n-rollouts 5 --headless --isaaclab
```

With the viewer (WSL: check Windows taskbar for the window):

```bash
python policy_rollout.py --eval1 \
    --policy ../../checkpts/policy.pt \
    --n-rollouts 1 --realtime --isaaclab
```

## Eval variants

| Flag | Task |
|------|------|
| `--eval1` | Pick single cube → place in bowl |
| `--eval2` | Pick target-colored cube from clutter |
| `--eval3` | Sequential multi-step pick and place |

## Common options

| Flag | Description |
|------|-------------|
| `--n-rollouts N` | Number of episodes (default: 5) |
| `--max-steps N` | Hard step limit per episode (default: 500) |
| `--headless` | No viewer, faster |
| `--realtime` | Throttle to real-time speed |
| `--isaaclab` | Use `IsaacLabJointPolicy` (27-dim obs, 6-dim joint action) |
| `--target-color COLOR` | Pin cube color: red/blue/green/yellow/orange/purple |
| `--bowl-xyz X Y Z` | Override bowl position in robot frame (meters) |
| `--device cuda:0` | Torch device for inference (default: cpu) |

## IsaacLabJointPolicy — tuning parameters

Defined as class-level constants at the top of `IsaacLabJointPolicy` in `policy_rollout.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `ARM_ACTION_SCALE` | `0.5` | Scales raw policy output to joint target offsets (rad). Lower = smaller steps. |
| `GRIPPER_ACTION_SCALE` | `0.3` | Same for gripper joint. |
| `ARM_SMOOTH_ALPHA` | `0.5` | Exponential smoothing on arm targets. `1.0` = no smoothing, `0.3` = very smooth/slow. Compensates for MuJoCo vs Isaac Lab physics timestep mismatch. |
| `BOWL_HOVER_HEIGHT` | `0.12` | Z offset added to bowl position in the obs (meters). Should match Isaac Lab training value. |
| `EE_LOCAL_OFFSET` | `[0.01, 0, -0.09]` | Local offset from `gripper_body` to end-effector tip (meters). |

## Observation vector (27-dim, Isaac Lab format)

| Indices | Content |
|---------|---------|
| `[0:6]` | `joint_pos - default_pos` (5 arm + 1 gripper) |
| `[6:12]` | Joint velocities |
| `[12:15]` | EE position in robot root frame |
| `[15:18]` | Initial cube position at episode start (frozen) |
| `[18:21]` | Bowl position + hover offset in robot root frame |
| `[21:27]` | Last raw policy action |

## Sanity check with random policy

```bash
python policy_rollout.py --eval1 --random --n-rollouts 3 --headless
```

## Debug output

On the first step of the first rollout the script logs:
- Resolved MuJoCo body IDs for the robot, cube, and bowl
- Joint qpos addresses (confirms correct indexing into `mjData.qpos`)
- Full obs vector components so you can verify values are physically plausible
