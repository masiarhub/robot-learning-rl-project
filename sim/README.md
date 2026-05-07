# `sim/` — ManiSkill3 / SO-101 simulation track

This directory houses the **simulation training track** for Project 3.
It is adapted from the open-source [Squint](https://github.com/aalmuzairee/squint)
repository and [ManiSkill](https://maniskill.readthedocs.io/en/latest/index.html) (MIT-licensed, source commit recorded in `THIRD_PARTY.md`) and adds
three project-specific envs that match the TA-supplied evaluation protocol:

| Eval | Env id | What it does |
|---|---|---|
| 1 | `SO101PlaceBowlCube-v1` / `*Fixed-v1` | Single block → single bowl, goal-conditioned on bowl XY |
| 2 | `SO101TargetedPlace-v1` / `*Fixed-v1` | 2-block clutter, goal-conditioned on (color one-hot, bowl XY) |
| 3 | `SO101MultiBlockSeq-v1` | 4 blocks, 3 fixed bowls, 3-step sequence |

The trainer in `train_sim.py` is the squint visual-SAC trainer (1024 parallel
GPU envs, distributional C51 critic, torch.compile + CudaGraphs); we did not
modify it.

## System requirements

| Component | Training (required) | Smoke test / headless |
|---|---|---|
| **GPU** | NVIDIA, CUDA 12.4+, ≥ 8 GB VRAM | not required |
| **Vulkan driver** | required (SAPIEN renderer) | not required |
| **RAM** | ≥ 16 GB | ≥ 8 GB |
| **OS** | Linux recommended; Windows supported | Linux / Windows |
| **Python** | 3.10 | 3.10 |
| **conda** | required | required |

> Training runs 1024 parallel GPU envs with `torch.compile` + CudaGraphs.
> A single RTX 3080 converges Eval 1 in ~5 min and Eval 2 in ~15 min.
> Without a CUDA-capable GPU, only the headless smoke test (physics only,
> no rendering) is supported — see [Smoke test](#smoke-test) below.

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

### Step 1 — configure your hardware (`deploy_utils/robot_config.py`)

Edit the constants at the top of `deploy_utils/robot_config.py` for your machine:

| Parameter | What to set |
|---|---|
| `port` | Serial port of the SO-101 arm — `/dev/ttyACM0` on Linux, `COM3` / `COM4` etc. on Windows |
| `index_or_path` | Webcam device — `/dev/video0` on Linux, integer index `0` / `1` on Windows |
| `width`, `height` | Physical camera capture resolution; must match what `tune_camera.py` was calibrated with |
| `id` | Must match the LeRobot calibration filename under `calibration_dir` |
| `calibration_dir` | Path to the directory containing your `.json` calibration file |

If you are using a RealSense camera instead of a webcam, uncomment the
`RealSenseCameraConfig` block and set `serial_number_or_name` to your device's serial.

If your SO-101 gripper's physical servo range differs from the one used during
development, adjust the mapping constants in `deploy_utils/manipulator.py`:

```python
self._gripper_servo_min = -62.5   # servo degrees at fully closed
self._gripper_servo_max =  64.62  # servo degrees at fully open
```

### Step 2 — align the wrist camera (`deploy_utils/tune_camera.py`)

Run the interactive alignment tool once per physical setup (it requires a GPU):

```bash
python deploy_utils/tune_camera.py
```

A side-by-side window shows **Real | Sim | Blended**. Use the trackbars to
adjust X / Y / Z position, Roll / Pitch / Yaw, and FOV until the sim overlay
matches the real camera view. Press `p` to print the tuned values, then copy
them into `envs/base_random_env.py`:

```python
# WristCameraEnv — base_random_env.py lines 497-499
WRIST_CAMERA_BASE_POS     = (-0.0049, 0.0498, -0.0591)          # metres
WRIST_CAMERA_BASE_ROT_RAD = (np.deg2rad(-90), np.deg2rad(91), np.deg2rad(-35.31))
WRIST_CAMERA_FOV          = np.deg2rad(71)
```

### Step 3 — run deployment

```bash
# Eval 1
python deploy_sim_eval.py --eval=1 --checkpoint=runs/eval1_place_bowl/ckpt.pt

# Eval 2
python deploy_sim_eval.py --eval=2 --checkpoint=runs/eval2_targeted/ckpt.pt

# Eval 3 — reuses Eval 2 checkpoint via sequential runner
python deploy_sim_eval.py --eval=3 \
    --checkpoint=runs/eval2_targeted/ckpt.pt \
    --bowl_positions 0.32 -0.10  0.32 0.00  0.32 0.10 \
    --sequence 0 1 2
```

## Closing the sim-to-real gap

All domain randomization knobs live in the `RandomizationConfig` dataclass in
`envs/base_random_env.py`. Pass any subset as a dict to `gym.make` via
`domain_randomization_config={...}`, or edit the dataclass defaults directly.

### Background / table appearance

The single highest-impact change is replacing the default black background with
a real photo of your table and workspace:

```python
rgb_overlay_path = "path/to/your_table_photo.png"  # RandomizationConfig
```

The overlay is composited behind the robot and task objects using the
segmentation mask (greenscreen technique), so only the background pixels change.
Set `apply_overlay=False` to disable compositing entirely and use raw sim images.

### Lighting

```python
randomize_lighting = True   # randomises ambient intensity each episode
```

The directional lights are fixed in `base_random_env.py:_load_lighting`. Edit
their direction vectors and colour tuples there to match your real lighting rig.

### Robot appearance

```python
robot_color = [0.8, 0.8, 0.8]   # fixed RGB (0–1) to match real robot paint
robot_color = "random"           # per-episode random colour for robustness
```

### Camera noise (wrist camera)

| Parameter | Default | Effect |
|---|---|---|
| `wrist_camera_pos_noise` | `(2 mm, 2 mm, 2 mm)` | Per-step position jitter relative to gripper |
| `wrist_camera_rot_noise` | `(1°, 1°, 1°)` | Per-step roll/pitch/yaw jitter |
| `wrist_camera_fov_noise` | `1°` | Per-step FOV jitter around the base 71° |

Widen these ranges if your camera mount has mechanical play; narrow them if
the mount is rigid and the gap is dominated by other factors.

### Gripper dynamics

```python
gripper_stiffness_range = (500, 2000)   # N·m/rad
gripper_damping_range   = (50,  200)    # N·m·s/rad
```

Narrow these ranges toward the values measured on your real servo if you can
characterise it; wider ranges improve robustness at the cost of training time.

### Observation resolution

Obs resolution is set per-eval in `configs/eval{1,2,3}_*.yaml`:

```yaml
render_size: 128   # internal sim render resolution before downsampling
image_size:  16    # final obs fed to the CNN encoder
```

Increasing `image_size` gives more visual detail but requires re-training;
`render_size` only affects rendering cost during training.

## Smoke test

Run this before training to confirm env registration, physics, and (optionally)
rendering all work correctly.

**GPU machine** — full RGB visualization, cv2 window per task:

```bash
cd sim
python examples/visualize_sim.py
```

**CPU-only / no Vulkan** — headless physics-only test, no rendering required:

```bash
cd sim
python examples/visualize_sim.py --headless
```

The headless mode tests four tasks (`SO101ReachCube-v1`, `SO101LiftCube-v1`,
`SO101PlaceBowlCube-v1`, `SO101TargetedPlace-v1`), runs 20 steps each, prints
per-step rewards, and reports a pass/fail summary. It bypasses all Vulkan/SAPIEN
camera calls so it runs on any machine.

Expected output (headless):

```
[SO101ReachCube-v1] instantiating...
  step 01/20  reward=0.0000  done=False
  ...
[SO101ReachCube-v1] PASSED
...
==================================================
Results: 4 passed, 0 failed
```

If you see `vk::Queue::submit: ErrorDeviceLost` when running without `--headless`,
your Vulkan driver is not functional — use `--headless` instead.

## Other sanity checks

```bash
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

## Modifying environment shapes

### A. Geometric primitives — the 6-step pipeline

Every shape (cube, can, bowl segment) is built inside `_load_scene()` using
this sequence, which must be followed exactly for batched GPU simulation:

```
1. Sample per-env geometry/physics params  →  numpy arrays of shape (num_envs,)
2. Loop over envs: builder = self.scene.create_actor_builder()
3. Add collision:  builder.add_box_collision / add_cylinder_collision
4. Add visual:     builder.add_box_visual / add_cylinder_visual / add_sphere_visual
5. Set pose + assign env: builder.initial_pose = ...; builder.set_scene_idxs([i])
6. Build + merge: builder.build() → append → Actor.merge(list, name=...)
```

**Primitive reference:**

| Shape | Collision method | Visual method |
|---|---|---|
| Box | `add_box_collision(half_size=[x,y,z], material=..., density=...)` | `add_box_visual(half_size=[x,y,z], material=...)` |
| Cylinder | `add_cylinder_collision(radius=..., half_length=..., material=..., density=..., pose=...)` | `add_cylinder_visual(...)` |
| Sphere | — (not used for collision) | `add_sphere_visual(radius=..., material=...)` |

Cylinder default axis is X; rotate 90° around Y to stand it upright:
`pose=sapien.Pose(q=euler2quat(0, np.pi/2, 0))` — see `envs/place_bowl.py:255`.

**Build method semantics:**

| Method | Physics | Gravity | Moveable | When to use |
|---|---|---|---|---|
| `build()` | Yes | Yes | Yes | Graspable objects (cubes, cans) |
| `build_static()` | Yes | No | No | Fixed objects (bowl, walls) |
| `build_kinematic()` | No | No | Programmatically | Goal markers, camera mounts |

**Materials:**
```python
# Physics (friction, restitution)
sapien.pysapien.physx.PhysxMaterial(static_friction, dynamic_friction, restitution)

# Visual (colour, RGBA 0–1)
sapien.render.RenderMaterial(base_color=[R, G, B, A])
```

**Per-env geometry variation:** sample with
`self._batched_episode_rng.uniform(lo, hi)` → numpy array of shape `(num_envs,)`;
extract per-env scalar inside the loop via `float(array[i])`.

The bowl is procedurally generated (cylinder floor + ring of 16 box rim segments)
so its radius and height can vary per env without loading any mesh file — see
`envs/place_bowl.py:247–286` for the full pattern.

**Table colour:** override post-build via `RenderBodyComponent.render_shapes`
(see `envs/place_bowl.py:170–175`). Project spec colour `#B8ADA9` translates to
`(0xB8/255, 0xAD/255, 0xA9/255)`.

### B. Custom CAD mesh objects

To replace a procedural shape with a mesh drawn in AutoCAD, Fusion 360, Blender, etc.:

**Step 1 — Export from CAD**

Export the visual mesh as **STL** (binary or ASCII), **OBJ**, or **GLB/GLTF**.
SAPIEN loads it only for rendering so it can be arbitrarily detailed.
USD/USDA/USDC/USDZ are supported too — SAPIEN auto-converts them to GLB at load time.

**Step 2 — Generate collision geometry**

Physics engines require convex (or convex-decomposed) geometry. Three options:

| Option | When to use | How |
|---|---|---|
| Single convex hull | Simple convex shapes | Import STL into Blender → Convex Hull modifier → export as `_convex.obj` |
| Multiple pre-convexified parts | Complex shapes (bowl cavity, handle) | Decompose manually in Blender → export each part → use `decomposition="none"` |
| Automatic CoACD at runtime | Complex shapes, no Blender access | Feed original STL to SAPIEN → use `decomposition="coacd"` (slow on first load) |

**Step 3 — Place mesh files**

Drop files into `sim/envs/robot/meshes/` alongside the existing SO-101 meshes,
or create a new subdirectory.

**Step 4 — Load in `_load_scene()`**

```python
import sapien

bowls = []
for i in range(self.num_envs):
    builder = self.scene.create_actor_builder()

    # Visual — full-detail CAD mesh, rendered only
    builder.add_visual_from_file(
        filename="envs/robot/meshes/my_bowl.stl",
        pose=sapien.Pose(),                 # local offset relative to actor origin
        scale=(1.0, 1.0, 1.0),
        material=sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0]),
    )

    # Collision — option A: single convex hull (fast, approximate)
    builder.add_convex_collision_from_file(
        filename="envs/robot/meshes/my_bowl_convex.obj",
        pose=sapien.Pose(),
        scale=(1.0, 1.0, 1.0),
        material=sapien.pysapien.physx.PhysxMaterial(0.5, 0.5, 0.0),
        density=200.0,
    )

    # Collision — option B: multiple pre-convexified parts (accurate, no runtime cost)
    # builder.add_multiple_convex_collisions_from_file(
    #     filename="envs/robot/meshes/my_bowl_parts.obj",
    #     decomposition="none",
    # )

    # Collision — option C: automatic CoACD decomposition at runtime
    # builder.add_multiple_convex_collisions_from_file(
    #     filename="envs/robot/meshes/my_bowl.stl",
    #     decomposition="coacd",
    # )

    builder.initial_pose = sapien.Pose(p=[bowl_x, bowl_y, 0.0])
    builder.set_scene_idxs([i])
    bowl = builder.build_static(name=f"bowl-{i}")
    bowls.append(bowl)

self.bowl = Actor.merge(bowls, name="bowl")
self.add_to_state_dict_registry(self.bowl)
```

**Existing mesh assets:** `deploy_utils/blender_stls/` contains `bin.stl`,
`can.stl`, `cube.stl`, and `large_cube.stl` — ready-made task objects not yet
wired into any environment. They can be loaded with the same pattern above.

**Robot geometry** is declared in `envs/robot/so101.urdf`, which pairs a visual
STL with a pre-convexified `_convex.obj` per link (see `envs/robot/meshes/`).
To change the robot's appearance replace the STL; to change its collision
geometry replace the matching `_convex.obj`.

## SAC agent internals

### Where the code lives

All network classes and the training loop are in `train_sim.py`. At deploy time
only the encoder and actor are needed; they are wrapped by `DeployAgent` which
is what `deploy_sim.py` loads from the checkpoint.

| Class | File | Role |
|---|---|---|
| `CNNEncoder` | `train_sim.py` | Shared visual backbone |
| `Actor` | `train_sim.py` | Policy π(a\|s) |
| `Critic` | `train_sim.py` | Distributional Q-function (C51 ensemble) |
| `DeployAgent` | `train_sim.py` | Inference-only wrapper (encoder + actor) |

### Reward definition

Each env implements `compute_dense_reward` and `compute_normalized_dense_reward`.
For `PlaceBowlCube` (Eval 1) the reward is **staged** — it assigns a base value
to each behavioural phase and adds shaped sub-terms on top:

```
Phase 0 — free (not grasped):    reward = reaching_reward             ∈ [0, 2]
Phase 1 — grasped:               reward = 3 + place_reward            ∈ [3, 5]
Phase 2 — above bowl:            reward = 4 + place + dropped
                                         + gripper_openness + static  ∈ [4, 8]
Phase 3 — success:               reward = 9

Penalties (always active):
  robot touching table  → −6
  robot touching bowl   → −3
  item not lifted       → −1
```

`compute_normalized_dense_reward` divides by 9 to put rewards in `[−1, 1]`,
matching the critic support `[v_min, v_max] = [−20, 20]`.

### Observation encoding

Every observation is encoded the same way for both actor and critic:

```
RGB (16×16×C) ──► CNNEncoder ──► 1024-dim feature
                                         │
state vector ─────────────────► Projection ──► 306-dim joint embedding
                                 (50-dim rgb_proj ⊕ 256-dim state_proj)
```

The CNN backbone is **only in the critic optimizer** — it learns entirely
through the critic's loss. The actor reads off the resulting features without
gradient flow back to the encoder.

### Policy — Actor

The actor outputs a **Gaussian with tanh squashing**:

```
joint_embedding → 3 × [Linear(256) + LayerNorm + ReLU]
               → fc_mean   → μ
               → fc_logstd → σ  (clamped log-std ∈ [−5, 2])

x_t  ~ N(μ, σ)                     ← reparameterisation trick
a    = tanh(x_t) × scale + bias    ← squashed into action bounds

log π(a|s) = Σ log N(x_t) − log(scale × (1 − tanh²(x_t)) + ε)
               ↑ Gaussian log-prob   ↑ tanh change-of-variables correction
```

At eval/deploy time `get_eval_action` uses `tanh(μ)` directly — no sampling.

### Critic — Distributional C51 ensemble

Instead of a scalar Q-value, each Q-network outputs **101 logits** over a
fixed support `[−20, 20]`. The expected Q-value is `softmax(logits) · support`.
Two Q-networks run in parallel via `torch.vmap` over stacked parameters.

### Critic update

```
1. Sample next action a' and log π(a'|s') from current policy (no_grad)
2. Soft Bellman target:
     r_soft = r − γ · bootstrap · α · log π(a'|s')
3. C51 categorical projection using target critic:
     Φ = C51_project(Q_target(s', a'), r_soft, γ)   # [num_q, batch, 101]
4. Cross-entropy loss:
     L_critic = −Σ Φ · log softmax(Q_online(s, a))
5. Backprop through critic + encoder
```

The C51 projection shifts and clips the target atoms by `r + γ·z`, then
redistributes probability mass onto the nearest two support atoms via linear
interpolation (Bellemare et al. 2017).

### Actor update (every 4 critic steps)

```
L_actor = E[ α · log π(a|s) − mean_Q(s, a) ]
```

Critic weights are frozen during this step — only the actor MLP gets gradients.

### Entropy coefficient α

`log_alpha` is a learned scalar. Its loss drives the policy entropy toward a
target of `−dim(action_space)`:

```
L_α = −log_alpha · (log π(a|s) + target_entropy)
```

### Target network

Polyak averaging after every critic step:

```
θ_target ← (1 − τ) · θ_target + τ · θ_online     (τ = 0.01)
```

### Full update cycle per iteration

```
collect 1 step × 1024 parallel envs  →  store in replay buffer

repeat 256 gradient steps:
  sample 512 transitions
  ├─ update_main:    critic + encoder + α
  ├─ every 4th step: update_actor (actor MLP only)
  └─ every step:     Polyak-update critic_target
```

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
