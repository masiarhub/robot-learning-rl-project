# Task 1 Visual-Coordinate Policy — Sim-to-Real Deployment Guide

> **Audience**: Engineer deploying the Task 1 VisualCoord policy on the real SO-ARM101 for the first time.
> This policy replaces the teacher (which needed the true 3D cube position) with analytic wrist-camera
> image coordinates produced by HSV colour segmentation. No camera neural network, no ResNet, no depth
> sensor — only joint encoders + one camera frame per control step.

---

## 1. Where to Find the Relevant Files

```
robot-learning-rl-project/
├── Sim-to-Real/
│   ├── INSTRUCTIONS.md                          ← this file
│   └── task_1/
│       ├── task_1_no_camera/                    ← teacher & initial-pos policies (reference only)
│       └── task_1_visual_coord/                 ← VisualCoord checkpoint (copied here after training)
│           └── <run_name>/
│               ├── exported/
│               │   ├── policy.pt                ← TorchScript model for deployment
│               │   └── policy.onnx              ← ONNX export (optional)
│               ├── params/
│               │   ├── agent.yaml               ← network architecture + normalisation
│               │   └── env.yaml                 ← sim settings for reference
│               └── model_<iter>.pt              ← raw RSL-RL checkpoint
│
└── isaac_so_arm101/src/isaac_so_arm101/tasks/task_1/
    ├── _wrist_cam.py                            ← authoritative camera constants (FOV, offset)
    ├── _colors.py                               ← 6-colour palette HSV targets
    ├── task_one_visual_coord_env_cfg.py         ← VisualCoord env config (obs, rewards, DR)
    ├── agents/rsl_rl_ppo_cfg.py                 ← network architecture config
    ├── joint_pos_env_cfg.py                     ← robot init state + action scaling
    └── mdp/
        ├── observations.py                      ← cube_image_coords() — the FK projection
        └── events.py                            ← set_cube_target_color() — colour assignment
```

Training command (for reference, runs in `isaac_so_arm101/`):

```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py \
    --task Isaac-SO-ARM101-Task-One-VisualCoord-v0 \
    --num_envs 4096 --headless
```

After training, copy the run folder from `logs/rsl_rl/task_1_visual_coord/<run_name>/` into
`Sim-to-Real/task_1/task_1_visual_coord/`. Use `policy.pt` (TorchScript) for deployment.

---

## 2. What Changed vs the Teacher Policy

| Aspect | Teacher (task_1_no_camera) | VisualCoord (this policy) |
|---|---|---|
| **Cube observation** | True 3D cube position (robot frame) — oracle | (u, v, visible) from HSV segmentation |
| **Color** | Single cube, fixed red | Single cube, random colour each episode from 6-palette |
| **Color observation** | None | 6-class one-hot of target colour |
| **EE position** | FK-derived (`ee_position_in_robot_root_frame_for_deployment`) | Same — FK only, no extra sensor |
| **Actor input dims** | 27 | **33** |
| **Critic input dims** | 30 (teacher) | 27 (privileged cube pos, not deployed) |
| **Action space** | Identical | Identical |
| **Network** | [256, 128, 64] | [256, 128, 64] (same size) |
| **Domain randomisation** | Base ranges | Tightened: table friction (0.30–0.55), gripper (0.30–0.70), cube mass ±20% |
| **Curriculum** | cube_in_bowl at 36k steps | cube_in_bowl at **60k steps** (later, softer ramp) |

The critical difference: at **deployment** you replace the oracle 3D cube position with the HSV
blob centroid from the real wrist camera, normalised to NDC. The policy sees the exact same
`[u, v, visible]` format in both cases.

---

## 3. Robot Initial Position

The robot is commanded to a fixed resting pose at the start of each episode. On the real robot
you must drive the arm to this position before the policy starts.

| Joint | Default position (rad) |
|---|---|
| shoulder_pan | 0.0 |
| shoulder_lift | −1.4 |
| elbow_flex | 0.4 |
| wrist_flex | 1.4 |
| wrist_roll | −1.57 |
| gripper | 0.2 |

There is a ±0.02 rad reset jitter on the arm joints in simulation (±0.01 rad on the gripper).
For deployment you can place the arm at the exact default; the policy is trained to handle small
positional errors.

---

## 4. Observations — Exact Order and Dimensions (Actor Input: 33 dims)

The policy network receives a **single flat vector of 33 floats** every control step.
Build it in this exact order:

```
index  dims  name                   source
----------------------------------------------
0–5     6    joint_pos_rel          encoder readings minus default joint positions (see table above)
6–11    6    joint_vel_rel          encoder velocities minus default (≈ 0 at default speed)
12–14   3    ee_pos                 FK + fixed offset (see FK snippet below)
15–17   3    bowl_pos               bowl centre + 0.12 m hover height, in robot base frame
18–20   3    cube_image             [u, v, visible] from HSV segmentation (see Section 6)
21–26   6    color_one_hot          one-hot for the target colour (see Section 7)
27–32   6    last_action            raw network output from the previous step (zero on step 0)
```

**joint_pos_rel** — subtract the default position from the measured absolute encoder reading:
```
joint_pos_rel[0] = q_shoulder_pan  − 0.0
joint_pos_rel[1] = q_shoulder_lift − (−1.4)
joint_pos_rel[2] = q_elbow_flex    − 0.4
joint_pos_rel[3] = q_wrist_flex    − 1.4
joint_pos_rel[4] = q_wrist_roll    − (−1.57)
joint_pos_rel[5] = q_gripper       − 0.2
```

**joint_vel_rel** — the training default velocity is 0 for all joints, so:
```
joint_vel_rel[i] = dq_i / dt   (measured velocity, not subtracted from anything)
```

**ee_pos** — computed by pure FK + fixed offset in gripper_link local frame:

```python
# one-time setup
import pinocchio as pin
URDF_PATH = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
model = pin.buildModelFromUrdf(URDF_PATH)
data  = model.createData()

def _jidx(name):
    return model.joints[model.getJointId(name)].idx_q

J_IDX = {
    "shoulder_pan":  _jidx("shoulder_pan"),
    "shoulder_lift": _jidx("shoulder_lift"),
    "elbow_flex":    _jidx("elbow_flex"),
    "wrist_flex":    _jidx("wrist_flex"),
    "wrist_roll":    _jidx("wrist_roll"),
    "gripper":       _jidx("gripper"),
}
GRIPPER_LINK_FRAME_ID = model.getFrameId("gripper_link")
EE_OFFSET = np.array([0.01, 0.0, -0.09])   # metres, gripper_link local frame

# per-step call
def get_ee_pos(q_abs: dict) -> np.ndarray:
    """Returns EE position in robot base frame (3,)."""
    q = np.zeros(model.nq)
    for name, idx in J_IDX.items():
        q[idx] = q_abs[name]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = data.oMf[GRIPPER_LINK_FRAME_ID]
    return T.translation + T.rotation @ EE_OFFSET
```

**bowl_pos** — position of the bowl's horizontal centre, offset 0.12 m upward, expressed in the
robot base frame. This is a fixed measurement you do once after placing the bowl on the table.
It does NOT change during an episode unless the bowl is moved. Use your calibrated robot base
frame (same frame as the URDF `base_link`).

**cube_image** — see Section 6 below.

**color_one_hot** — see Section 7 below.

**last_action** — the raw 6-dim network output from the previous step. Use zeros on the first step.

---

## 5. Actions — Scaling and Application

The network outputs a **6-dim vector** of unbounded floats (approximately −1 to 1 in practice).

Joint order in the output:
```
index  joint          scale  default_q
0      shoulder_pan   0.5    0.0
1      shoulder_lift  0.5   -1.4
2      elbow_flex     0.5    0.4
3      wrist_flex     0.5    1.4
4      wrist_roll     0.5   -1.57
5      gripper        0.3    0.2
```

Conversion to joint position target:
```
target_q[i] = default_q[i] + scale[i] * action[i]
```

This is `JointPositionActionCfg(scale=..., use_default_offset=True)`. The gripper is **continuous**
(not binary): `action[5]` close to −1 → gripper more closed, close to +1 → gripper more open
(relative to the 0.2 rad default).

Send `target_q` as a position command to the motor controller at every control step
(decimation=2, so 100 Hz / 2 = **50 Hz policy frequency**).

---

## 6. Camera Pipeline — HSV Segmentation to (u, v, visible)

### 6.1 Physical Camera Mount

The wrist camera is mounted on `gripper_link` with the following canonical offset
(from `_wrist_cam.py`, the authoritative source):

| Parameter | Value |
|---|---|
| Position offset (gripper_link local) | (−0.0049, 0.0498, −0.0591) m |
| Rotation (wxyz quaternion) | (0.9537, −0.3035, 0.0, 0.0) |
| Rotation meaning | pitch down 35.31° from gripper_link orientation |
| Simulated image size | 256 × 144 px (16:9) |
| Simulated focal length | 9.8 mm |
| Simulated horizontal aperture | 20.955 mm |
| Resulting sim HFOV | ≈ 93.8° |
| Resulting sim VFOV | ≈ 62.0° |

### 6.2 Recommended Real Camera Resolution

Use the **full native resolution** for capture and HSV detection (1280 × 720 if that is what the
camera outputs). Resolution only affects detection reliability — the (u, v) NDC output is always
in [−1, 1] regardless of pixel dimensions. Higher resolution → more accurate centroid →
cleaner (u, v) signal.

Do **not** downsample to 256×144. That is the simulation sensor resolution; it has nothing to do
with what the deployment code needs. Run at 720p, then compute NDC.

### 6.3 Critical: FOV Calibration

The policy's u, v values were computed assuming the simulated FOV (HFOV ≈ 93.8°). If the real
camera has a **different FOV**, the same physical position of the cube will map to a different NDC
value than in simulation, and the policy will receive wrong observations.

**Step 1 — Measure the real camera's HFOV.** One reliable method:
1. Place a ruler 0.50 m in front of the camera, perpendicular to the optical axis.
2. Check how many centimetres are visible edge-to-edge → `visible_width_m`.
3. `HFOV_real = 2 × atan(visible_width_m / (2 × 0.50))` in degrees.

**Step 2 — Apply correction if needed.**

If HFOV_real ≈ 93.8°: no correction needed, use simple NDC.
If HFOV_real differs:

```python
import math

TAN_HALF_HFOV_SIM = 1.0691   # from _wrist_cam.py (pre-computed)
TAN_HALF_VFOV_SIM = 0.6014   # from _wrist_cam.py (pre-computed)

TAN_HALF_HFOV_REAL = math.tan(math.radians(HFOV_real / 2))
TAN_HALF_VFOV_REAL = math.tan(math.radians(VFOV_real / 2))

scale_u = TAN_HALF_HFOV_REAL / TAN_HALF_HFOV_SIM
scale_v = TAN_HALF_VFOV_REAL / TAN_HALF_VFOV_SIM

# Then in NDC conversion (see 6.4), multiply by scale:
u = u_ndc * scale_u
v = v_ndc * scale_v
```

If the scales are close to 1.0 (within ±10%), you can skip the correction; the policy is
somewhat robust to small FOV mismatches due to observation noise during training.

### 6.4 HSV Segmentation and NDC Conversion

For each camera frame at 50 Hz:

```python
import cv2
import numpy as np

def get_cube_image_coords(frame_bgr, target_hsv_range, W, H,
                          scale_u=1.0, scale_v=1.0,
                          min_blob_area_px=20):
    """
    frame_bgr : BGR image from camera (real resolution, e.g. 1280×720).
    target_hsv_range : ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)) for the target cube colour.
    W, H : frame dimensions.
    scale_u, scale_v : FOV correction factors (1.0 if FOVs match).
    min_blob_area_px : blobs smaller than this are noise and ignored.

    Returns: np.ndarray of shape (3,) — [u, v, visible]
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lo = np.array(target_hsv_range[0], dtype=np.uint8)
    hi = np.array(target_hsv_range[1], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # For red (hue wraps around 0/180): combine two ranges
    # mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0])

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_blob_area_px:
        return np.array([0.0, 0.0, 0.0])

    M = cv2.moments(c)
    px = M["m10"] / M["m00"]   # centroid x, 0 = left edge
    py = M["m01"] / M["m00"]   # centroid y, 0 = top edge

    # NDC: u in [−1, 1] left→right, v in [−1, 1] bottom→top
    u_ndc = (px - W / 2) / (W / 2)
    v_ndc = (H / 2 - py) / (H / 2)   # flip y: pixel 0 is top, but v positive = up

    u = np.clip(u_ndc * scale_u, -1.0, 1.0)
    v = np.clip(v_ndc * scale_v, -1.0, 1.0)

    return np.array([u, v, 1.0])
```

When the cube is not found (no blob above threshold) return `[0.0, 0.0, 0.0]`.
The policy learned that visible=0 means "cube lost" and will attempt to re-orient the wrist
to bring the cube back into view (incentivised by the `cube_visibility` reward during training).

### 6.5 HSV Ranges for the 6 Palette Colours

The simulation colours in sRGB (from `_colors.py` / `events.py`):

| Index | Name | sRGB (0–255) |
|---|---|---|
| 0 | blue | (31, 61, 222) |
| 1 | red | (222, 26, 26) |
| 2 | green | (26, 199, 56) |
| 3 | yellow | (242, 224, 13) |
| 4 | purple | (148, 26, 199) |
| 5 | orange | (242, 128, 13) |

These are the **simulated** colours. For real cubes, convert these to HSV and widen the range
by ±10–15 hue and ±40–60 saturation/value to account for real-world lighting variation.

**Example ranges for red (hue wraps around 0 and 180 in OpenCV HSV):**
```python
# Red requires two ranges:
red_lo1 = (0,   80, 80)
red_hi1 = (10, 255, 255)
red_lo2 = (170,  80, 80)
red_hi2 = (180, 255, 255)
```

**Example for blue:**
```python
blue_lo = (100,  80,  60)
blue_hi = (135, 255, 255)
```

**Tip**: use an interactive HSV picker (e.g. `cv2.createTrackbar`) under the same lighting
conditions you'll run the deployment in. Tune the ranges **on the real cube under real light**.
The simulation colours are approximate — real paint/plastic differs in saturation and value.

---

## 7. Colour One-Hot (6 dims)

At deployment, the target colour is **fixed** for the episode — you pick one cube to pick-and-place
and tell the policy which colour it is via this vector.

| Index | Colour |
|---|---|
| 0 | blue |
| 1 | red |
| 2 | green |
| 3 | yellow |
| 4 | purple |
| 5 | orange |

To select red: `color_one_hot = [0, 1, 0, 0, 0, 0]`
To select blue: `color_one_hot = [1, 0, 0, 0, 0, 0]`

This vector is **constant** for the entire episode (unlike in training, where it was randomised
each reset). The HSV segmentation must filter for the **same** colour as this one-hot encodes.

---

## 8. Full Deployment Loop

```python
import torch
import numpy as np
import cv2

# --- one-time setup ---
policy = torch.jit.load("Sim-to-Real/task_1/task_1_visual_coord/<run>/exported/policy.pt")
policy.eval()

TARGET_COLOR_IDX = 1           # e.g. red
target_hsv_range = (red_lo1, red_hi1)  # plus red_lo2/hi2 combined with OR

color_one_hot = np.zeros(6, dtype=np.float32)
color_one_hot[TARGET_COLOR_IDX] = 1.0

bowl_pos_base = np.array([...])   # calibrated bowl centre in robot base frame, z += 0.12

DEFAULT_Q = np.array([0.0, -1.4, 0.4, 1.4, -1.57, 0.2])
SCALE = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.3])

last_action = np.zeros(6, dtype=np.float32)

# scale_u, scale_v: set from FOV calibration (Section 6.3)
scale_u = 1.0
scale_v = 1.0

W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# --- 50 Hz control loop ---
while True:
    ret, frame = cap.read()
    q_abs = robot.read_joint_positions()   # dict: {joint_name: angle_rad}
    dq    = robot.read_joint_velocities()

    # 1. Observations
    joint_pos_rel = np.array([
        q_abs["shoulder_pan"]  - 0.0,
        q_abs["shoulder_lift"] - (-1.4),
        q_abs["elbow_flex"]    - 0.4,
        q_abs["wrist_flex"]    - 1.4,
        q_abs["wrist_roll"]    - (-1.57),
        q_abs["gripper"]       - 0.2,
    ], dtype=np.float32)

    joint_vel_rel = np.array([
        dq["shoulder_pan"],
        dq["shoulder_lift"],
        dq["elbow_flex"],
        dq["wrist_flex"],
        dq["wrist_roll"],
        dq["gripper"],
    ], dtype=np.float32)

    ee_pos = get_ee_pos(q_abs).astype(np.float32)

    cube_img = get_cube_image_coords(
        frame, target_hsv_range, W, H, scale_u, scale_v
    ).astype(np.float32)

    obs = np.concatenate([
        joint_pos_rel,   # 6
        joint_vel_rel,   # 6
        ee_pos,          # 3
        bowl_pos_base,   # 3
        cube_img,        # 3
        color_one_hot,   # 6
        last_action,     # 6
    ]).astype(np.float32)   # total: 33

    # 2. Policy inference
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0)   # (1, 33)
        action_t = policy(obs_t)                     # (1, 6)
    action = action_t.squeeze(0).numpy()
    last_action = action.copy()

    # 3. Apply action
    target_q = DEFAULT_Q + SCALE * action
    robot.send_joint_position_targets(target_q)

    # 4. Termination: stop when cube_in_bowl and gripper open
    # (implement using the robot's own sensors / a separate contact check)
```

---

## 9. Debugging and Testing the Setup

### 9.1 Verify FOV and Camera Mount

Run the robot to its default position. Place a cube at a known measured position in front of
the camera. Print the raw (u, v) output of `get_cube_image_coords`. Then compute what
`cube_image_coords` from `mdp/observations.py` would return for the same geometric position.
They should agree within ≈ ±0.05. If they don't, re-check the FOV scaling and camera tilt angle.

### 9.2 HSV Segmentation Sanity Check

Before running the policy, stream the camera and display:
- The raw mask for the target colour
- The detected centroid overlaid on the frame
- The computed (u, v) printed to the terminal

This lets you verify that the segmentation is clean and the NDC conversion is correct without
needing the robot to move.

```python
while True:
    ret, frame = cap.read()
    coords = get_cube_image_coords(frame, target_hsv_range, W, H)
    u, v, vis = coords

    # overlay centroid
    if vis > 0:
        cx = int((u + 1) / 2 * W)
        cy = int((1 - v) / 2 * H)
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(frame, f"u={u:.2f} v={v:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NOT VISIBLE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("segmentation", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### 9.3 Physical Cube Colours for Testing

Use **solid matte** coloured cubes — glossy surfaces shift apparent hue under directional light.
Good test sequence:

1. Start with **red** (index 1) — most distinct hue, easiest to segment. Set `color_one_hot[1]=1`.
2. Try **blue** (index 0) — well separated from red in hue space.
3. Try **green** (index 2) — check for confusion with table reflections.
4. Try **yellow** (index 3) — widest saturation range needed.

For each colour:
- Verify the HSV mask is clean under the deployment lighting conditions before running the policy.
- If the mask bleeds onto the table or bowl, **narrow the saturation/value lower bounds** until
  only the cube is segmented.
- If the cube is not detected at certain orientations (cube face reflects differently), widen the
  value (brightness) range.

### 9.4 Policy Behaviour Checklist

Expected behaviour at deployment:

1. **First ~0.5 s**: arm reaches toward the cube while wrist reorients to keep the cube in FOV.
   You should see (u, v) stay close to (0, 0) — the policy learned to track the cube.
2. **Reaching phase**: gripper opens as the EE approaches the cube.
3. **Grasping**: gripper closes, contact forces are felt, cube is lifted.
4. **Transport**: arm moves toward the bowl with cube grasped.
5. **Release**: gripper opens above the bowl, cube drops in.

If the gripper does **not** open before reaching the cube, the `ee_pos` observation may be
miscalculated (check the FK snippet) or the bowl_pos is wrong (the policy confuses "reaching"
and "dropping" phases).

If the arm does **not** reorient toward the cube, (u, v, visible) is likely stuck at (0, 0, 0).
Check the camera stream and HSV segmentation first.

---

## 10. Observation Normalisation

The VisualCoord policy uses **running mean/variance normalisation** on the actor observations
(`actor_obs_normalization: true` in `agent.yaml`). The normalisation statistics are stored
inside `policy.pt` (TorchScript export bundles them). You do **not** need to maintain a separate
normalisation file — loading `policy.pt` and calling it directly handles normalisation internally.

If you load a raw `model_<iter>.pt` checkpoint instead:
- The normaliser weights live under `model.normalizer_mean` / `model.normalizer_var` in the
  RSL-RL checkpoint dict. You must apply them manually before passing obs to the actor.
- Prefer the `exported/policy.pt` to avoid this.

---

## 11. Key Constants Reference

From `_wrist_cam.py`:
```python
FOCAL_LENGTH_MM      = 9.8
HORIZONTAL_APERTURE_MM = 20.955
IMAGE_WIDTH          = 256
IMAGE_HEIGHT         = 144
OFFSET_POS           = (-0.0049, 0.0498, -0.0591)   # gripper_link local frame (m)
OFFSET_QUAT_WXYZ     = (0.9537, -0.3035, 0.0, 0.0)  # pitch down 35.31°
TAN_HALF_HFOV        = 1.0691   # pre-computed
TAN_HALF_VFOV        = 0.6014   # pre-computed
```

From `joint_pos_env_cfg.py`:
```python
DEFAULT_Q = [0.0, -1.4, 0.4, 1.4, -1.57, 0.2]   # [pan, lift, flex, wflex, wroll, gripper]
ARM_SCALE    = 0.5
GRIPPER_SCALE = 0.3
```

From `task_one_env_cfg.py`:
```python
BOWL_HOVER_HEIGHT = 0.12   # metres added to bowl z before expressing bowl_pos in observation
POLICY_FREQ = 50           # Hz  (sim dt=0.01 s × decimation=2)
EPISODE_LENGTH = 8.0       # seconds
```
