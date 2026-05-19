# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from .._wrist_cam import (
    OFFSET_POS as _CAM_OFFSET_POS,
    OFFSET_QUAT_WXYZ as _CAM_OFFSET_QUAT_WXYZ,
    TAN_HALF_HFOV as _TAN_HALF_HFOV,
    TAN_HALF_VFOV as _TAN_HALF_VFOV,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_offset: float = 0.0,
) -> torch.Tensor:
    """The position of the object in the robot's root frame, with an optional z offset in world frame.

    height_offset is applied in world frame before the frame transform, so it shifts the
    returned position upward by that amount (e.g. 0.12 m above the bowl centre).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3].clone()
    object_pos_w[:, 2] += height_offset
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def initial_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the cube at episode reset time, expressed in the robot root frame.

    Frozen for the duration of the episode — stored in env._initial_cube_pos_w by
    reset_bowl_and_cube and transformed to the robot frame each step. Falls back to
    zeros before the first reset (e.g. during env initialisation).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_initial_cube_pos_w"):
        return torch.zeros(env.num_envs, 3, device=env.device)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        env._initial_cube_pos_w,
    )
    return pos_b


def initial_red_cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the red cube at episode reset time, expressed in the robot root frame.

    Reads env._initial_red_cube_pos_w set by reset_bowl_and_two_cubes.
    Use this in Task 2 two-cube environments; use initial_object_position_in_robot_root_frame
    only in single-cube environments where reset_bowl_and_cube sets _initial_cube_pos_w.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_initial_red_cube_pos_w"):
        env._initial_red_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        env._initial_red_cube_pos_w,
    )
    return pos_b


def initial_blue_cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of the blue cube at episode reset time, expressed in the robot root frame.

    Mirrors initial_object_position_in_robot_root_frame but reads
    env._initial_blue_cube_pos_w set by reset_bowl_and_two_cubes.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_initial_blue_cube_pos_w"):
        env._initial_blue_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        env._initial_blue_cube_pos_w,
    )
    return pos_b


def target_color_one_hot(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Two-element one-hot encoding of the target cube colour for the current episode.

    [1, 0] = red is target (target_color_id == 0)
    [0, 1] = blue is target (target_color_id == 1)
    """
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.randint(0, 2, (env.num_envs,), dtype=torch.int64, device=env.device)
    one_hot = torch.zeros(env.num_envs, 2, device=env.device)
    one_hot.scatter_(1, env._target_color_id.unsqueeze(1), 1.0)
    return one_hot


def object_orientation_z_angle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The z-axis yaw of the object in the robot's root frame, encoded as [sin(θ), cos(θ)].

    This encoding is continuous and wraps correctly at ±π, avoiding the discontinuity
    of a raw angle while only costing 2 observation dimensions.

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot articulation.
        object_cfg: Configuration for the target object.

    Returns:
        [sin(yaw), cos(yaw)] tensor of shape (num_envs, 2).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # Object quaternion in world frame (wxyz convention)
    obj_quat_w = obj.data.root_quat_w.clone()  # (num_envs, 4)
    # Robot root quaternion in world frame (wxyz)
    robot_quat_w = robot.data.root_state_w[:, 3:7].clone()  # (num_envs, 4)

    # Transform object quaternion into robot root frame:
    # obj_quat_b = robot_quat^-1 * obj_quat_w
    robot_quat_conj = quat_conjugate(robot_quat_w)
    obj_quat_b = quat_mul(robot_quat_conj, obj_quat_w)

    # Extract sin(z) and cos(z) directly from quaternion components (wxyz order).
    w, x, y, d = obj_quat_b[:, 0], obj_quat_b[:, 1], obj_quat_b[:, 2], obj_quat_b[:, 3]
    sin_z = 2.0 * (w * d + x * y)
    cos_z = 1.0 - 2.0 * (x * x + d * d)

    return torch.stack([sin_z, cos_z], dim=-1)  # (num_envs, 2)

def gripper_link_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
) -> torch.Tensor:
    """Position of the fixed (non-moving) gripper base link in the robot root frame.

    Stable across gripper open/close cycles — unlike the EE tip, this point does not
    shift when the jaw moves, giving the policy a clean spatial anchor for the gripper.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.find_bodies(["gripper_link"])[0][0]
    gripper_pos_w = robot.data.body_pos_w[:, body_idx, :]
    gripper_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], gripper_pos_w
    )
    return gripper_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in the robot root frame.

    Reads the already-existing ee_frame FrameTransformer (no extra sensor cost).
    Gives the policy an explicit Cartesian EE position so it doesn't have to
    learn forward kinematics implicitly from joint_pos alone.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]          # [B, 3] world frame
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b                                            # [B, 3] robot root frame


def ee_position_in_robot_root_frame_for_deployment(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
    ee_offset: tuple = (0.01, 0.0, -0.09),
) -> torch.Tensor:
    """EE position in robot root frame — computed from FK + fixed local offset only.

    Identical output to ee_position_in_robot_root_frame but uses body_pos_w /
    body_quat_w directly instead of the FrameTransformer sensor, so the logic
    maps 1-to-1 to a standard FK call on the real robot.

    Kinematic chain:
        base_link → [shoulder_pan] → shoulder_link → [shoulder_lift] → upper_arm_link
                  → [elbow_flex]   → lower_arm_link → [wrist_flex]   → wrist_link
                  → [wrist_roll]   → gripper_link

    gripper_link is the FIXED jaw. The gripper revolute joint only moves
    moving_jaw_so101_v1_link and does NOT affect gripper_link's pose.
    ee_offset [0.01, 0.0, -0.09] (metres, gripper_link local frame) approximates
    the fingertip centre and matches the FrameTransformerCfg in joint_pos_env_cfg.py.

    -------------------------------------------------------------------------
    DEPLOYMENT FK SNIPPET (copy-paste, requires: pip install pin)
    -------------------------------------------------------------------------

    import numpy as np
    import pinocchio as pin

    # --- one-time setup at startup ---
    URDF_PATH = "isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf"
    model = pin.buildModelFromUrdf(URDF_PATH)
    data  = model.createData()

    # joint index helpers  (Pinocchio uses tree order, not URDF file order)
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
    EE_OFFSET = np.array([0.01, 0.0, -0.09])  # metres, gripper_link local frame

    # default joint positions (must match joint_pos_env_cfg.py InitialStateCfg)
    Q_DEFAULT = {
        "shoulder_pan":  0.0,
        "shoulder_lift": -0.4,
        "elbow_flex":    -0.3,
        "wrist_flex":    1.57,
        "wrist_roll":    -1.57,
        "gripper":       0.2,
    }

    # --- per-step call ---
    def get_ee_obs(q_abs: dict) -> tuple[np.ndarray, np.ndarray]:
        \"\"\"
        q_abs: {joint_name: angle_rad} — absolute encoder readings.
        Returns (joint_pos_rel, ee_pos) ready to feed into the policy.
            joint_pos_rel : (6,)  zero-centred around training defaults
            ee_pos        : (3,)  fingertip-centre in robot base frame [m]
        \"\"\"
        q = np.zeros(model.nq)
        for name, idx in J_IDX.items():
            q[idx] = q_abs[name]

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        T = data.oMf[GRIPPER_LINK_FRAME_ID]          # SE3: base_link → gripper_link
        ee_pos = T.translation + T.rotation @ EE_OFFSET  # (3,) in base frame

        joint_pos_rel = np.array([
            q_abs["shoulder_pan"]  - Q_DEFAULT["shoulder_pan"],
            q_abs["shoulder_lift"] - Q_DEFAULT["shoulder_lift"],
            q_abs["elbow_flex"]    - Q_DEFAULT["elbow_flex"],
            q_abs["wrist_flex"]    - Q_DEFAULT["wrist_flex"],
            q_abs["wrist_roll"]    - Q_DEFAULT["wrist_roll"],
            q_abs["gripper"]       - Q_DEFAULT["gripper"],
        ])
        return joint_pos_rel, ee_pos

    -------------------------------------------------------------------------
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.find_bodies(["gripper_link"])[0][0]

    gripper_pos_w  = robot.data.body_pos_w[:, body_idx, :]   # [B, 3]
    gripper_quat_w = robot.data.body_quat_w[:, body_idx, :]  # [B, 4] wxyz

    offset = torch.tensor(ee_offset, device=env.device, dtype=gripper_pos_w.dtype).expand(env.num_envs, -1)
    ee_pos_w = gripper_pos_w + quat_apply(gripper_quat_w, offset)          # [B, 3]

    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b  # [B, 3]


# ---------------------------------------------------------------------------
# Frozen ResNet18 encoder — singleton, built once on first call.
# Output: 512-dim feature vector (layer4 → AdaptiveAvgPool2d(1,1) → flatten).
# ---------------------------------------------------------------------------

_RESNET_DIM = 512

_resnet_encoder: nn.Module | None = None
_imagenet_mean: torch.Tensor | None = None
_imagenet_std: torch.Tensor | None = None


def _get_resnet_encoder(device: torch.device) -> nn.Module:
    global _resnet_encoder, _imagenet_mean, _imagenet_std
    if _resnet_encoder is not None:
        return _resnet_encoder

    import torchvision.models as tv_models

    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    _resnet_encoder = encoder.to(device)
    _imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    print(f"[INFO] Frozen ResNet18 encoder initialised on {device} — output dim: {_RESNET_DIM}")
    return _resnet_encoder


def wrist_camera_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    flatten: bool = True,
) -> torch.Tensor:
    """Encode the wrist camera RGB image via a frozen pretrained ResNet18.

    Returns a [B, 512] feature vector. The encoder is built once on the first
    call and reused for all subsequent steps. ImageNet normalisation is applied
    before the forward pass so the pretrained weights are used correctly.

    The ``flatten`` parameter is kept for API compatibility but has no effect —
    the output is always a flat 512-dim vector per environment.
    """
    if not env.sim.is_playing():
        return torch.zeros(env.num_envs, _RESNET_DIM, device=env.device)

    sensor = env.scene.sensors[sensor_cfg.name]
    raw = sensor.data.output["rgb"]                                         # [B, H, W, 3]
    img = raw[..., :3].permute(0, 3, 1, 2).float().clamp(0.0, 255.0) / 255.0  # [B, 3, H, W]
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    if env.common_step_counter % 100 == 0:
        import imageio, os
        # 1) Raw sensor output — uint8 straight from the simulator.
        raw_np = raw[0, ..., :3].cpu().numpy()
        imageio.imwrite(os.path.expanduser("~/robot-learning/wrist_cam_raw.png"), raw_np)
        # 2) What ResNet actually receives: apply ImageNet norm then rescale to [0,1] for display.
        mean = torch.tensor([0.485, 0.456, 0.406], device=env.device).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=env.device).view(3, 1, 1)
        resnet_input = (img[0] - mean) / std          # [3, H, W], range ≈ [-2, 2]
        lo, hi = resnet_input.min(), resnet_input.max()
        display_np = ((resnet_input - lo) / (hi - lo + 1e-8)).permute(1, 2, 0).cpu().numpy()
        imageio.imwrite(
            os.path.expanduser("~/robot-learning/wrist_cam_resnet_input.png"),
            (display_np * 255).astype("uint8"),
        )

        print(
            f"[DEBUG] raw    → ~/robot-learning/wrist_cam_raw.png          shape={raw.shape} max={raw.max():.1f}\n"
            f"[DEBUG] resnet → ~/robot-learning/wrist_cam_resnet_input.png img∈[{img.min():.3f},{img.max():.3f}]"
        )


    encoder = _get_resnet_encoder(env.device)
    img = (img - _imagenet_mean) / _imagenet_std

    with torch.no_grad():
        features = encoder(img)  # [B, 512, 1, 1]

    return features.flatten(start_dim=1)  # [B, 512]


##
# Visual-coord observations (no camera sensor, analytic FK projection)
##


def target_cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
) -> torch.Tensor:
    """Current position of the TARGET cube in the robot root frame.

    Uses env._target_is_red to select between object_red and object_blue.
    Critic-side privileged observation — not available at deployment.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    red_obj: RigidObject = env.scene[red_object_cfg.name]
    blue_obj: RigidObject = env.scene[blue_object_cfg.name]

    red_pos_w  = red_obj.data.root_pos_w[:, :3]
    blue_pos_w = blue_obj.data.root_pos_w[:, :3]

    if hasattr(env, "_target_is_red"):
        tgt_red = env._target_is_red
    elif hasattr(env, "_target_color_id"):
        tgt_red = env._target_color_id == 0
    else:
        tgt_red = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    target_pos_w = torch.where(tgt_red.unsqueeze(1), red_pos_w, blue_pos_w)
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_pos_w
    )
    return target_pos_b


def target_cube_image_coords(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
) -> torch.Tensor:
    """Analytic projection of the TARGET cube onto the wrist camera image plane.

    Uses FK + the fixed camera offset from _wrist_cam.py — no TiledCamera sensor needed.
    Selects between object_red and object_blue based on env._target_is_red set by
    set_two_cube_colors each episode reset.

    Returns (num_envs, 3): [u, v, visible]
        u, v   : NDC coordinates clamped to [-1, 1] (0,0 = image centre).
        visible: 1.0 if target cube is inside the camera frustum, 0.0 otherwise.

    At deployment: run HSV segmentation for the target color, compute blob centroid
    in pixels, normalise to NDC. The policy sees the same format in both cases.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    red_obj: RigidObject = env.scene[red_object_cfg.name]
    blue_obj: RigidObject = env.scene[blue_object_cfg.name]

    body_idx = robot.find_bodies(["gripper_link"])[0][0]
    gripper_pos_w  = robot.data.body_pos_w[:, body_idx, :]   # (B, 3)
    gripper_quat_w = robot.data.body_quat_w[:, body_idx, :]  # (B, 4) wxyz

    offset_pos = torch.tensor(_CAM_OFFSET_POS, device=env.device, dtype=gripper_pos_w.dtype)
    cam_pos_w = gripper_pos_w + quat_apply(gripper_quat_w, offset_pos.expand(env.num_envs, -1))

    cam_local_q = torch.tensor(
        _CAM_OFFSET_QUAT_WXYZ, device=env.device, dtype=gripper_quat_w.dtype
    ).unsqueeze(0).expand(env.num_envs, -1)
    cam_quat_w = quat_mul(gripper_quat_w, cam_local_q)

    def _project(obj_pos_w: torch.Tensor):
        pts_cam, _ = subtract_frame_transforms(cam_pos_w, cam_quat_w, obj_pos_w)
        z = pts_cam[:, 2]
        safe_neg_z = (-z).clamp(min=1e-3)
        u = pts_cam[:, 0] / (safe_neg_z * _TAN_HALF_HFOV)
        v = pts_cam[:, 1] / (safe_neg_z * _TAN_HALF_VFOV)
        in_view = (z < -1e-3) & (u.abs() <= 1.0) & (v.abs() <= 1.0)
        return u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0), in_view.float()

    red_u, red_v, red_vis  = _project(red_obj.data.root_pos_w[:, :3])
    blue_u, blue_v, blue_vis = _project(blue_obj.data.root_pos_w[:, :3])

    if hasattr(env, "_target_is_red"):
        tgt_red = env._target_is_red
    elif hasattr(env, "_target_color_id"):
        tgt_red = env._target_color_id == 0
    else:
        tgt_red = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    u   = torch.where(tgt_red, red_u,   blue_u)
    v   = torch.where(tgt_red, red_v,   blue_v)
    vis = torch.where(tgt_red, red_vis, blue_vis)

    return torch.stack([u, v, vis], dim=-1)


def random_target_color_one_hot(
    env: ManagerBasedRLEnv,
    num_colors: int = 6,
) -> torch.Tensor:
    """Six-class one-hot encoding of the target cube color for the current episode.

    Color index → one-hot position (must match events.set_two_cube_colors):
        0=blue  1=red  2=green  3=yellow  4=purple  5=orange

    The color ID is set each episode by set_two_cube_colors, which also applies
    the matching visual to the corresponding cube in the simulator. This function
    just reads that stored value and builds the one-hot vector.

    Deployment: pass a fixed one-hot for the desired target color. Run HSV
    segmentation for that color and supply its (u, v, visible) as the
    target_cube_image observation.
    """
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    one_hot = torch.zeros(env.num_envs, num_colors, device=env.device)
    one_hot.scatter_(1, env._target_color_id.unsqueeze(1), 1.0)
    return one_hot
