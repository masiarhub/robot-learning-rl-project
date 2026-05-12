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
from isaaclab.utils.math import subtract_frame_transforms

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


def gripper_link_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_link"]),
) -> torch.Tensor:
    """Position of the fixed (non-moving) gripper base link in the robot root frame.

    Stable across gripper open/close cycles — unlike the EE tip, this point does not
    shift when the jaw moves, giving the policy a clean spatial anchor for the gripper.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]
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
