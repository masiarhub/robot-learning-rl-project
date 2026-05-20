# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wrist_camera_image(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns flattened normalized RGB image from the wrist camera.

    Output shape: (num_envs, H * W * 3)
    """
    sensor = env.scene[sensor_cfg.name]
    # data.output["rgb"] shape: (num_envs, H, W, 4) — last channel is alpha
    rgb = sensor.data.output["rgb"][..., :3].float() / 255.0
    return rgb.flatten(start_dim=1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def bowl_center_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns bowl position in robot root frame. Shape: (num_envs, 3)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    bowl: RigidObject = env.scene[asset_cfg.name]
    bowl_pos_w = bowl.data.root_pos_w[:, :3]
    bowl_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], bowl_pos_w
    )
    return bowl_pos_b