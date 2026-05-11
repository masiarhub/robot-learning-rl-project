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
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, quat_from_euler_xyz, subtract_frame_transforms

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


def object_orientation_z_angle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The z-axis Euler angle of the object expressed in the robot's root frame.

    This extracts the in-plane rotation of the cube relative to the robot base,
    which helps the policy plan the gripper approach angle.

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot articulation.
        object_cfg: Configuration for the target object.

    Returns:
        Z-axis rotation angle (in radians) of shape (num_envs,).
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

    # Extract z-axis Euler angle from the transformed quaternion.
    # For a pure z-rotation relative to the base, convert to euler and take the z component.
    # Using the standard wxyz → zyx euler extraction:
    # sin_z = 2 * (w*d - x*y)
    # cos_z = 1 - 2*(d^2 + y^2)  ... but simpler to use atan2
    w, x, y, d = obj_quat_b[:, 0], obj_quat_b[:, 1], obj_quat_b[:, 2], obj_quat_b[:, 3]
    sin_z = 2.0 * (w * d + x * y)
    cos_z = 1.0 - 2.0 * (x * x + d * d)
    z_angle = torch.atan2(sin_z, cos_z)

    return z_angle  # (num_envs,)
