# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the pick_place task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.frame_transformer import FrameTransformer

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_pick_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "pick_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the pick goal position.

    Args:
        env: The environment.
        command_name: The name of the pick command. Defaults to "pick_pose".
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return distance < threshold


def object_reached_place_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "place_pose",
    threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the place goal position.

    Args:
        env: The environment.
        command_name: The name of the place command. Defaults to "place_pose".
        threshold: The threshold for the object to reach the goal position. Defaults to 0.05.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the object to the goal: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # also check that object is at appropriate height (not on ground)
    object_height = object.data.root_pos_w[:, 2]
    return (distance < threshold) & (object_height > 0.02)


def pick_place_success(
    env: ManagerBasedRLEnv,
    pick_threshold: float = 0.02,
    place_threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for successful pick and place.

    This terminates when the object has been picked and placed in the target zone.

    Args:
        env: The environment.
        pick_threshold: The threshold for picking. Defaults to 0.02.
        place_threshold: The threshold for placing. Defaults to 0.05.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # Check if object reached place goal
    return object_reached_place_goal(
        env,
        command_name="place_pose",
        threshold=place_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
    )

# in terminations.py
def cube_placed_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    ee_min_height_above_bowl: float = 0.02,
    consecutive_steps: int = 3,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    bowl = env.scene[bowl_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    cube_pos = obj.data.root_pos_w[:, :3]
    bowl_pos = bowl.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    c1 = torch.norm(cube_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold
    c2 = cube_pos[:, 2] < (bowl_pos[:, 2] + z_max)
    c3 = ee_pos[:, 2] > (bowl_pos[:, 2] + ee_min_height_above_bowl)
    satisfied = c1 & c2 & c3

    if not hasattr(env, "_cube_placed_counter"):
        env._cube_placed_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env._cube_placed_counter[~satisfied] = 0
    env._cube_placed_counter[satisfied] += 1
    return env._cube_placed_counter >= consecutive_steps