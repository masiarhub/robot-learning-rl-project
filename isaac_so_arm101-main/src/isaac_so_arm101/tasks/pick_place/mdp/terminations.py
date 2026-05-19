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

def no_task_progress(
    env: ManagerBasedRLEnv,
    window_steps: int = 100,
    min_progress: float = 0.001,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
) -> torch.Tensor:


    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]

    dist = torch.norm(
        obj.data.root_pos_w[:, :3] - ee.data.target_pos_w[..., 0, :], dim=-1
    )
    contact_force = (
        cube_sensor.data.net_forces_w_history[:, :, 0]
        .norm(dim=-1)
        .max(dim=-1)[0]
    )
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    object_height = obj.data.root_pos_w[:, 2]

    # composite progress signal — any stage improving resets counter
    progress_signal = (
        -dist                  # closer = better
        + 0.1 * contact_force  # any contact = better
        + 0.1 * (-gripper_pos) # closing gripper = better
        + 5.0 * object_height  # lifting = better
    )

    # init buffers
    if not hasattr(env, "_best_progress"):
        env._best_progress = progress_signal.clone()
        env._stagnation_counter = torch.zeros(env.num_envs, device=env.device)

    # reset buffers on episode start
    reset_mask = env.episode_length_buf == 1
    env._best_progress = torch.where(reset_mask, progress_signal, env._best_progress)
    env._stagnation_counter = torch.where(
        reset_mask,
        torch.zeros_like(env._stagnation_counter),
        env._stagnation_counter,
    )

    # check if any meaningful progress was made
    improved = (progress_signal - env._best_progress) > min_progress
    env._best_progress = torch.maximum(env._best_progress, progress_signal)
    env._stagnation_counter = torch.where(
        improved,
        torch.zeros_like(env._stagnation_counter),
        env._stagnation_counter + 1,
    )

    return env._stagnation_counter >= window_steps