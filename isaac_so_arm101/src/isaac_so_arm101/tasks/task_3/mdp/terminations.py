# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_placed_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    ee_min_height_above_bowl: float = 0.055,
    consecutive_steps: int = 3,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Success termination: True once all conditions hold for `consecutive_steps` in a row.

    The buffer prevents spurious terminations from transient physics states (e.g. the
    cube bouncing briefly through the bowl position).  The counter is stored on the env
    object and is self-resetting: any step where the conditions are not all met resets
    the counter to 0, including the first step after an episode reset when the cube is
    placed outside the bowl by the exclusion zone.

    Mirrors the cube_in_bowl reward conditions exactly — keep thresholds in sync.

    consecutive_steps: number of back-to-back steps all three conditions must hold.
                       2–3 is enough to filter single-step physics transients.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    cube_pos = obj.data.root_pos_w[:, :3]
    bowl_pos = bowl.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    c1 = torch.norm(cube_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold
    c2 = cube_pos[:, 2] < (bowl_pos[:, 2] + z_max)
    # C3 — EE has moved upward away from bowl
    c3 = ee_pos[:, 2] > (bowl_pos[:, 2] + ee_min_height_above_bowl)

    satisfied = c1 & c2 & c3




    # Lazy-initialise a per-env step counter on the env object.
    if not hasattr(env, "_cube_in_bowl_steps"):
        env._cube_in_bowl_steps = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Increment where all conditions hold; reset to 0 on any violation.
    env._cube_in_bowl_steps = torch.where(
        satisfied,
        env._cube_in_bowl_steps + 1,
        torch.zeros_like(env._cube_in_bowl_steps),
    )

    return env._cube_in_bowl_steps >= consecutive_steps


def target_cube_placed_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    ee_min_height_above_bowl: float = 0.055,
    consecutive_steps: int = 3,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Success termination for the four-cube task.

    Triggers when, for `consecutive_steps` in a row:
      1. The target cube is inside the bowl (XY < threshold, z in range).
      2. No wrong cube is inside the bowl.
      3. The EE has retreated above the bowl (released the cube).
    """
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    tid = env._target_color_id
    N = env.num_envs

    bowl: RigidObject = env.scene[bowl_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    bowl_pos = bowl.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    def _in_bowl(obj_pos: torch.Tensor) -> torch.Tensor:
        c1 = torch.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold
        c2 = obj_pos[:, 2] < bowl_pos[:, 2] + z_max
        return c1 & c2

    all_in = torch.stack(
        [
            _in_bowl(env.scene[red_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[blue_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[green_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3]),
        ],
        dim=1,
    )  # [N, 4] bool
    target_in = all_in[torch.arange(N, device=tid.device), tid]
    target_mask = torch.zeros(N, 4, dtype=torch.bool, device=env.device)
    target_mask.scatter_(1, tid.unsqueeze(1), True)
    any_wrong_in = (all_in & ~target_mask).any(dim=1)
    ee_retreated = ee_pos[:, 2] > (bowl_pos[:, 2] + ee_min_height_above_bowl)

    satisfied = target_in & (~any_wrong_in) & ee_retreated

    if not hasattr(env, "_target_cube_in_bowl_steps"):
        env._target_cube_in_bowl_steps = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
    env._target_cube_in_bowl_steps = torch.where(
        satisfied,
        env._target_cube_in_bowl_steps + 1,
        torch.zeros_like(env._target_cube_in_bowl_steps),
    )
    return env._target_cube_in_bowl_steps >= consecutive_steps


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
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
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


def target_cube_dropping(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Terminate when the *target* cube falls below minimum_height (dropped off table)."""
    from .rewards import _get_target_color_id
    tid = _get_target_color_id(env)
    N = env.num_envs
    all_dropped = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[blue_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[green_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
        ],
        dim=1,
    )  # [N, 4] bool
    return all_dropped[torch.arange(N, device=tid.device), tid]
