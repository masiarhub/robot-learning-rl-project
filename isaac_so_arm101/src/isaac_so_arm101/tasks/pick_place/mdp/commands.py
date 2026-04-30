# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BowlHoverCommand(UniformPoseCommand):
    """Command term that sets the goal to a fixed height above the bowl.

    On every resample (i.e. episode reset) it reads the bowl's current world-frame
    position and sets the command to bowl_pos + [0, 0, hover_height], expressed in
    the robot body frame. This means the goal automatically follows the bowl when
    bowl position randomization is added later.
    """

    cfg: BowlHoverCommandCfg

    def __init__(self, cfg: BowlHoverCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.bowl: RigidObject = env.scene[cfg.bowl_name]

    def _resample_command(self, env_ids: Sequence[int]):
        # Read the bowl's current world-frame position for the envs being reset.
        bowl_pos_w = self.bowl.data.root_pos_w[env_ids]  # (len(env_ids), 3)

        # Hover point: directly above the bowl centre.
        hover_pos_w = bowl_pos_w.clone()
        hover_pos_w[:, 2] += self.cfg.hover_height

        # Convert world frame → robot body frame.
        # subtract_frame_transforms is the inverse of combine_frame_transforms:
        #   pos_b = R^T * (pos_w - origin_w)
        robot_pos_w = self.robot.data.root_pos_w[env_ids]
        robot_quat_w = self.robot.data.root_quat_w[env_ids]
        target_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, hover_pos_w)

        self.pose_command_b[env_ids, :3] = target_pos_b
        # Identity quaternion — orientation is not tracked for this task.
        self.pose_command_b[env_ids, 3] = 1.0
        self.pose_command_b[env_ids, 4:] = 0.0


@configclass
class BowlHoverCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`BowlHoverCommand`."""

    class_type: type = BowlHoverCommand

    bowl_name: str = "bowl"
    """Scene entity name of the bowl. Defaults to ``'bowl'``."""

    hover_height: float = 0.12
    """Height above the bowl centre to use as the goal (in metres). Defaults to 0.12."""

    def __post_init__(self):
        # ranges are unused by BowlHoverCommand (goal is derived from bowl position),
        # but UniformPoseCommandCfg requires them to be non-MISSING at config time.
        self.ranges = UniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        )
