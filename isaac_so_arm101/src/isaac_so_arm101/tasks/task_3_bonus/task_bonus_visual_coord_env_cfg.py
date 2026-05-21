# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Visual-coord alias for the Task Bonus singulation environment.
# The base TaskBonusEnvCfg already provides a fully deployable asymmetric AC:
#
#   Actor  (24 dims): joint_pos 6 + joint_vel 6 + ee_pos 3
#                     + initial_stack_center 3 + actions 6
#     → deployable: only needs FK + a one-shot stack position measurement.
#
#   Critic (36 dims): same robot state + current positions of all 3 cubes.
#
# At deployment the actor input is obtained from:
#   - joint_pos_rel, joint_vel_rel : encoder readings - default joint positions
#   - ee_pos                       : FK using ee_position_in_robot_root_frame_for_deployment
#   - initial_stack_center         : measure the stack once at episode start
#                                    (depth sensor, overhead camera, or manual marker)

from . import mdp
from .task_bonus_env_cfg import TaskBonusEnvCfg, EventCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass


@configclass
class TaskBonusVisualCoordEnvCfg(TaskBonusEnvCfg):
    """Task Bonus singulation environment — deployable asymmetric AC variant."""
    pass


@configclass
class PlayEventCfg(EventCfg):
    """EventCfg with cube color randomization re-enabled for visualization."""

    randomize_cube_colors = EventTerm(
        func=mdp.randomize_three_cube_colors,
        mode="reset",
    )


@configclass
class TaskBonusVisualCoordPlayEnvCfg(TaskBonusVisualCoordEnvCfg):
    """Play variant: same policy, but cubes get random colors each episode."""

    events: PlayEventCfg = PlayEventCfg()
