# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Teacher environment for Task 3 (Phase 1a) — two-cube color-conditioned pick-and-place.
#
# The teacher actor receives the full privileged state:
#   joint_pos 6 + joint_vel 6 + ee_pos 3 + red_cube_pos 3 + blue_cube_pos 3
#   + bowl_pos 3 + target_color_one_hot 2 + actions 6  =  32 dims.
#
# The teacher observation space (actor) is also used verbatim as the TeacherCfg
# in task_three_distill_env_cfg.py — the two must always stay in sync so that
# distillation checkpoint loading succeeds (identical input dimensions).

from . import mdp
from .task_three_env_cfg import (
    BOWL_HOVER_HEIGHT,
    TaskThreeEnvCfg,
)

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


@configclass
class TaskThreeTeacherObservationsCfg:
    """Observations for the teacher policy (full privileged state, no camera).

    Actor == Critic: the teacher has oracle access to all state, so there is no
    need for an asymmetric setup — both use the same 40-dim vector.

    Dims: joint_pos 6 + joint_vel 6 + ee_pos 3
          + red_pos 3 + blue_pos 3 + green_pos 3 + yellow_pos 3
          + bowl_pos 3 + target_color_one_hot 4 + actions 6  =  40.

    These 40 dims MUST match the TeacherCfg in task_three_distill_env_cfg.py.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Teacher actor — full privileged state, no camera."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        red_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_red")},
        )
        blue_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_blue")},
        )
        green_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_green")},
        )
        yellow_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_yellow")},
        )
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        target_color = ObsTerm(func=mdp.target_color_one_hot)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Teacher critic — identical to actor (symmetric; no additional secrets needed)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        red_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_red")},
        )
        blue_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_blue")},
        )
        green_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_green")},
        )
        yellow_cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_yellow")},
        )
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        target_color = ObsTerm(func=mdp.target_color_one_hot)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TaskThreeTeacherEnvCfg(TaskThreeEnvCfg):
    """Task One teacher environment — full privileged state, no camera (Phase 1).

    Inherits the scene, rewards, terminations, events, and curriculum from
    TaskThreeEnvCfg and overrides:
      - Observations: full 30-dim actor with current cube position.
      - cube_visibility reward: active (weight=1.5) so the teacher learns to orient
        the wrist camera toward the cube during the first 20 steps of each episode.
        This bakes "look first" behaviour into the teacher trajectory, which the
        camera student then imitates through distillation.
      - action_rate / joint_vel: tightened 10× for smoother teacher trajectories
        → smoother BC targets for the student.
    """

    observations: TaskThreeTeacherObservationsCfg = TaskThreeTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Tighter action smoothing: teacher trajectories become the BC targets for
        # the student, so smoother teacher = smoother student.
        self.rewards.action_rate.weight = -5e-5   # 10× tighter than base
        self.rewards.joint_vel.weight = -1e-4     # 10× tighter than base
        # self.rewards.lifting_object.params["saturation_height"] = 0.025
        # self.rewards.lifting_object.weight = 10
