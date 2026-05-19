# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Distillation environment for Task 1 (Phase 2).
#
# The camera-based student observes wrist RGB (ResNet18 → 512 dims) + proprioception.
# The teacher (loaded from a Phase 1 teacher checkpoint) receives the same 40-dim
# privileged state vector that was used as its actor observations during Phase 1
# training — this is mandatory for the checkpoint weights to load correctly.
#
# Student actor dims : 536  (joint_pos 6 + joint_vel 6 + ee_pos 3 +
#                             bowl_pos 3 + ResNet18 512 + actions 6)
# Teacher input dims : 40   (joint_pos 6 + joint_vel 6 + ee_pos 3
#                             + red 3 + blue 3 + green 3 + yellow 3
#                             + bowl_pos 3 + target_one_hot 4 + actions 6)
#
# The TeacherCfg below MUST stay in sync with TaskThreeTeacherObservationsCfg.PolicyCfg
# in task_three_teacher_env_cfg.py (40 dims).

import isaaclab.sim as sim_utils

from . import mdp
from ._colors import BOWL_BASE_COLOR, CUBE_BASE_COLOR, TABLE_BASE_COLOR, GRIPPER_BASE_COLOR
from .task_three_env_cfg import (
    BOWL_HOVER_HEIGHT,
    EventCfg,
    ObjectTableSceneCfg,
    TaskThreeEnvCfg,
)

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ._wrist_cam import (
    FOCAL_LENGTH_MM as _CAM_FOCAL_MM,
    FOCUS_DISTANCE_M as _CAM_FOCUS_DIST,
    F_STOP as _CAM_FSTOP,
    HORIZONTAL_APERTURE_MM as _CAM_H_APERTURE_MM,
    IMAGE_WIDTH as _CAM_IMG_W,
    IMAGE_HEIGHT as _CAM_IMG_H,
    OFFSET_POS as _CAM_OFFSET_POS,
    OFFSET_QUAT_WXYZ as _CAM_OFFSET_QUAT_WXYZ,
)


##
# Scene definition (camera variant — shared with task_three_cam_ppo_env_cfg.py)
##


@configclass
class ObjectTableCameraSceneCfg(ObjectTableSceneCfg):
    """Extends the base task-one scene with a wrist-mounted RGB camera."""

    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.02,
        height=_CAM_IMG_H,
        width=_CAM_IMG_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_CAM_FOCAL_MM,
            focus_distance=_CAM_FOCUS_DIST,
            f_stop=_CAM_FSTOP,
            horizontal_aperture=_CAM_H_APERTURE_MM,
            clipping_range=(0.01, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=_CAM_OFFSET_POS,
            rot=_CAM_OFFSET_QUAT_WXYZ,
            convention="opengl",
        ),
    )


##
# Observations (distillation variant)
##


@configclass
class DistillObservationsCfg:
    """Observations for the distillation phase.

    Student actor (policy): wrist camera + proprioception, no cube state.
    Teacher input  (critic): 30-dim privileged state matching the Phase 1 teacher
                             actor — identical to TaskThreeTeacherObservationsCfg.PolicyCfg.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Student actor observations — wrist camera replaces cube state."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        wrist_image = ObsTerm(
            func=mdp.wrist_camera_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "flatten": True},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Teacher input — must exactly match TaskThreeTeacherObservationsCfg.PolicyCfg (40 dims).

        joint_pos(6) + joint_vel(6) + ee_pos(3)
        + red_pos(3) + blue_pos(3) + green_pos(3) + yellow_pos(3)
        + bowl_pos(3) + target_one_hot(4) + actions(6) = 40 dims.
        """

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
    # 'critic' attribute name is required — obs_groups routes {"teacher": ["critic"]}
    critic: TeacherCfg = TeacherCfg()


##
# Camera-enabled event config (distillation + post-training)
##


@configclass
class CameraEventCfg(EventCfg):
    """Extends the base EventCfg with color domain randomization.

    Only used by camera-based environments (distillation, post-training, cam-PPO)
    where the wrist image is an observation. Excluded from non-camera envs
    (teacher, phase-1b) to avoid the USD material update overhead at 4096 envs.
    """

    set_bowl_nominal_color = EventTerm(
        func=mdp.set_bowl_color,
        mode="startup",
        params={"color": BOWL_BASE_COLOR},
    )

    randomize_cube_color = EventTerm(
        func=mdp.randomize_cube_color,
        mode="reset",
        params={
            "base_color": CUBE_BASE_COLOR,
            "delta": (0.08, 0.03, 0.04),
        },
    )

    randomize_table_color = EventTerm(
        func=mdp.randomize_table_color,
        mode="reset",
        params={
            "base_color": TABLE_BASE_COLOR,
            "delta": (0.05, 0.05, 0.05),
        },
    )

    randomize_gripper_color = EventTerm(
        func=mdp.randomize_gripper_color,
        mode="reset",
        params={
            "base_color": GRIPPER_BASE_COLOR,
            "delta": (0.03, 0.03, 0.03),
        },
    )


##
# Environment configuration (distillation variant)
##


@configclass
class TaskThreeDistillEnvCfg(TaskThreeEnvCfg):
    """Task One distillation environment — camera student, privileged teacher (Phase 2).

    Inherits scene, rewards, terminations, events, and curriculum from TaskThreeEnvCfg,
    then replaces the scene with a camera-equipped variant and the observations with
    the distillation-specific student/teacher observation groups.

    Distillation trains via behaviour cloning (not RL rewards) — the curriculum is
    disabled and reward weights are set to their final values for monitoring only.
    """

    scene: ObjectTableCameraSceneCfg = ObjectTableCameraSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DistillObservationsCfg = DistillObservationsCfg()
    events: CameraEventCfg = CameraEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Distillation does not use the curriculum.
        self.curriculum = None
        # Set reward weights to final values so they serve as meaningful monitoring signals.
        self.rewards.cube_in_bowl.weight = 2000.0
        self.rewards.robot_bowl_contact.weight = -0.2
        # Higher-quality rendering for the wrist camera (not needed for non-camera envs).
        self.sim.render.samples_per_pixel = 2
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.dome_light_upper_lower_strategy = 4
        self.sim.render.rendering_mode = "quality"
