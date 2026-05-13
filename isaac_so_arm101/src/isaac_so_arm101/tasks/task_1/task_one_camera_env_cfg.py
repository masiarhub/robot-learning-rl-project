# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils

from . import mdp
from .task_one_env_cfg import (
    BOWL_HOVER_HEIGHT,
    ObjectTableSceneCfg,
    ObservationsCfg,
    TaskOneEnvCfg,
)

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from isaacsim.core.utils.rotations import euler_angles_to_quat
import numpy as np

_WRIST_CAM_ROT: tuple = tuple(
    float(x) for x in euler_angles_to_quat(
        np.array([-35.31, 0.0, 0.0]), degrees=True
    )
)

##
# Scene definition (camera variant)
##


@configclass
class ObjectTableCameraSceneCfg(ObjectTableSceneCfg):
    """Extends the base task-one scene with a wrist-mounted RGB camera."""

    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.1,
        height=72,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=9.8,
            focus_distance=0.05,
            f_stop=100,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.0049, 0.0498, -0.0591),
            rot=_WRIST_CAM_ROT,
            convention="opengl",
        ),
    )


##
# Observations (camera variant)
##


@configclass
class CameraObservationsCfg:
    """Observation specifications for the MDP (camera variant)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — includes wrist camera features."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_link_position = ObsTerm(
            func=mdp.gripper_link_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=["gripper_link"])},
        )
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
    class CriticCfg(ObsGroup):
        """Critic observations — privileged state, no camera."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Environment configuration (camera variant)
##


@configclass
class TaskOneCameraEnvCfg(TaskOneEnvCfg):
    """Configuration for the task one environment with wrist camera."""

    scene: ObjectTableCameraSceneCfg = ObjectTableCameraSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CameraObservationsCfg = CameraObservationsCfg()
