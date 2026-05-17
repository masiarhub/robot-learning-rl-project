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
from pathlib import Path

from . import mdp
from ._colors import CUBE_BASE_COLOR
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401
from .task_two_env_cfg import TaskTwoEnvCfg
from .task_two_teacher_env_cfg import TaskTwoTeacherEnvCfg
from .task_two_distill_env_cfg import TaskTwoDistillEnvCfg
from .task_two_post_train_env_cfg import TaskTwoPostTrainEnvCfg
from .task_two_cam_ppo_env_cfg import TaskTwoCamPPOEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

# Relative path to the bowl USD asset (resolved from this file's location)
_BOWL_USD_PATH = str(Path(__file__).resolve().parent.parent.parent / "robots" / "rl_bowl" / "bowl_scaled.usd")


def _setup_soarm101(cfg) -> None:
    """Apply SO-ARM101-specific robot, action, object, bowl, and EE-frame config."""
    cfg.scene.robot = SO_ARM101_CFG.replace(
        spawn=SO_ARM101_CFG.spawn.replace(
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(1.0, 0.0, 0.0, 0.0),
            # old (unstable) initial position
            # joint_pos={
            #     "shoulder_pan": 0.0,
            #     "shoulder_lift": -0.4,
            #     "elbow_flex": -0.3,
            #     "wrist_flex": 1.57,
            #     "wrist_roll": -1.57,
            #     "gripper": 0.2,
            # },
            # joint_vel={".*": 0.0},
            # new position
            # joint_pos={
            #     "shoulder_pan": 0.0,
            #     "shoulder_lift": -1.6,
            #     "elbow_flex": 0.4,
            #     "wrist_flex": 1.57,
            #     "wrist_roll": -1.57,
            #     "gripper": 0.2,
            # },
            # joint_vel={".*": 0.0},
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": -1.4,
                "elbow_flex": 0.4,
                "wrist_flex": 1.4,
                "wrist_roll": -1.57,
                "gripper": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
    )
    cfg.actions.arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    cfg.actions.gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.3,
        use_default_offset=True,
    )

    cfg.scene.object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.01], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_BASE_COLOR),
        ),
    )

    cfg.scene.bowl = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bowl",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.30, 0.0, 0.01], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=_BOWL_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
    ),
)

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    cfg.scene.ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
            ),
        ],
    )


##
# Phase 1a — teacher (full privileged state, no camera)
##


@configclass
class SoArm101TaskTwoTeacherEnvCfg(TaskTwoTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _setup_soarm101(self)


@configclass
class SoArm101TaskTwoTeacherEnvCfg_PLAY(SoArm101TaskTwoTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.rewards.debug_grasp = RewTerm(
            func=mdp.debug_grasp_state,
            params={
                "print_interval": 50,
                "lift_start_height": 0.015,
                "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
                "robot_gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
            weight=1e-9,
        )


##
# Phase 1b — initial cube state policy (no camera, no current cube pos)
##


@configclass
class SoArm101TaskTwoEnvCfg(TaskTwoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _setup_soarm101(self)


@configclass
class SoArm101TaskTwoEnvCfg_PLAY(SoArm101TaskTwoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# Phase 2 — distillation (camera student + privileged teacher)
##


@configclass
class SoArm101TaskTwoDistillEnvCfg(TaskTwoDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _setup_soarm101(self)


@configclass
class SoArm101TaskTwoDistillEnvCfg_PLAY(SoArm101TaskTwoDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# Phase 3 — post-training RL fine-tune of distilled student
##


@configclass
class SoArm101TaskTwoPostTrainEnvCfg(TaskTwoPostTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _setup_soarm101(self)


@configclass
class SoArm101TaskTwoPostTrainEnvCfg_PLAY(SoArm101TaskTwoPostTrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


##
# Alternative — direct camera PPO (asymmetric AC, no distillation)
##


@configclass
class SoArm101TaskTwoCamPPOEnvCfg(TaskTwoCamPPOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _setup_soarm101(self)


@configclass
class SoArm101TaskTwoCamPPOEnvCfg_PLAY(SoArm101TaskTwoCamPPOEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
