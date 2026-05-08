# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
# import mdp
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
import isaaclab.sim.spawners.shapes as shape_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.pick_place.pick_place_env_cfg import PickPlaceEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm101PickPlaceCubeEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set so arm as robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )

        # Set the body name for the end effector
        self.commands.pick_pose.body_name = ["gripper_link"]
        self.commands.place_pose.body_name = ["gripper_link"]

        # TODO make camera model more accurate
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_link/wrist_cam",
            update_period=0.1,
            height=72,
            width=128,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=21,
                clipping_range=(0.01, 3.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.05, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),  # faces forward — tune if needed
                convention="ros",
            ),
        )

        # Set Cube as object - initial spawn position on table
        # TODO change this to correct places
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.35, 0.0, 0.05], # Initial pos
                  rot=[1, 0, 0, 0]), # orientation Quaternion
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.02, 0.02, 0.02), # Cube size
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Set target place zone - visual marker for where to place
        # TODO set correct goal
        # Bottom disc: 10cm diameter, 3mm thick
        self.scene.bowl_bottom = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl/Bottom",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.45, 0.0, 0.0015],  # sits on table, 1.5mm half-height
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CylinderCfg(
                radius=0.05,           # 10cm diameter bottom → 5cm radius
                height=0.003,          # 3mm thick base
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
            ),
        )
        # Reward/command logic references a generic place-zone entity.
        # Use bowl bottom center as the canonical target placement zone.
        self.scene.target_place_zone = self.scene.bowl_bottom

        # Wall ring: tapered from r=5cm (bottom) to r=7.5cm (top), 5cm tall
        # Isaac Sim has no native truncated cone primitive, so approximate with a
        # thin-walled cylinder at the mean radius with a cone visual override, or
        # use 4 box "staves" arranged in a ring.
        # Simplest physics-correct approximation: cylinder wall at mean radius
        self.scene.bowl_wall = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bowl/Wall",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.45, 0.0, 0.025],   # centre of 5cm wall height
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CylinderCfg(
                radius=0.0625,         # mean radius: (5 + 7.5) / 2 = 6.25cm
                height=0.05,
                rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, -0.09, 0.01],
                    ),
                ),
            ],
        )


@configclass
class SoArm101PickPlaceCubeEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False