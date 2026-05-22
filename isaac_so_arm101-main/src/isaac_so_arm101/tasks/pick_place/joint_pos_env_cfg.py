# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.pick_place.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaac_so_arm101.tasks.pick_place.pick_place_env_cfg import PickPlaceEnvCfg, ObservationsCfg
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.bowl import RL_BOWL_CFG

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
import numpy as np

_WRIST_CAM_ROT: tuple = tuple(
    float(x) for x in euler_angles_to_quat(
        np.array([-35.31, 0.0, 0.0]), degrees=True
    )
)


@configclass
class SoArm101PickPlaceCubeEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ── Robot ────────────────────────────────────────────────────────
        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan":  0.0,
                    "shoulder_lift": -0.4,
                    "elbow_flex":    -0.3,
                    "wrist_flex":    1.57,
                    "wrist_roll":    -1.57,
                    "gripper":       0.3,
                },
                joint_vel={".*": 0.0},
            ),
            spawn=SO_ARM101_CFG.spawn.replace(
                activate_contact_sensors=True,  # ← add this
            ),
        )

        # ── Robot visual material (matte black PLA) ───────────────────────
        self.scene.robot.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.02, 0.02, 0.02),
            metallic=0.0,
            roughness=0.9,
        )

        # ── Actions ──────────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale= 0.3,           # was 0.3
            use_default_offset=True,
            clip={"gripper": (-0.1, 0.5)},  # physical gripper limits
        )

        # ── Wrist camera ──────────────────────────────────────────────────
        self.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
            update_period=0.0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=9.8,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 5.0),
            ),
            width=32, height=18,
            offset=CameraCfg.OffsetCfg(
                pos=(-0.0049, 0.0498, -0.0591),
                rot=(0.9537, -0.3035, 0.0, 0.0),
                convention="opengl",
            ),
        )

        # ── Object (cube) ─────────────────────────────────────────────────
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.01], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                activate_contact_sensors=True,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_angular_velocity=100.0,
                    max_linear_velocity=10.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )

        # ── Sphere light ──────────────────────────────────────────────────
        self.scene.sphere_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/SphereLight",
            spawn=sim_utils.SphereLightCfg(
                intensity=5000.0,
                radius=0.2,
                color=(0.8, 0.8, 0.8),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 0.6)),
        )

        # ── Bowl ──────────────────────────────────────────────────────────
        self.scene.bowl_bottom = RL_BOWL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/BowlBottom",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.45, 0.0, 0.003),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ── End-effector frame transformer ───────────────────────────────
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[-0.015, 0.0, 0.0]),
                ),
            ]
        )


@configclass
class SoArm101PickPlaceCubeEnvCfg_PLAY(SoArm101PickPlaceCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101PickPlaceCubeEnvCfg_Vision(SoArm101PickPlaceCubeEnvCfg):
    """Vision-based observations (wrist cam image + target pos + robot config)."""
    def __post_init__(self):
        super().__post_init__()


@configclass
class SoArm101PickPlaceCubeEnvCfg_State(SoArm101PickPlaceCubeEnvCfg):
    """State-based observations (cube pos + target pos + robot config)."""
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = ObservationsCfg.StatePolicyCfg()
        self.scene.wrist_cam = None
