# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import isaac_so_arm101.tasks.pick_place.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sensors import ContactSensorCfg


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    wrist_cam: CameraCfg = MISSING
    sphere_light: AssetBaseCfg = MISSING
    bowl_bottom: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            visual_material=PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),
                roughness=0.8,
                metallic=0.0,
            ),
        ),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )

    contact_forces_cube = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
    )

    # Robot-side table collision sensor: upper_arm_link is the most likely
    # link to hit the table. Any contact here is penalised.
    contact_forces_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    )


@configclass
class CommandsCfg:
    pass


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:

    @configclass
    class VisionPolicyCfg(ObsGroup):
        wrist_image = ObsTerm(func=mdp.wrist_camera_image, params={"sensor_cfg": SceneEntityCfg("wrist_cam")})
        bowl_pos    = ObsTerm(func=mdp.bowl_center_position, params={"asset_cfg": SceneEntityCfg("bowl_bottom")})
        joint_pos   = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel   = ObsTerm(func=mdp.joint_vel_rel)
        actions     = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StatePolicyCfg(ObsGroup):
        object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})
        bowl_pos   = ObsTerm(func=mdp.bowl_center_position, params={"asset_cfg": SceneEntityCfg("bowl_bottom")})
        joint_pos  = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel  = ObsTerm(func=mdp.joint_vel_rel)
        actions    = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: VisionPolicyCfg | StatePolicyCfg = VisionPolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_bowl_and_object = EventTerm(
        func=mdp.reset_bowl_and_object_non_overlapping,
        mode="reset",
        params={
            "bowl_cfg":        SceneEntityCfg("bowl_bottom"),
            "object_cfg":      SceneEntityCfg("object", body_names="Object"),
            "bowl_xy_range":   {"x": (0.35, 0.55), "y": (-0.20, 0.20)},
            "object_xy_range": {"x": (0.25, 0.45), "y": (-0.20, 0.20)},
            "min_xy_distance": 0.13,
        },
    )

    randomize_object_orientation = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (0.0,0.0),"y": (0.0,0.0),"z": (0.0,0.0),"roll": (0.0,0.0),"pitch": (0.0,0.0),"yaw": (-3.14159,3.14159)},
            "velocity_range": {},
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params":   (0.75, 1.25),
            "operation": "scale", "distribution": "uniform",
        },
    )

    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link","moving_jaw_so101_v1_link"]),
            "static_friction_range": (0.4, 1.2), "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.0), "num_buckets": 16, "make_consistent": True,
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.4, 1.6), "operation": "scale", "distribution": "uniform"},
    )

    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 1.1), "dynamic_friction_range": (0.15, 0.85),
            "restitution_range": (0.0, 0.05), "num_buckets": 8,
        },
    )

    randomize_table_friction = EventTerm(
        func=mdp.randomize_table_friction,
        mode="reset",
        params={"static_friction_range": (0.25, 0.90), "dynamic_friction_range": (0.18, 0.70)},
    )

    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="reset",
        params={"intensity_range": (300.0, 1400.0), "color_range": (0.60, 0.90)},
    )

    randomize_sphere_light = EventTerm(
        func=mdp.randomize_sphere_light,
        mode="reset",
        params={
            "intensity_range": (2000.0, 9000.0), "color_range": (0.60, 0.90),
            "radius_range": (0.1, 0.5), "pos_x_range": (-0.3, 0.6),
            "pos_y_range": (-0.5, 0.5), "pos_z_range": (0.25, 1.1),
        },
    )


@configclass
class RewardsCfg:
    reaching_object_coarse = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.15},
        weight=0.5,
    )
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.03},  # only rewards when within ~3cm — cube is 2cm
        weight=2.0,
    )

    gripper_close_near_cube_coarse = RewTerm(
        func=mdp.gripper_close_when_near,
        params={
            "proximity_std": 0.15,
            "gripper_target": -0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=5.0,
    )
    gripper_close_near_cube = RewTerm(
        func=mdp.gripper_close_when_near,
        params={
            "proximity_std": 0.03,
            "gripper_target": -0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=25.0,
    )

    gripper_force_coarse = RewTerm(
        func=mdp.gripper_force_reward,
        params={
            "target_force": 2.0, "force_tolerance": 2.0, "force_max": 8.0,
            "proximity_std": 0.15, "debug_print_interval": 0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=15.0,
    )
    gripper_force_fine = RewTerm(
        func=mdp.gripper_force_reward,
        params={
            "target_force": 3.0, "force_tolerance": 1.5, "force_max": 8.0,
            "proximity_std": 0.03, "debug_print_interval": 0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=75.0,
    )

    object_grasped = RewTerm(
        func=mdp.object_grasped_contact_continuous,
        params={"force_saturation": 5.0, "force_balance_ratio": 3.0,
                "debug_print_interval": 0,
                "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube")},
        weight=100.0,
    )

    lifting_object_coarse = RewTerm(
        func=mdp.lifting_object_grasped,
        params={"start_height": 0.012, "saturation_height": 0.05,
                "force_saturation": 5.0, "force_balance_ratio": 3.0,
                "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube")},
        weight=8.0,
    )
    lifting_object_fine = RewTerm(
        func=mdp.lifting_object_grasped,
        params={"start_height": 0.012, "saturation_height": 0.025,
                "force_saturation": 5.0, "force_balance_ratio": 3.0,
                "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube")},
        weight=40.0,
    )
    object_to_bowl_coarse = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.25, "minimal_height": 0.07, "bowl_cfg": SceneEntityCfg("bowl_bottom")},
        weight=6.0,
    )
    object_to_bowl_fine = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.05, "minimal_height": 0.07, "bowl_cfg": SceneEntityCfg("bowl_bottom")},
        weight=8.0,
    )
    dropping_success = RewTerm(
        func=mdp.object_released_in_zone,
        params={"threshold": 0.05, "target_cfg": SceneEntityCfg("bowl_bottom")},
        weight=100.0,
    )
    robot_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        params={
            "threshold": 0.5,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces_table",
                body_names=[
                    "base_link",
                    "shoulder_link",
                    "upper_arm_link",
                    "lower_arm_link",
                    "wrist_link",
                ],  # excludes gripper_link and moving_jaw_so101_v1_link
            ),
        },
        weight=-1.0,
    )
    time_penalty_no_grasp = RewTerm(
        func=mdp.time_penalty_if_not_lifted,
        params={"start_height": 0.05, "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube")},
        weight=-1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})
    object_out_of_bounds = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("object")})
    no_progress = DoneTerm(
        func=mdp.no_task_progress,
        params={
            "window_steps": 100,   # ~1s before termination
            "min_progress": 0.001,
        },
    )

@configclass
class CurriculumCfg:
    pass


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene:        ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg    = ObservationsCfg()
    actions:      ActionsCfg         = ActionsCfg()
    commands:     CommandsCfg        = CommandsCfg()
    rewards:      RewardsCfg         = RewardsCfg()
    terminations: TerminationsCfg    = TerminationsCfg()
    events:       EventCfg           = EventCfg()
    curriculum:   CurriculumCfg      = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
