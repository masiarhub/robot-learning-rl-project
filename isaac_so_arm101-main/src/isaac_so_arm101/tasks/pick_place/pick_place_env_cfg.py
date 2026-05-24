# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    wrist_cam: CameraCfg = MISSING
    sphere_light: AssetBaseCfg = MISSING
    bowl_bottom: RigidObjectCfg = MISSING

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, -0.5]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 1),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
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
    #contact_forces_table = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/.*",
    #    history_length=3,
    #    track_air_time=False,
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],
    #)


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
        bowl_pos = ObsTerm(
            func=mdp.bowl_center_position,
            params={
                "asset_cfg": SceneEntityCfg("bowl_bottom"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StatePolicyCfg(ObsGroup):
        """Actor — initial cube pos only (matches real deployment)."""
        object_pos = ObsTerm(func=mdp.initial_cube_position_in_robot_root_frame)
        bowl_pos = ObsTerm(
            func=mdp.bowl_center_position,
            params={
                "asset_cfg": SceneEntityCfg("bowl_bottom"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        #ee_pos    = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticObsCfg(ObsGroup):
        """Critic — live cube pos (privileged sim info). Always state-based regardless of actor mode."""
        object_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame)  # ← live, only difference
        bowl_pos = ObsTerm(
            func=mdp.bowl_center_position,
            params={
                "asset_cfg": SceneEntityCfg("bowl_bottom"),
                "robot_cfg": SceneEntityCfg("robot"),
            },
        )
        #ee_pos    = ObsTerm(func=mdp.ee_position_in_robot_root_frame)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Switch actor mode here: StatePolicyCfg for state-based, VisionPolicyCfg for visual
    policy: VisionPolicyCfg | StatePolicyCfg = StatePolicyCfg()
    critic: CriticObsCfg = CriticObsCfg()  # always privileged live state


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_bowl_and_cube = EventTerm(
        func=mdp.reset_bowl_and_cube,
        mode="reset",
        params={
            # Placement point = first revolute joint (local frame).
            "placement_point": (0.048, 0.0),
            # Bowl: annular ring [0.20, 0.40] m from placement point, x ≥ 0.148, |y| ≤ 0.20.
            "bowl_dist_range": (0.20, 0.40),
            "bowl_x_min": 0.148,
            "bowl_y_max": 0.20,
            # Bowl radius: keep-out circle + occlusion-cone half-width (wider than physical 0.0775 to account for 3D camera perspective).
            "bowl_radius": 0.1,
            # Cube: annular ring [0.15, 0.30] m from placement point, x ≥ 0.148, |y| ≤ 0.20.
            "cube_dist_range": (0.15, 0.30),
            "cube_x_min": 0.148,
            "cube_y_max": 0.20,
            # Two-phase sampling: 100 random tries, then safety positions fallback.
            "safe_fallback_after": 100,
            "max_placement_tries": 200,
            "safety_positions": [
                (0.268, +0.000),
                (0.253, +0.143),
                (0.253, -0.143),
                (0.293, +0.114),
                (0.293, -0.114),
                (0.338, +0.000),
                (0.189, +0.169),
                (0.189, -0.169),
            ],
            # Randomize cube orientation around z-axis: full 360° range.
            "cube_z_rotation_range": (0.0, 2.0 * math.pi),
        },
    )


    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params":   (0.9, 1.1),
            "operation": "scale", "distribution": "uniform",
        },
    )

    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link","moving_jaw_so101_v1_link"]),
            "static_friction_range": (0.6, 1.0), "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.0), "num_buckets": 16, "make_consistent": True,
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.7, 1.3), "operation": "scale", "distribution": "uniform"},
    )

    randomize_table_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.5, 0.70),
            "dynamic_friction_range": (0.40, 0.60),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )



@configclass
class RewardsCfg:
    #reaching_object_coarse = RewTerm(
    #    func=mdp.object_ee_distance,
    #    params={"std": 0.15},
    #    weight=0.1,
    #)
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.05},  # only rewards when within ~3cm — cube is 2cm
        weight=1.0,
    )

    #gripper_close_near_cube_coarse = RewTerm(
    #    func=mdp.gripper_close_when_near,
    #    params={
    #        "proximity_std": 0.15,
    #        "gripper_target": -0.1,
    #        "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
    #    },
    #    weight=5.0,
    #)
    gripper_close_near_cube = RewTerm(
        func=mdp.gripper_close_when_near,
        params={
            "proximity_std": 0.02,
            "gripper_target": 0.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=5.0,
    )

    #gripper_force_coarse = RewTerm(
    #    func=mdp.gripper_force_reward,
    #    params={
    #        "target_force": 2.0, "force_tolerance": 2.0, "force_max": 5.0,
    #        "proximity_std": 0.05, "debug_print_interval": 0,
    #        "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
    #    },
    #    weight=5.0,
    #)
    #gripper_force_fine = RewTerm(
    #    func=mdp.gripper_force_reward,
    #    params={
    #        "target_force": 0.2,       # was 2.0 — realistic for 5g cube
    #        "force_tolerance": 0.2,    # was 2.0
    #        "force_max": 1.0,          # was 8.0
    #        "proximity_std": 0.05,
    #        "debug_print_interval": 0,
    #        "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
    #    },
    #    weight=10.0,
#)

    object_grasped = RewTerm(
        func=mdp.object_grasped_contact_continuous,
        params={
            "force_saturation": 0.3,       # was 5.0
            "force_balance_ratio": 3.0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=15.0,
    )
    #gripper_aperture = RewTerm(
    #    func=mdp.gripper_aperture_reward,
    #    params={
    #        "std": 0.1,
    #        "close_joint_pos": 0.0,
    #        "target_open_pos": 0.5,
    #        "object_cfg": SceneEntityCfg("object"),
    #        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    #        "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
    #    },
    #    weight=0.01,
    #)

    lifting_object_coarse = RewTerm(
        func=mdp.lifting_object_grasped,
        params={
            "start_height": 0.011,
            "saturation_height": 0.03,
            "force_saturation": 0.3,       # was 5.0
            "force_balance_ratio": 3.0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=30.0,
    )
    lifting_object = RewTerm(
        func=mdp.lifting_object_grasped,
        params={
            "start_height": 0.03,
            "saturation_height": 0.07,
            "force_saturation": 0.5,       # was 5.0
            "force_balance_ratio": 3.0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=20.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.3, "minimal_height": 0.06, "height_offset": 0.1, "debug_vis": True},
        weight=15.0,
    )

    object_goal_tracking_fine = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.05, "minimal_height": 0.07, "height_offset": 0.1},
        weight=10.0,
    )
    dropping_success = RewTerm(
        func=mdp.object_released_in_zone,
        params={"threshold": 0.05, "target_cfg": SceneEntityCfg("bowl_bottom")},
        weight=100.0,
    )
    #cube_moved_before_grasp = RewTerm(
    #    func=mdp.cube_moved_before_grasp_penalty,
    #    params={
    #        "height_threshold": 0.01,
    #        "force_threshold": 0.05,
    #        "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
    #    },
    #    weight=-1.0,
    #)
    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.001)
    #robot_table_contact = RewTerm(
    #    func=mdp.robot_table_contact_penalty,
    #    params={
    #        "threshold": 0.5,
    #        "sensor_cfg": SceneEntityCfg(
    #            "contact_forces_table",
    #            body_names=[
    #                "base_link",
    #                "shoulder_link",
    #                "upper_arm_link",
    #                "lower_arm_link",
    #                "wrist_link",
    #            ],  # excludes gripper_link and moving_jaw_so101_v1_link
    #        ),
    #    },
    #    weight=-1.0,
    #)
    #time_penalty_no_grasp = RewTerm(
    #    func=mdp.time_penalty_if_not_lifted,
    #    params={"start_height": 0.05, "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube")},
    #    weight=-1.0,
    #)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #object_dropping = DoneTerm(
    #    func=mdp.root_height_below_minimum,
    #    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    #)
    object_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("object")}
    )
    cube_placed_in_bowl = DoneTerm(
        func=mdp.cube_placed_in_bowl,
        params={
            "xy_threshold": 0.07,
            "z_max": 0.06,
            "ee_min_height_above_bowl": 0.02,
            "consecutive_steps": 7,
            "bowl_cfg": SceneEntityCfg("bowl_bottom"),
        },
    )

@configclass
class CurriculumCfg:
    pass


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
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
        self.clip_actions = 1.0        # ← ADD THIS
        self.clip_observations = 5.0   # ← ADD THIS
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
