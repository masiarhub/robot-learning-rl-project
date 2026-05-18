# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Direct camera PPO environment for Task 1 (alternative pipeline).
#
# Trains from scratch with PPO using:
#   Actor  (policy): proprioception + bowl position + ResNet18 wrist image — no explicit cube pos.
#   Critic (critic): proprioception + current cube position + bowl position (privileged).
#
# This is a fully self-contained asymmetric AC environment.  No teacher is required
# and no distillation is involved; the agent must discover cube-finding behaviour
# purely from visual experience.
#
# Actor dims : 536  (joint_pos 6 + joint_vel 6 + ee_pos 3 + bowl_pos 3 + ResNet18 512 + actions 6)
# Critic dims: 27   (joint_pos 6 + joint_vel 6 + ee_pos 3 + obj_pos 3 + bowl_pos 3 + actions 6)

from dataclasses import MISSING
import math

from . import mdp
from ._colors import CUBE_BASE_COLOR, BOWL_BASE_COLOR, TABLE_BASE_COLOR, GRIPPER_BASE_COLOR
from .task_one_distill_env_cfg import ObjectTableCameraSceneCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# Height above the bowl centre used as the goal position (observation + rewards).
BOWL_HOVER_HEIGHT: float = 0.12


##
# Actions
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


##
# Observations (direct camera PPO — asymmetric actor-critic)
##


@configclass
class CamPPOObservationsCfg:
    """Asymmetric actor-critic observations for direct PPO training with a wrist camera.

    Actor  (policy): proprioception + bowl position + ResNet18 wrist image (512 dims)
                     — no explicit cube position; inferred from camera at runtime.
    Critic (critic): proprioception + current cube position + bowl position (privileged)
                     — compact state, no camera.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — camera replaces cube state."""

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
    class CriticCfg(ObsGroup):
        """Critic observations — privileged state with current cube position only."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# Rewards
##


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.15}, weight=1.0)

    gripper_aperture = RewTerm(
        func=mdp.gripper_aperture_reward,
        params={
            "std": 0.05,
            "saturation_pos":0.15,
            "cube_sensor_cfg":
    SceneEntityCfg("contact_forces_cube"),
        },
        weight=2.0, 
    )

    object_grasped = RewTerm(
        func=mdp.object_grasped_contact_continuous,
        params={
            "force_saturation": 5.0,
            "force_balance_ratio": 3.0,
            "debug_print_interval": 50,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=10.0,
    )

    # lifting_object = RewTerm(
    #     func=mdp.object_is_lifted,
    #     params={"start_height": 0.015, "saturation_height": 0.02, "min_reward": 0.0},
    #     weight=15,
    # )
    lifting_object = RewTerm(
        func=mdp.lifting_object_grasped,
        params={
            "start_height": 0.012,
            "saturation_height": 0.02,
            "force_saturation": 5.0,
            "force_balance_ratio": 3.0,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=15,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.3, "minimal_height": 0.05, "height_offset": BOWL_HOVER_HEIGHT, "debug_vis": False},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.1, "minimal_height": 0.06, "height_offset": BOWL_HOVER_HEIGHT},
        weight=10.0,
    )

    cube_in_bowl = RewTerm(
        func=mdp.cube_in_bowl,
        params={
            "xy_threshold": 0.055,
            "z_max": 0.04,
            "z_min": -0.00,
            "consecutive_steps": 5,
            "ee_min_height_above_bowl": 0.055,
            "bowl_cfg": SceneEntityCfg("bowl"),
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.0,
    )

    cube_lifted_pct = RewTerm(
        func=mdp.log_cube_lifted_pct,
        params={"min_height": 0.03},
        weight=1e-9,
    )

    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.001)

    object_drop_penalty = RewTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
        weight=-0.5,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    robot_body_cube_contact = RewTerm(
        func=mdp.robot_body_cube_contact_penalty,
        params={"threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces_cube_body")},
        weight=-3.0,
    )

    robot_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces_table")},
        weight=-2.0,
    )

    robot_bowl_contact = RewTerm(
        func=mdp.robot_bowl_contact_penalty,
        params={"threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces_bowl")},
        weight=-0.0,
    )


    # Visibility reward: keeps the cube centred in the wrist camera during the first
    # max_steps of each episode.  Weight=0 here; override to a positive value in
    # task_one_teacher_env_cfg.py (teacher) and task_one_cam_ppo_env_cfg.py (direct PPO).
    # cube_visibility = RewTerm(
    #     func=mdp.cube_initial_visibility_reward,
    #     params={"max_steps": 20, "std_offset": 0.5},
    #     weight=0.0,
    # )


##
# Terminations
##


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    task_success = DoneTerm(
        func=mdp.cube_placed_in_bowl,
        params={
            "xy_threshold": 0.055,
            "z_max": 0.04,
            "consecutive_steps": 5,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
    )


##
# Events
##


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_bowl_nominal_color = EventTerm(
        func=mdp.set_bowl_color,
        mode="startup",
        params={"color": BOWL_BASE_COLOR},
    )

    # randomize_cube_color = EventTerm(
    #     func=mdp.randomize_cube_color,
    #     mode="reset",
    #     params={
    #         "base_color": CUBE_BASE_COLOR,
    #         "delta": (0.08, 0.03, 0.04),
    #     },
    # )

    # randomize_table_color = EventTerm(
    #     func=mdp.randomize_table_color,
    #     mode="reset",
    #     params={
    #         "base_color": TABLE_BASE_COLOR,
    #         "delta": (0.05, 0.05, 0.05),
    #     },
    # )

    # randomize_gripper_color = EventTerm(
    #     func=mdp.randomize_gripper_color,
    #     mode="reset",
    #     params={
    #         "base_color": GRIPPER_BASE_COLOR,
    #         "delta": (0.03, 0.03, 0.03),
    #     },
    # )

    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        is_global_time=True,
        params={
            "prim_path": "/World/light",
            "intensity_range": (300.0, 800.0),
            "color_range": (0.65, 0.85),
        },
    )

    # # DistantLight: primary shadow caster with randomized direction each episode.
    # randomize_distant_light = EventTerm(
    #     func=mdp.randomize_distant_light,
    #     mode="reset",
    #     is_global_time=True,
    #     params={
    #         "prim_path": "/World/DistantLight",
    #         "intensity_range": (1500.0, 3500.0),
    #         "angle_range": (30.0, 70.0),
    #         "azimuth_range": (0.0, 360.0),
    #     },
    # )

    randomize_sphere_light = EventTerm(
        func=mdp.randomize_sphere_light,
        mode="reset",
        params={
            "intensity_range": (1000.0, 4000.0),
            "color_range": (0.6, 0.9),
            "radius_range": (0.15, 0.35),
            "pos_x_range": (-0.3, 0.6),
            "pos_y_range": (-0.5, 0.5),
            "pos_z_range": (0.6, 1.4),
        },
    )

    # -- Wrist camera domain randomization (sim-to-real) --

    # randomize_camera_intrinsics = EventTerm(
    #     func=mdp.randomize_wrist_camera_intrinsics,
    #     mode="reset",
    #     params={
    #         "prim_path": "/World/envs/env_.*/Robot/gripper_link/wrist_camera",
    #         "focal_length_noise_pct": 0.10,
    #     },
    # )

    # randomize_camera_extrinsics = EventTerm(
    #     func=mdp.randomize_wrist_camera_extrinsics,
    #     mode="reset",
    #     params={
    #         "prim_path": "/World/envs/env_.*/Robot/gripper_link/wrist_camera",
    #         "position_noise_m": (-0.002, 0.002),
    #         "rotation_noise_deg": (-2.0, 2.0),
    #     },
    # )

    reset_bowl_and_cube = EventTerm(
        func=mdp.reset_bowl_and_cube,
        mode="reset",
        params={
            "placement_point": (0.048, 0.0),
            "bowl_dist_range": (0.20, 0.40),
            "bowl_x_min": 0.148,
            "bowl_y_max": 0.20,
            "bowl_radius": 0.12,
            "cube_dist_range": (0.15, 0.30),
            "cube_x_min": 0.148,
            "cube_y_max": 0.20,
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
            "cube_z_rotation_range": (0.0, 2.0 * math.pi),
        },
    )

    randomize_arm_joint_angles = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            ),
            "position_range": (-0.02, 0.02),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_gripper_joint_angle = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "position_range": (-0.01, 0.01),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Randomize gripper contact friction — covers different gripper surface conditions.
    # Targets the two contact links: fixed jaw and moving jaw.
    # randomize_gripper_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]),
    #         "static_friction_range": (0.4, 1),
    #         "dynamic_friction_range": (0.3, 0.9),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #         "make_consistent": True,
    #     },
    # )

    # Randomize cube mass ±30% — covers different real cube materials and sizes.
    # randomize_object_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "mass_distribution_params": (0.7, 1.3),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # Randomize table surface friction — covers different real table surface conditions.
    # randomize_table_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("table"),
    #         "static_friction_range": (0.3, 0.8),
    #         "dynamic_friction_range": (0.2, 0.7),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #         "make_consistent": True,
    #     },
    # )


##
# Curriculum
##


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    robot_bowl_contact = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "robot_bowl_contact", "weight": -0.2, "num_steps": 24_000},
    )
    cube_in_bowl = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "cube_in_bowl", "weight": 5000.0, "num_steps": 80_000},
    )





##
# Environment configuration
##


@configclass
class TaskOneCamPPOEnvCfg(ManagerBasedRLEnvCfg):
    """Task One with wrist camera and asymmetric actor-critic for direct PPO training.

    Fully standalone — does not inherit from TaskOneEnvCfg so all settings can be
    tuned here independently of the no-camera variants.

    No teacher or distillation involved; the agent learns purely from visual feedback
    and the privileged critic provides a better value baseline during training only.
    """

    scene: ObjectTableCameraSceneCfg = ObjectTableCameraSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: CamPPOObservationsCfg = CamPPOObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 2
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.dome_light_upper_lower_strategy = 4
        self.sim.render.rendering_mode = "quality"
