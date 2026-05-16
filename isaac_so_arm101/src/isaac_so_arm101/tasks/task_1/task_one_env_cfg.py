# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



# CUBE DROPPING: Finetuning
# For the best policy up to now, first a policy was trained to reach a goal positon over the bowl
# Then, the CUBE DROPPING reward and termination were activated, the curriculum regularization terms were deactivated
# The last checkpoint of the first policy was used to continue training: (command looks like this)
# python src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-PickPlace-v0 --num_envs 4096 --headless --max_iterations=3000 --resume --load_run=2026-05-01_12-55-26 --checkpoint=model_1499.pt --video

from dataclasses import MISSING

import math
import isaaclab.sim as sim_utils

from . import mdp
from ._colors import TABLE_BASE_COLOR

from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
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
from isaaclab.sensors import ContactSensorCfg

# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # bowl: will be populated by agent env cfg
    bowl: RigidObjectCfg = MISSING

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, -0.5]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_BASE_COLOR),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )

    # # Scene-wide directional light — casts parallel shadows from all objects including the robot arm.
    # # Direction is randomized each episode so shadow angle varies across training.
    # light_distant = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(intensity=2000.0, angle=2.0, color=(1.0, 1.0, 1.0)),
    # )

    # Per-environment sphere light — adds local positional lighting variation per env.
    # treat_as_point=False uses sphere radius for shadow softness (no hard circles).
    # light_local = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/SphereLight",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 0.8)),
    #     spawn=sim_utils.SphereLightCfg(intensity=3000.0, radius=0.2, treat_as_point=False),
    # )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",  # covers ALL robot bodies
        update_period=0.0,          # update every physics step
        history_length=3,           # keep last 3 frames for stability
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],  # only measure contact WITH the cube
    )

    contact_forces_table = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    update_period=0.0,
    history_length=3,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Table"],  # table contact only
    )

    contact_forces_bowl = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    update_period=0.0,
    history_length=3,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Bowl"],
    )


##
# MDP settings
##

# Height above the bowl centre used as the goal position (observation + rewards).
BOWL_HOVER_HEIGHT: float = 0.12


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — asymmetric setup: only the initial cube position is given.

        The actor cannot observe where the cube is at runtime; it only knows where it
        started the episode.  The critic (CriticCfg) retains full privileged state.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        # Only the reset-time cube position — frozen for the episode.
        initial_object_position = ObsTerm(func=mdp.initial_object_position_in_robot_root_frame)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — privileged full state including current and initial cube position."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        initial_object_position = ObsTerm(func=mdp.initial_object_position_in_robot_root_frame)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize joint angles at each reset (arm ±0.02 rad, gripper ±0.01 rad).
    # Split into two terms because reset_joints_by_offset takes a single (low, high) tuple
    # and the gripper uses a tighter range than the arm joints.
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
    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]),
            "static_friction_range": (0.4, 1),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # Randomize cube mass ±30% — covers different real cube materials and sizes.
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize table surface friction — covers different real table surface conditions.
    randomize_table_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.3, 0.8),
            "dynamic_friction_range": (0.2, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # Dome: dark-area fill only — kept very low so DistantLight shadows remain visible.
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

    # SphereLight: per-env local accent light for additional positional variation.
    # treat_as_point=False in the spawn config means radius controls shadow softness.
    # randomize_sphere_light = EventTerm(
    #     func=mdp.randomize_sphere_light,
    #     mode="reset",
    #     params={
    #         "intensity_range": (1000.0, 4000.0),
    #         "color_range": (0.6, 0.9),
    #         "radius_range": (0.15, 0.35),
    #         "pos_x_range": (-0.3, 0.6),
    #         "pos_y_range": (-0.5, 0.5),
    #         "pos_z_range": (0.6, 1.4),
    #     },
    # )


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
            "bowl_radius": 0.14,
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
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.15}, weight=1.0)

    # object_grasped = RewTerm(
    #     func=mdp.gripper_closed_near_object,
    #     params={"std": 0.015},
    #     weight=2.0,
    # )

    # for general teacher without camera cube pos only
    # lifting_object = RewTerm(
    #     func=mdp.object_is_lifted,
    #     params={"start_height": 0.015, "saturation_height": 0.025, "min_reward": 0.0},
    #     weight=10,
    # )

    # for initial cube pos only
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"start_height": 0.015, "saturation_height": 0.02, "min_reward": 0.0},
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

    # Sparse success reward: cube inside the bowl and gripper open.
    # Starts at 0 and is ramped up by curriculum — avoids overwhelming early exploration.
    cube_in_bowl = RewTerm(
        func=mdp.cube_in_bowl,
        params={
            "xy_threshold": 0.055,
            "z_max": 0.04,
            "z_min": -0.00,
            "consecutive_steps": 5,
            "ee_min_height_above_bowl": 0.07,
            "bowl_cfg": SceneEntityCfg("bowl"),
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.0,
    )

    # time penalty: -0.001 per step encourages faster task completion
    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.001)

    # cube dropping penalty
    object_drop_penalty = RewTerm(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    weight=-0.5,
    )

    ### REGULARZATION
    # action penalty (regularization)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)

    # joint velocity penalty (regularization)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Dont' use for now (no obvious improvement)
    # joint_acc = RewTerm(
    # func=mdp.joint_acc_l2,
    # weight=-5e-6,  # usually smaller magnitude than vel penalty
    # params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # torque = RewTerm(
    # func=mdp.joint_torques_l2,
    # weight=-1e-4,
    # params={"asset_cfg": SceneEntityCfg("robot")},
    # )


    robot_body_cube_contact = RewTerm(
    func=mdp.robot_body_cube_contact_penalty,
    params={
        "threshold": 0.5,
        "sensor_cfg": SceneEntityCfg("contact_forces"),
        "robot_cfg": SceneEntityCfg(
            "robot",
            body_names=[
                "base_link",
                "shoulder_link",
                "upper_arm_link",
                "lower_arm_link",
                "wrist_link",
            ],
        ),
    },
    weight=-1.0,
    )

    robot_table_contact = RewTerm(
    func=mdp.robot_table_contact_penalty,
    params={
        "threshold": 1.0,
        "sensor_cfg": SceneEntityCfg("contact_forces_table"),
        "robot_cfg": SceneEntityCfg(
            "robot",
            body_names=[
                "shoulder_link",
                "upper_arm_link",
                "lower_arm_link",
                "wrist_link",
                # "gripper_link",
                # "moving_jaw_so101_v1_link",
            ],
        ),
    },
    weight=-2.0,
)
    # could be useful: penalize robot - bowl contacts
    robot_bowl_contact = RewTerm(
    func=mdp.robot_bowl_contact_penalty,
    params={
        "threshold": 0.5,
        "sensor_cfg": SceneEntityCfg("contact_forces_bowl"),
        "robot_cfg": SceneEntityCfg(
            "robot",
            body_names=[
                "shoulder_link",
                "upper_arm_link",
                "lower_arm_link",
                "wrist_link",
                "gripper_link",
                "moving_jaw_so101_v1_link",
            ],
        ),
    },
    weight=-0.0,
)

    # Visibility reward: keeps the cube centred in the wrist camera during the first
    # max_steps of each episode.  Weight=0 here; override to a positive value in
    # task_one_teacher_env_cfg.py (teacher) and task_one_cam_ppo_env_cfg.py (direct PPO).
    cube_visibility = RewTerm(
        func=mdp.cube_initial_visibility_reward,
        params={"max_steps": 20, "std_offset": 0.5},
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # CUBE DROPPING: Finetuning
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
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


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Ramp in the sparse success reward once the agent has had time to learn lifting/tracking.
    # Starts at weight=0 (see RewardsCfg) and reaches 500 after ~2000 iterations (4096 envs × 500 steps × 2000 = 4M steps).
    robot_bowl_contact = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "robot_bowl_contact", "weight": -0.2, "num_steps": 12_000}
    )
    cube_in_bowl = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "cube_in_bowl", "weight": 2000.0, "num_steps": 36_000}
    )
    # just for testing purposes -> what happens if cube in bowl is active frm the start (tested without robot_bowl_contact)
    # result: works, reaches 90% SR after 1000 iterations, however less nice (very fast and close to bowl)
    # cube_in_bowl = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "cube_in_bowl", "weight": 5000.0, "num_steps": 0}
    # )





##
# Environment configuration
##


@configclass
class TaskOneEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the task one environment (no camera)."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2 # TODO maybe try to change to 5, 2 is quite fast for manipulation
        self.episode_length_s = 8.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01  # low threshold suits the small 2 cm cube
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.render.antialiasing_mode = "DLAA"          # instead of default DLSS in many modes
        self.sim.render.enable_dl_denoiser = True           # higher-quality denoiser for low res
