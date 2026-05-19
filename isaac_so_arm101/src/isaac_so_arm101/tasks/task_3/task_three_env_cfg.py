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
    # red target cube: will be populated by agent env cfg
    object_red: RigidObjectCfg | DeformableObjectCfg = MISSING
    # blue target cube: will be populated by agent env cfg
    object_blue: RigidObjectCfg | DeformableObjectCfg = MISSING
    # green target cube: will be populated by agent env cfg
    object_green: RigidObjectCfg | DeformableObjectCfg = MISSING
    # yellow target cube: will be populated by agent env cfg
    object_yellow: RigidObjectCfg | DeformableObjectCfg = MISSING
    # bowl: will be populated by agent env cfg
    bowl: RigidObjectCfg = MISSING

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, -0.5]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 1),
            activate_contact_sensors=True,
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

    # All contact sensors sit on the non-articulation (object/table/bowl) side and filter
    # for specific robot links. force_matrix_w_history only works when the sensor body is a
    # plain RigidObject — articulation links as the sensor body always give all-zero matrices.

    # Grasp reward: red cube feels force from each finger.
    contact_forces_cube_red = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
    )

    # Grasp reward: blue cube feels force from each finger.
    contact_forces_cube_blue = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ObjectBlue",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
    )

    # Grasp reward: green cube feels force from each finger.
    contact_forces_cube_green = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ObjectGreen",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
    )

    # Grasp reward: yellow cube feels force from each finger.
    contact_forces_cube_yellow = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/ObjectYellow",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
    )

    # Penalty: red cube feels force from non-gripper arm links (bad — arm body touching cube).
    contact_forces_cube_red_body = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/base_link",
            "{ENV_REGEX_NS}/Robot/shoulder_link",
            "{ENV_REGEX_NS}/Robot/upper_arm_link",
            "{ENV_REGEX_NS}/Robot/lower_arm_link",
            "{ENV_REGEX_NS}/Robot/wrist_link",
        ],
    )

    # Penalty: table feels force from upper-arm links (base_link never reaches the table).
    contact_forces_table = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/shoulder_link",
            "{ENV_REGEX_NS}/Robot/upper_arm_link",
            "{ENV_REGEX_NS}/Robot/lower_arm_link",
            "{ENV_REGEX_NS}/Robot/wrist_link",
        ],
    )

    # Penalty: bowl feels force from any robot link (gripper included — must release before bowl).
    # Uses /Bowl/.* to match the rigid body sub-prim inside the USD asset (the root /Bowl
    # prim is an Xform and does not carry PhysxContactReportAPI directly).
    contact_forces_bowl = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Bowl/.*",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/shoulder_link",
            "{ENV_REGEX_NS}/Robot/upper_arm_link",
            "{ENV_REGEX_NS}/Robot/lower_arm_link",
            "{ENV_REGEX_NS}/Robot/wrist_link",
            "{ENV_REGEX_NS}/Robot/gripper_link",
            "{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        ],
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
        """Actor observations — initial (reset-time) positions of all four cubes + target color.

        The actor cannot observe current cube positions; it only knows where they
        started the episode and which color to pick up.
        Dims: joint_pos 6 + joint_vel 6 + ee_pos 3 + init_red 3 + init_blue 3
              + init_green 3 + init_yellow 3 + bowl_pos 3 + target_one_hot 4 + actions 6 = 40.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        initial_red_cube_position = ObsTerm(func=mdp.initial_red_cube_position_in_robot_root_frame)
        initial_blue_cube_position = ObsTerm(func=mdp.initial_blue_cube_position_in_robot_root_frame)
        initial_green_cube_position = ObsTerm(func=mdp.initial_green_cube_position_in_robot_root_frame)
        initial_yellow_cube_position = ObsTerm(func=mdp.initial_yellow_cube_position_in_robot_root_frame)
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
        """Critic observations — full privileged state: current + initial positions of all cubes.

        Dims: joint_pos 6 + joint_vel 6 + ee_pos 3
              + cur_red 3 + cur_blue 3 + cur_green 3 + cur_yellow 3
              + init_red 3 + init_blue 3 + init_green 3 + init_yellow 3
              + bowl_pos 3 + target_one_hot 4 + actions 6 = 52.
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
        initial_red_cube_position = ObsTerm(func=mdp.initial_red_cube_position_in_robot_root_frame)
        initial_blue_cube_position = ObsTerm(func=mdp.initial_blue_cube_position_in_robot_root_frame)
        initial_green_cube_position = ObsTerm(func=mdp.initial_green_cube_position_in_robot_root_frame)
        initial_yellow_cube_position = ObsTerm(func=mdp.initial_yellow_cube_position_in_robot_root_frame)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        target_color = ObsTerm(func=mdp.target_color_one_hot)
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

    initialize_four_cube_state = EventTerm(func=mdp.initialize_four_cube_state, mode="startup")

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

    # # Randomize cube mass ±30% — covers different real cube materials and sizes.
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

    # # Randomize table surface friction — covers different real table surface conditions.
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


    reset_bowl_and_four_cubes = EventTerm(
        func=mdp.reset_bowl_and_four_cubes,
        mode="reset",
        params={
            "placement_point": (0.048, 0.0),
            "bowl_dist_range": (0.20, 0.40),
            "bowl_x_min": 0.148,
            "bowl_y_max": 0.20,
            "bowl_radius": 0.14,
            "cube_dist_range": (0.15, 0.30),
            "cube_x_min": 0.148,
            "cube_y_max": 0.20,
            # Half-gap between adjacent cube centres (m).
            # 0.011 m = cube half-width (0.010) + 1 mm physics clearance → ~2 mm edge gap.
            # The 2×2 cluster diagonal expands constraints by gap × √2 ≈ 0.0156 m.
            "cluster_half_gap": 0.011,
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
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP — all object-centric terms select the *target* cube."""

    # Reach the target cube
    reaching_object = RewTerm(
        func=mdp.reaching_target_cube_reward,
        params={"std": 0.15},
        weight=1.0,
    )

    # Keep gripper open while approaching the target cube
    gripper_aperture = RewTerm(
        func=mdp.target_gripper_aperture_reward,
        params={
            "std": 0.05,
            "saturation_pos": 0.15,
        },
        weight=2.0,
    )

    # Grasp quality on the target cube
    object_grasped = RewTerm(
        func=mdp.target_cube_grasped_reward,
        params={
            "force_saturation": 5.0,
            "force_balance_ratio": 3.0,
            "debug_print_interval": 50,
        },
        weight=10.0,
    )

    # Lift the target cube while grasped
    lifting_object = RewTerm(
        func=mdp.target_cube_lifted_reward,
        params={
            "start_height": 0.012,
            "saturation_height": 0.02,
            "force_saturation": 5.0,
            "force_balance_ratio": 3.0,
        },
        weight=15.0,
    )

    # Transport target cube toward bowl
    object_goal_tracking = RewTerm(
        func=mdp.target_cube_to_bowl_reward,
        params={"std": 0.3, "minimal_height": 0.05, "height_offset": BOWL_HOVER_HEIGHT},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.target_cube_to_bowl_reward,
        params={"std": 0.1, "minimal_height": 0.06, "height_offset": BOWL_HOVER_HEIGHT},
        weight=10.0,
    )

    # Sparse success: target cube in bowl, gripper retreated.
    # Starts at weight=0 and is ramped by curriculum.
    cube_in_bowl = RewTerm(
        func=mdp.target_cube_in_bowl_reward,
        params={
            "xy_threshold": 0.055,
            "z_max": 0.04,
            "z_min": 0.0,
            "consecutive_steps": 3,
            "ee_min_height_above_bowl": 0.07,
        },
        weight=0.0,
    )

    # Penalty: touching the wrong cube
    wrong_cube_grasped = RewTerm(
        func=mdp.wrong_cube_grasped_penalty,
        params={"grasp_quality_threshold": 0.1},
        weight=-2.0,
    )

    # Penalty: wrong cube ends up in bowl
    wrong_cube_in_bowl = RewTerm(
        func=mdp.wrong_cube_in_bowl_penalty,
        params={"xy_threshold": 0.055, "z_max": 0.04},
        weight=-5.0,
    )

    # Time penalty
    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.001)

    object_drop_penalty = RewTerm(
        func=mdp.target_cube_drop_penalty,
        params={"minimum_height": -0.05},
        weight=-0.5,
    )

    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    robot_body_cube_contact = RewTerm(
        func=mdp.robot_body_cube_contact_penalty,
        params={"threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces_cube_red_body")},
        weight=-1.0,
    )

    robot_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces_table")},
        weight=-2.0,
    )

    robot_bowl_contact = RewTerm(
        func=mdp.robot_bowl_contact_penalty,
        params={"threshold": 0.5, "sensor_cfg": SceneEntityCfg("contact_forces_bowl")},
        weight=0.0,
    )

    # Metric: % of envs with target cube lifted
    cube_lifted_pct = RewTerm(
        func=mdp.log_target_cube_lifted_pct,
        params={"min_height": 0.03},
        weight=1e-9,
    )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.target_cube_dropping, params={"minimum_height": -0.05}
    )


    task_success = DoneTerm(
        func=mdp.target_cube_placed_in_bowl,
        params={
            "xy_threshold": 0.055,
            "z_max": 0.04,
            "consecutive_steps": 3,
            "ee_min_height_above_bowl": 0.055,
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
    # cube_in_bowl = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "cube_in_bowl", "weight": 5000.0, "num_steps": 36_000}
    # )
    cube_in_bowl = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "cube_in_bowl", "weight": 5000.0, "num_steps": 48_000}
    )
    





##
# Environment configuration
##


@configclass
class TaskThreeEnvCfg(ManagerBasedRLEnvCfg):
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
