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

import isaaclab.sim as sim_utils

from . import mdp

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

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, -0.5]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.2, 1),
            # HEX: #B8ADA9   -> RGB: rgb(184 173 169)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(184/255, 173/255, 169/255)),
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

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    # Per-environment sphere light — randomized independently per env at every reset.
    light_local = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SphereLight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.3, 0.0, 0.5)),
        spawn=sim_utils.SphereLightCfg(intensity=8000.0, radius=0.3, treat_as_point=False),
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
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # observation of bowl position, but offset (target where cube should get dropped)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Set bowl color once at startup — HEX #d4be9f.
    set_bowl_color = EventTerm(
        func=mdp.set_bowl_color,
        mode="startup",
        params={"color": (212 / 255, 190 / 255, 159 / 255)},
    )


    # Randomize gripper contact friction — covers different gripper surface conditions.
    # Targets the two contact links: fixed jaw and moving jaw.
    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
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
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # Randomize dome ambient intensity and slight color temperature shift every 30–120 sim-seconds.
    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        is_global_time=True,
        params={
            "prim_path": "/World/light",
            "intensity_range": (200.0, 400.0),
            "color_range": (0.65, 0.85),
        },
    )

    # Randomize per-env sphere light independently at every reset.
    randomize_sphere_light = EventTerm(
        func=mdp.randomize_sphere_light,
        mode="reset",
        params={
            "intensity_range": (3000.0, 15000.0),
            "color_range": (0.65, 0.85),
            "radius_range": (0.1, 0.4),
            "pos_x_range": (-0.2, 0.5),
            "pos_y_range": (-0.4, 0.4),
            "pos_z_range": (0.3, 1.0),
        },
    )

    reset_bowl_and_cube = EventTerm(
        func=mdp.reset_bowl_and_cube,
        mode="reset",
        params={
            # bowl init_state is at [0.30, 0.0, 0.0] (set in joint_pos_env_cfg.py).
            # x offset (-0.05, +0.10) → bowl local x ∈ [0.25, 0.40].
            # y offset (-0.20, +0.20) → bowl local y ∈ [-0.20, +0.20].
            "bowl_pose_range": {"x": (-0.05, 0.10), "y": (-0.20, 0.20)},
            # Absolute XY sampling rectangle for the cube in local (robot-relative) frame.
            # Visualise valid regions with debug/cube_placement_constraints.py.
            "cube_world_range": {"x": (0.10, 0.3), "y": (-0.3, 0.3)},
            "exclusion_radius": 0.10,
            "exclusion_shape": "box",
            "y_occlusion_threshold": 0.20,
            "max_placement_tries": 100,
        },
    )
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # reward for EE being (very)close to cube
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    # binary reward when object is lifted over minimal_height
    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.015, "saturation_height": 0.02}, weight=15.0) # adjusted minmal height: 0.025 -> 0.02
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.015}, weight=15.0) # adjusted minmal height: 0.025 -> 0.02



    # track distance object - (bowl + height_offset), only if lifted over minimal_height
    object_goal_tracking = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.3, "minimal_height": 0.05, "height_offset": BOWL_HOVER_HEIGHT, "debug_vis": True},
        weight=16.0,
    )

    # fine-grained distance reward, tighter std to reward precise placement
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_bowl_distance,
        params={"std": 0.05, "minimal_height": 0.08, "height_offset": BOWL_HOVER_HEIGHT},
        weight=5.0,
    )

    # Sparse success reward: cube inside the bowl and gripper open.
    # Starts at 0 and is ramped up by curriculum — avoids overwhelming early exploration.
    cube_in_bowl = RewTerm(
        func=mdp.cube_in_bowl,
        params={
            "xy_threshold": 0.06,           # bowl inner radius at scale 1.35 ≈ 0.06 m
            "z_max": 0.05,                  # bowl wall height ≈ 0.05 m
            "gripper_open_threshold": 0.35, # open cmd = 0.5 rad; 0.35 filters half-open
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=0.0,
    )

    # time penalty: -1.0 per step encourages faster task completion
    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.0)

    # action penalty (regularization)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # joint velocity penalty (regularization)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
            "xy_threshold": 0.06,
            "z_max": 0.05,
            "gripper_open_threshold": 0.35,
            "consecutive_steps": 3,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 18000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 18000}
    )

    action_rate_dropping = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-4, "num_steps": 25000}
    )

    joint_vel_dropping = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-4, "num_steps": 25000}
    )

    # Ramp in the sparse success reward once the agent has had time to learn lifting/tracking.
    # Starts at weight=0 (see RewardsCfg) and reaches 2000 after num_steps env steps.
    cube_in_bowl = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "cube_in_bowl", "weight": 2000.0, "num_steps": 25000}
    )
    alive_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "alive_penalty", "weight": -1.0, "num_steps": 25000}
    )




    # Regularization stays flat at -1e-4 throughout — no ramp-up so the success curriculum
    # is not fighting increased smoothness penalties during the critical learning phase.


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick-and-place environment."""

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
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01  # low threshold suits the small 2 cm cube
        self.sim.physx.friction_correlation_distance = 0.00625
