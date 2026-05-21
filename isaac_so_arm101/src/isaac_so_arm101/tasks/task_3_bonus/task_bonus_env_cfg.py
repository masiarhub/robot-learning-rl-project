# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Task Bonus — Singulation.
# Three cubes of distinct fixed colors (red, blue, green) start in a vertical stack
# at a randomly sampled reachable position each episode.
# Goal: bring all three cubes to the table in a graspable (non-stacked) configuration.
# No bowl — the robot must simply separate the stack.

from dataclasses import MISSING
from pathlib import Path

import math
import isaaclab.sim as sim_utils

from . import mdp
from ._colors import (
    TABLE_BASE_COLOR,
    CUBE_RED_COLOR,
    CUBE_BLUE_COLOR,
    CUBE_GREEN_COLOR,
)

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaac_so_arm101.robots import SO_ARM101_CFG


##
# Scene definition
##


@configclass
class TaskBonusSceneCfg(InteractiveSceneCfg):
    """Three cubes + table. Colors randomised each episode from a 6-color palette."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object_red: RigidObjectCfg = MISSING
    object_blue: RigidObjectCfg = MISSING
    object_green: RigidObjectCfg = MISSING

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

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=500.0),
    )

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


##
# MDP settings
##


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Asymmetric actor-critic observations for the bonus singulation task.

    Actor  (24 dims): joint_pos 6 + joint_vel 6 + ee_pos 3
                      + initial_stack_center 3 + actions 6
      → Fully deployable.  The initial stack centre is measured once per episode
        (e.g. from a depth sensor or overhead camera) and kept fixed.

    Critic (36 dims): same robot state, but current positions of all three cubes
                      replace the frozen stack centre (privileged, training only).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        initial_stack_center = ObsTerm(
            func=mdp.initial_stack_center_position_in_robot_root_frame,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        cube_0_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_red")},
        )
        cube_1_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_blue")},
        )
        cube_2_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object_green")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Rewards for knocking the stack flat and keeping cubes close together.

    Curriculum:
      1. ee_approach_stack  — dense reaching gradient; draws the arm toward the stack.
      2. cubes_flat         — grows as cubes settle to table height; main task signal.
      3. cubes_clustered    — penalises cubes scattering far; rewards a clean push.
      Regularisers keep motion smooth and prevent arm-on-table contact.
    """

    # ── Task rewards ──────────────────────────────────────────────────────────
    ee_approach_stack = RewTerm(
        func=mdp.ee_approach_stack_reward,
        weight=0.5,
        params={"std": 0.08},
    )
    cubes_flat = RewTerm(
        func=mdp.cubes_flat_reward,
        weight=3.0,
        params={
            "table_z": 0.01,
            "height_std": 0.008,
            "object_0_cfg": SceneEntityCfg("object_red"),
            "object_1_cfg": SceneEntityCfg("object_blue"),
            "object_2_cfg": SceneEntityCfg("object_green"),
        },
    )
    cubes_clustered = RewTerm(
        func=mdp.cubes_clustered_reward,
        weight=1.0,
        params={
            "cluster_std": 0.05,
            "object_0_cfg": SceneEntityCfg("object_red"),
            "object_1_cfg": SceneEntityCfg("object_blue"),
            "object_2_cfg": SceneEntityCfg("object_green"),
        },
    )

    # ── Regularisers ──────────────────────────────────────────────────────────
    alive_penalty = RewTerm(func=mdp.is_alive, weight=-0.001)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-5e-5)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    robot_table_contact = RewTerm(
        func=mdp.robot_table_contact_penalty,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces_table")},
        weight=-2.0,
    )


@configclass
class EventCfg:
    initialize_cube_state = EventTerm(
        func=mdp.initialize_three_cube_state,
        mode="startup",
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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

    reset_stacked_cubes = EventTerm(
        func=mdp.reset_stacked_three_cubes,
        mode="reset",
        params={
            "stack_x_range": (0.20, 0.35),
            "stack_y_max": 0.12,
        },
    )

    # Color randomization is cosmetic only (actor has no camera obs) — disabled during
    # training to avoid per-env USD writes.  Re-enable in play/visualization configs.


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class TaskBonusEnvCfg(ManagerBasedRLEnvCfg):
    """Task Bonus: singulate a vertical stack of six colored cubes."""

    scene: TaskBonusSceneCfg = TaskBonusSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # ── Robot ──────────────────────────────────────────────────────────────
        self.scene.robot = SO_ARM101_CFG.replace(
            spawn=SO_ARM101_CFG.spawn.replace(
                activate_contact_sensors=True,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
            ),
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan":  0.0,
                    "shoulder_lift": -1.4,
                    "elbow_flex":     0.4,
                    "wrist_flex":     1.4,
                    "wrist_roll":    -1.57,
                    "gripper":        0.2,
                },
                joint_vel={".*": 0.0},
            ),
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.01, 0.0, -0.09)),
                ),
            ],
        )

        # ── Actions ────────────────────────────────────────────────────────────
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.6,
            use_default_offset=True,
        )

        # ── Cubes ──────────────────────────────────────────────────────────────
        _cube_physics = dict(
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
        )

        self.scene.object_red = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.01], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                **_cube_physics,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_RED_COLOR),
            ),
        )
        self.scene.object_blue = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectBlue",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.03], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                **_cube_physics,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_BLUE_COLOR),
            ),
        )
        self.scene.object_green = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObjectGreen",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.05], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                **_cube_physics,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_GREEN_COLOR),
            ),
        )
        super().__post_init__()

        # ── Sim settings ───────────────────────────────────────────────────────
        self.decimation = 2
        self.episode_length_s = 2.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.enable_dl_denoiser = True
