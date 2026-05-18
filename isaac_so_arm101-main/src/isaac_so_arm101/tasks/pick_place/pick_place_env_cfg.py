# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick_place scene with a robot and an object."""

    # populated by agent env cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    wrist_cam: CameraCfg = MISSING
    sphere_light: AssetBaseCfg = MISSING

    # bowl_bottom is the single real-mesh bowl asset (RL_BOWL_CFG).
    # bowl_wall is gone — the real mesh provides its own wall geometry.
    bowl_bottom: RigidObjectCfg = MISSING

    # Table
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

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:

    @configclass
    class VisionPolicyCfg(ObsGroup):
        """Vision-based: wrist image + bowl pos + joint pos/vel + last action."""

        wrist_image = ObsTerm(
            func=mdp.wrist_camera_image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam")},
        )
        bowl_pos = ObsTerm(
            func=mdp.bowl_center_position,
            params={"asset_cfg": SceneEntityCfg("bowl_bottom")},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StatePolicyCfg(ObsGroup):
        object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},  # ← fix here only
        )
        bowl_pos = ObsTerm(
            func=mdp.bowl_center_position,
            params={"asset_cfg": SceneEntityCfg("bowl_bottom")},  # ← leave as is
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions   = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Default to vision; agent cfg can swap in StatePolicyCfg
    policy: VisionPolicyCfg | StatePolicyCfg = VisionPolicyCfg()


@configclass
class EventCfg:
    # ------------------------------------------------------------------
    # Scene reset — always first
    # ------------------------------------------------------------------
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # ------------------------------------------------------------------
    # Pose randomization
    # ------------------------------------------------------------------
    reset_bowl_and_object = EventTerm(
        func=mdp.reset_bowl_and_object_non_overlapping,
        mode="reset",
        params={
            "bowl_cfg":        SceneEntityCfg("bowl_bottom"),
            "object_cfg":      SceneEntityCfg("object", body_names="Object"),
            "bowl_xy_range":   {"x": (0.35, 0.55), "y": (-0.20, 0.20)},
            "object_xy_range": {"x": (0.25, 0.45), "y": (-0.20, 0.20)},
            "min_xy_distance": 0.13,  # bumped from 0.10 to reduce overlap risk
        },
    )

    # Full yaw randomization — policy must generalize to all cube faces
    randomize_object_orientation = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (-3.14159, 3.14159),  # full 360°
            },
            "velocity_range": {},
        },
    )
    # ------------------------------------------------------------------
    # Robot physical properties
    # ------------------------------------------------------------------

    # Servo stiffness/damping variation — biggest sim-to-real gap on SO-101
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params":   (0.8, 1.2),
            "operation":    "scale",
            "distribution": "uniform",
        },
    )

    # Gripper contact friction — covers rubber wear and surface variation
    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["gripper_link", "moving_jaw_so101_v1_link"],
            ),
            "static_friction_range":  (0.5, 1.0),   # tightened upper bound from 1.2
            "dynamic_friction_range": (0.4, 0.85),
            "restitution_range":      (0.0, 0.0),
            "num_buckets":            16,
            "make_consistent":        True,
        },
    )

    # ------------------------------------------------------------------
    # Object physical properties
    # ------------------------------------------------------------------

    # Mass ±50% — wider than before to cover more real-world cube materials
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg":                  SceneEntityCfg("object"),
            "mass_distribution_params":   (0.5, 1.5),
            "operation":    "scale",
            "distribution": "uniform",
        },
    )

    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range":  (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range":      (0.0, 0.0),
            "num_buckets": 8,
        },
    )


    # ------------------------------------------------------------------
    # Disturbance — add once policy is stable, comment out during early training
    # ------------------------------------------------------------------

    # Random velocity push — tests recovery, improves robustness
    #push_robot = EventTerm(
    #    func=mdp.push_by_setting_velocity,
    #    mode="interval",
    #    interval_range_s=(5.0, 12.0),
    #    params={
    #        "velocity_range": {
    #            "x": (-0.08, 0.08),
    #            "y": (-0.08, 0.08),
    #            "z": (0.0,   0.0),
    #        },
    #    },
    #)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.15}, weight=1.0)

    # object_grasped = RewTerm(
    #     func=mdp.gripper_closed_near_object,
    #     params={"std": 0.015},
    #     weight=2.0,
    # )

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

    time_penalty_no_grasp = RewTerm(
        func=mdp.time_penalty_if_not_lifted,
        params={
            "start_height": 0.05,
            "cube_sensor_cfg": SceneEntityCfg("contact_forces_cube"),
        },
        weight=-2.0,  # tune — too high and the agent just terminates early
    )

    # Stage 3 — carry toward bowl (coarse guidance from afar)
    object_to_bowl_coarse = RewTerm(
        func=mdp.object_bowl_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.08,
            "bowl_cfg": SceneEntityCfg("bowl_bottom"),
        },
        weight=8.0,
    )

    # Stage 3 — fine-grained positioning over bowl
    object_to_bowl_fine = RewTerm(
        func=mdp.object_bowl_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.08,   # was 0.10 — match coarse so fine isn't blocked
            "bowl_cfg": SceneEntityCfg("bowl_bottom"),
        },
        weight=10.0,
    )

    # Stage 4 — release over bowl (was referencing missing object_in_target_zone)
    dropping_success = RewTerm(
        func=mdp.object_released_in_zone,  # fix: use the function that actually exists
        params={
            "threshold": 0.05,
            "target_cfg": SceneEntityCfg("bowl_bottom"),
        },
        weight=100.0,
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    object_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.5, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass

class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass

##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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

