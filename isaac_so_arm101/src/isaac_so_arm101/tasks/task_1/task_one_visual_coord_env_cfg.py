# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Visual-coordinate PPO environment for Task 1.
#
# Replaces the ResNet18 wrist-image observation (or the privileged cube-position
# shortcut used during teacher training) with an analytic (u, v, visible)
# projection of the cube's world position onto the wrist camera image plane.
# No TiledCamera sensor is needed — the projection uses FK + the fixed camera
# offset from _wrist_cam.py, so the env runs at full state-based speed and
# supports 4096 parallel environments.
#
# At deployment the policy receives the same (u, v, visible) format produced by
# an HSV colour-segmentation step on the real wrist camera (see the docstring of
# mdp.cube_image_coords for the exact normalisation).
#
# Actor dims : 33  (joint_pos 6 + joint_vel 6 + ee_pos 3 + bowl_pos 3
#                   + cube_img 3 + color_one_hot 6 + actions 6)
# Critic dims: 27  (joint_pos 6 + joint_vel 6 + ee_pos 3 + obj_pos 3
#                   + bowl_pos 3 + actions 6)
#
# No ee_pos in either actor or critic — the policy learns FK implicitly from
# joint_pos, and deployment needs only joint encoders + HSV segmentation.
#
# Reward structure:
#   Inherits all terms from TaskOneEnvCfg unchanged, with two additions:
#     1. cube_visibility  weight → 1.0, max_steps → 99999 (continuous camera-facing incentive).
#     2. cube_visibility_pct     weight ≈ 0 (logging metric: % envs with cube in FOV).
#
# Domain randomisation (overriding base ranges for realistic deployment):
#   - Table friction   : static (0.30–0.55), dynamic (0.20–0.45)  — school laminate table
#   - Gripper friction : static (0.30–0.70), dynamic (0.20–0.55)  — 3D-printed PLA jaws
#   - Cube mass        : ±20% scale                               — wooden 2 cm cube
#
# Curriculum:
#   - cube_in_bowl enters at 2500 training iterations (num_steps = 60_000).

from . import mdp
from .task_one_env_cfg import (
    BOWL_HOVER_HEIGHT,
    RewardsCfg as _BaseRewardsCfg,
    TaskOneEnvCfg,
)

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


##
# Observations
##


@configclass
class VisualCoordObservationsCfg:
    """Asymmetric actor-critic observations for visual-coordinate PPO.

    Actor  (policy): proprioception + bowl pos + analytic cube image coords.
                     No explicit 3-D cube position — inferred at deployment from
                     HSV colour segmentation on the real wrist camera.
    Critic (critic): privileged full state — current cube position in robot frame.
                     Same dimensionality as the actor (27 dims each) so the same
                     network architecture works for both.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — 33 dims total.

        joint_pos(6) + joint_vel(6) + ee_pos(3) + bowl_pos(3) + cube_img(3)
        + color_one_hot(6) + actions(6) = 33

        ee_pos computed via FK + fixed gripper_link offset — deployment only needs
        joint encoders + a standard FK call (no extra sensors).
        color_one_hot: random during training; set to desired target color at deployment.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        cube_image = ObsTerm(
            func=mdp.cube_image_coords,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        target_color = ObsTerm(func=mdp.random_target_color_one_hot)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — 27 dims, privileged current cube position.

        joint_pos(6) + joint_vel(6) + ee_pos(3) + obj_pos(3) + bowl_pos(3) + actions(6) = 27

        Critic is never deployed — ee_pos is included for better value estimates.
        """

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
class VisualCoordRewardsCfg(_BaseRewardsCfg):
    """Reward structure identical to the Task 1 base, with two targeted changes.

    1. cube_visibility: weight raised to 1.0 and made always-active (max_steps=99999).
       This continuously incentivises the policy to orient the wrist camera toward the
       cube — the only way to get a useful (u, v) signal in the actor observations.

    2. cube_visibility_pct: near-zero-weight logging term that emits the percentage of
       envs currently with the cube inside the camera FOV to TensorBoard/WandB.
    """

    cube_visibility = RewTerm(
        func=mdp.cube_initial_visibility_reward,
        params={"max_steps": 99999, "std_offset": 0.5},
        weight=0.2,
    )

    cube_visibility_pct = RewTerm(
        func=mdp.log_cube_visibility_pct,
        params={"object_cfg": SceneEntityCfg("object")},
        weight=1e-9,
    )

    log_actions = RewTerm(
        func=mdp.log_actions,
        params={"print_interval": 100},
        weight=1e-9,
    )


##
# Environment configuration
##


@configclass
class TaskOneVisualCoordEnvCfg(TaskOneEnvCfg):
    """Task 1 with analytic wrist-camera image coordinates instead of privileged cube position.

    Inherits the scene (single cube + bowl + contact sensors), events (bowl-and-cube
    reset), terminations, curriculum, and sim settings from TaskOneEnvCfg.

    Only the observations and the visibility reward weight are changed:
      - Actor sees (u, v, visible) per cube from analytic FK projection — no camera sensor.
      - Critic sees full privileged state (current cube position).
      - cube_visibility reward is enabled continuously so the policy learns to keep
        the cube in the camera FOV throughout the episode, not just the first 20 steps.
    """

    observations: VisualCoordObservationsCfg = VisualCoordObservationsCfg()
    rewards: VisualCoordRewardsCfg = VisualCoordRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── Cube color: sample randomly each episode and apply to sim visual ──
        self.events.set_cube_target_color = EventTerm(
            func=mdp.set_cube_target_color,
            mode="reset",
        )

        # ── Domain randomisation: realistic deployment ranges ──────────────────
        # School laminate table — narrower than the base env (0.3–0.8 static).
        self.events.randomize_table_friction.params["static_friction_range"]  = (0.30, 0.55)
        self.events.randomize_table_friction.params["dynamic_friction_range"] = (0.20, 0.45)

        # 3D-printed PLA gripper jaws — lower ceiling than the base env (0.4–1.0 static).
        self.events.randomize_gripper_friction.params["static_friction_range"]  = (0.30, 0.70)
        self.events.randomize_gripper_friction.params["dynamic_friction_range"] = (0.20, 0.55)

        # Wooden 2 cm cube — ±20% mass variation (tighter than base ±30%).
        self.events.randomize_object_mass.params["mass_distribution_params"] = (0.80, 1.20)

        # ── Curriculum: cube_in_bowl enters at 2500 training iterations ────────
        # num_steps is incremented by num_steps_per_env (24) each iteration,
        # so 2500 iterations × 24 = 60_000.
        self.curriculum.cube_in_bowl.params["num_steps"] = 60_000
