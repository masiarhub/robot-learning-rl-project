# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Visual-coordinate PPO environment for Task 2.
#
# Replaces the ResNet18 wrist-image observation (or the privileged cube positions
# used during teacher training) with an analytic (u, v, visible) projection of
# the TARGET cube onto the wrist camera image plane.
# No TiledCamera sensor needed — uses FK + the fixed offset from _wrist_cam.py,
# so the env runs at full state-based speed and supports 4096 parallel envs.
#
# At deployment the policy receives the same (u, v, visible) format produced by
# HSV colour-segmentation on the real wrist camera (see target_cube_image_coords).
# The color_one_hot encodes which of the 6 palette colors is the target; set this
# to the desired target color at deployment time.
#
# Actor dims : 33  (joint_pos 6 + joint_vel 6 + ee_pos 3 + bowl_pos 3
#                   + target_cube_img 3 + color_one_hot 6 + actions 6)
# Critic dims: 27  (joint_pos 6 + joint_vel 6 + ee_pos 3 + target_cube_pos 3
#                   + bowl_pos 3 + actions 6)
#
# No ee_pos in either actor or critic — the policy learns FK implicitly from
# joint_pos, and deployment only needs joint encoders + HSV segmentation.
#
# Reward structure:
#   Inherits all terms from base TaskTwoEnvCfg, with two additions:
#     1. cube_visibility  weight → 1.0, max_steps → 99999 (continuous camera incentive).
#     2. cube_visibility_pct     weight ≈ 0 (logging: % envs with target in FOV).
#
# Domain randomisation (overriding base; all commented-out in base EventCfg):
#   - Table friction   : static (0.30–0.55), dynamic (0.20–0.45)  — school laminate
#   - Gripper friction : static (0.30–0.70), dynamic (0.20–0.55)  — 3D-printed PLA jaws
#   - Cube mass (both) : ±20% scale                               — wooden 2 cm cube
#
# Curriculum:
#   - cube_in_bowl enters at 2500 training iterations (num_steps = 60_000).

from . import mdp
from .task_two_env_cfg import (
    BOWL_HOVER_HEIGHT,
    RewardsCfg as _BaseRewardsCfg,
    TaskTwoEnvCfg,
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

    Actor  (policy): proprioception + bowl pos + analytic target-cube image coords
                     + 6-class color one-hot. No 3-D cube position or ee_pos.
    Critic (critic): privileged state — current TARGET cube position in robot frame.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — 33 dims total.

        joint_pos(6) + joint_vel(6) + ee_pos(3) + bowl_pos(3) + target_cube_img(3)
        + color_one_hot(6) + actions(6) = 33

        ee_pos computed via FK + fixed gripper_link offset — deployment only needs
        joint encoders + a standard FK call (no extra sensors).
        color_one_hot: set_two_cube_colors samples randomly each reset; at deployment,
        pass the fixed one-hot for the desired target color.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        bowl_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("bowl"), "height_offset": BOWL_HOVER_HEIGHT},
        )
        target_cube_image = ObsTerm(func=mdp.target_cube_image_coords)
        target_color = ObsTerm(func=mdp.random_target_color_one_hot)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — 27 dims, privileged current target cube position.

        joint_pos(6) + joint_vel(6) + ee_pos(3) + target_cube_pos(3) + bowl_pos(3) + actions(6) = 27

        Critic is never deployed — ee_pos is included for better value estimates.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ee_position = ObsTerm(func=mdp.ee_position_in_robot_root_frame_for_deployment)
        target_cube_position = ObsTerm(func=mdp.target_cube_position_in_robot_root_frame)
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
    """Reward structure identical to Task 2 base, with two targeted additions.

    1. cube_visibility: weight 1.0, always-active (max_steps=99999) — continuously
       incentivises the policy to orient the wrist camera toward the target cube,
       the only way to get a useful (u, v) signal in the actor observations.

    2. cube_visibility_pct: near-zero-weight logging term that emits the percentage
       of envs with the target cube inside the camera FOV to TensorBoard/WandB.
    """

    cube_visibility = RewTerm(
        func=mdp.target_cube_visibility_reward,
        params={"max_steps": 99999, "std_offset": 0.5},
        weight=0.2,
    )

    cube_visibility_pct = RewTerm(
        func=mdp.log_target_cube_visibility_pct,
        weight=1e-9,
    )


##
# Environment configuration
##


@configclass
class TaskTwoVisualCoordEnvCfg(TaskTwoEnvCfg):
    """Task 2 with analytic wrist-camera target-cube image coordinates.

    Inherits scene (both cubes + bowl + contact sensors), events (two-cube reset),
    terminations, curriculum, and sim settings from TaskTwoEnvCfg.

    Changes vs base:
      - Observations: actor sees target-cube (u,v,visible) + 6-class color one-hot;
                      critic sees privileged current target-cube position.
      - Events: set_two_cube_colors assigns 2 distinct colors from a 6-color palette
                each reset, updating USD visuals and storing which cube is the target.
      - Domain randomisation re-enabled with realistic deployment ranges.
      - Curriculum: cube_in_bowl delayed to 2500 iterations.
    """

    observations: VisualCoordObservationsCfg = VisualCoordObservationsCfg()
    rewards: VisualCoordRewardsCfg = VisualCoordRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── Color assignment: sample 2 different colors, apply visuals ───────────
        self.events.set_two_cube_colors = EventTerm(
            func=mdp.set_two_cube_colors,
            mode="reset",
        )

        # ── Domain randomisation: realistic deployment ranges ─────────────────────
        # School laminate table
        self.events.randomize_table_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("table"),
                "static_friction_range": (0.30, 0.55),
                "dynamic_friction_range": (0.20, 0.45),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
                "make_consistent": True,
            },
        )

        # 3D-printed PLA gripper jaws
        self.events.randomize_gripper_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]),
                "static_friction_range": (0.30, 0.70),
                "dynamic_friction_range": (0.20, 0.55),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
                "make_consistent": True,
            },
        )

        # Wooden 2 cm cubes — ±20% mass variation
        self.events.randomize_red_cube_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object_red"),
                "mass_distribution_params": (0.80, 1.20),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        self.events.randomize_blue_cube_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object_blue"),
                "mass_distribution_params": (0.80, 1.20),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # ── Curriculum: cube_in_bowl enters at 2500 training iterations ───────────
        # num_steps is incremented by num_steps_per_env (24) each iteration,
        # so 2500 iterations × 24 = 60_000.
        self.curriculum.cube_in_bowl.params["num_steps"] = 100_000
