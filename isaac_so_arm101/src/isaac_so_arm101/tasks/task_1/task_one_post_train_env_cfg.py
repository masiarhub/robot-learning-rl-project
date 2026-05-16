# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .task_one_distill_env_cfg import CameraEventCfg, TaskOneDistillEnvCfg


@configclass
class PostTrainEventCfg(CameraEventCfg):
    """Stronger domain randomization for post-training the distilled student.

    All parent terms (reset_all, reset_bowl_and_cube, randomize_dome_light,
    color randomizations) are inherited from CameraEventCfg; only the ones
    that need different parameters are overridden here.
    The physical property randomizers that were commented out in the base env are
    activated to expose the student to a wider distribution of dynamics.
    """

    # Wider dome light range: intensity ±2× baseline, hue ±15% vs ±10% in base.
    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(3.0, 7.0),
        is_global_time=True,
        params={
            "prim_path": "/World/light",
            "intensity_range": (400.0, 2500.0),
            "color_range": (0.55, 0.95),
        },
    )

    # Gripper surface friction variability — simulates different gripper wear states.
    randomize_gripper_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["gripper_link", "moving_jaw_so101_v1_link"]
            ),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )

    # Cube mass ±40% — covers different cube materials and imprecise fabrication.
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Table surface friction variability.
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
            "bowl_radius": 0.12,
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
            "cube_z_rotation_range": (0.0, 2.0 * math.pi),
        },
    )


@configclass
class TaskOnePostTrainEnvCfg(TaskOneDistillEnvCfg):
    """Camera-based environment for RL post-training of the distilled student.

    Inherits the full camera observation space (wrist RGB + proprioception) and
    asymmetric critic (privileged state) from TaskOneDistillEnvCfg, then replaces
    the event config with PostTrainEventCfg for stronger domain randomization.

    Rewards and curriculum are already handled by the parent:
      - curriculum = None  (set in TaskOneDistillEnvCfg.__post_init__)
      - cube_in_bowl weight = 2000.0  (fully active from episode 0)
      - robot_bowl_contact weight = -0.2
    """

    events: PostTrainEventCfg = PostTrainEventCfg()

    def __post_init__(self):
        super().__post_init__()
        # Re-enable visibility reward during RL fine-tuning so the policy continues
        # to orient the wrist camera at the cube in the early episode steps.
        self.rewards.cube_visibility.weight = 1.5
