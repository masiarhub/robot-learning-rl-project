# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .task_one_camera_env_cfg import TaskOneCameraEnvCfg
from .task_one_env_cfg import EventCfg


@configclass
class PostTrainEventCfg(EventCfg):
    """Stronger domain randomization for post-training the distilled student.

    All parent terms (reset_all, reset_bowl_and_cube, randomize_dome_light) are
    inherited; only the ones that need different parameters are overridden here.
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
            # bowl init_state x=0.30 → local x ∈ [0.23, 0.42], y ∈ [-0.22, 0.22]
            "bowl_pose_range": {"x": (-0.05, 0.10), "y": (-0.20, 0.20)},
            # Absolute XY sampling rectangle for the cube in local (robot-relative) frame.
            # Visualise valid regions with debug/cube_placement_constraints.py.
            "cube_world_range": {"x": (0.15, 0.35), "y": (-0.25, 0.25)},
            "exclusion_radius": 0.10,
            "exclusion_shape": "box",
            "y_occlusion_threshold": 0.30,
            "max_placement_tries": 100,
            "cube_z_rotation_range": (0.0, 2.0 * math.pi),
        },
    )


@configclass
class TaskOnePostTrainEnvCfg(TaskOneCameraEnvCfg):
    """Camera-based environment for RL post-training of the distilled student.

    Inherits the full camera observation space (wrist RGB + proprioception) and
    asymmetric critic (privileged state) from TaskOneCameraEnvCfg, then replaces
    the event config with PostTrainEventCfg for stronger domain randomization.

    Rewards and curriculum are already handled by the parent:
      - curriculum = None  (set in TaskOneCameraEnvCfg.__post_init__)
      - cube_in_bowl weight = 5000.0  (fully active from episode 0)
      - robot_bowl_contact weight = -0.2
    """

    events: PostTrainEventCfg = PostTrainEventCfg()
