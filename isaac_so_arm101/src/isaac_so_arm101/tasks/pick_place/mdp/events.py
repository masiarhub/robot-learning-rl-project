# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_bowl_and_cube(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    bowl_pose_range: dict[str, tuple[float, float]],
    cube_world_range: dict[str, tuple[float, float]],
    exclusion_radius: float = 0.10,
    y_occlusion_threshold: float = 0.20,
    max_placement_tries: int = 300,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the bowl to a randomised position, then place the cube with constraints.

    cube_world_range defines XY bounds in each environment's LOCAL frame —
    i.e. relative to the robot, matching the debug visualisation plot axes.

    Cube placement constraints:
      1. Exclusion zone  – distance from bowl centre > exclusion_radius (default 0.10 m)
      2. X occlusion     – cube_x <= bowl_x  UNLESS |cube_y - bowl_y| > y_occlusion_threshold
      3. Y occlusion     – if bowl_y < 0: cube_y >= bowl_y
                           if bowl_y > 0: cube_y <= bowl_y
                           if bowl_y == 0: no constraint

    Uses per-env rejection sampling. Emits a warning if placement fails.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    range_keys = ["x", "y", "z"]

    # ── Bowl ──────────────────────────────────────────────────────────────────
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    bowl_state[:, :3] += env.scene.env_origins[env_ids]

    bowl_ranges = torch.tensor(
        [bowl_pose_range.get(k, (0.0, 0.0)) for k in range_keys], device=env.device
    )
    bowl_offsets = math_utils.sample_uniform(
        bowl_ranges[:, 0], bowl_ranges[:, 1], (len(env_ids), 3), device=env.device
    )
    bowl_state[:, :3] += bowl_offsets

    bowl.write_root_pose_to_sim(bowl_state[:, :7], env_ids=env_ids)
    bowl.write_root_velocity_to_sim(bowl_state[:, 7:], env_ids=env_ids)

    # ── Cube (rejection sampling in local env frame) ───────────────────────────
    obj_state = obj.data.default_root_state[env_ids].clone()

    cube_ranges = torch.tensor(
        [cube_world_range.get(k, (0.0, 0.0)) for k in ["x", "y"]], device=env.device
    )  # shape (2, 2) — x and y only; z is always the cube's own default height

    n = len(env_ids)

    # Bowl position in LOCAL frame (subtract env_origin) for constraint checks.
    # cube_world_xy is also sampled in local frame, so offsets are consistent.
    bx_local = bowl_state[:, 0] - env.scene.env_origins[env_ids, 0]  # (n,)
    by_local = bowl_state[:, 1] - env.scene.env_origins[env_ids, 1]  # (n,)

    def check_validity(local_xy):
        ox = local_xy[:, 0] - bx_local  # x offset from bowl centre (local frame)
        oy = local_xy[:, 1] - by_local  # y offset from bowl centre (local frame)
        c1 = (ox**2 + oy**2) > exclusion_radius**2
        c2 = (torch.abs(oy) > y_occlusion_threshold) | (ox <= 0.0)
        c3 = (~(by_local < 0) | (oy >= 0)) & (~(by_local > 0) | (oy <= 0))
        return c1 & c2 & c3

    # Initial sample in local frame.
    cube_local_xy = math_utils.sample_uniform(
        cube_ranges[:, 0], cube_ranges[:, 1], (n, 2), device=env.device
    )
    needs_resample = ~check_validity(cube_local_xy)

    for _ in range(max_placement_tries):
        if not needs_resample.any():
            break
        new_xy = math_utils.sample_uniform(
            cube_ranges[:, 0], cube_ranges[:, 1], (n, 2), device=env.device
        )
        cube_local_xy[needs_resample] = new_xy[needs_resample]
        needs_resample[needs_resample.clone()] = ~check_validity(cube_local_xy)[needs_resample]

    if needs_resample.any():
        print(
            f"[WARNING] reset_bowl_and_cube: {needs_resample.sum().item()} env(s) failed to find "
            f"a valid cube placement after {max_placement_tries} tries. "
            f"Consider widening cube_world_range."
        )

    # Convert local → world frame by adding env_origins before writing to sim.
    obj_state[:, 0] = cube_local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
    obj_state[:, 1] = cube_local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
    obj_state[:, 2] = obj.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
    obj_state[:, 7:] = 0.0

    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)