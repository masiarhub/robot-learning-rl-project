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
    cube_offset_range: dict[str, tuple[float, float]],
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the bowl to a (optionally randomised) position, then place the cube relative to it.

    bowl_pose_range:   XYZ offset from the bowl's init_state.  Pass {} or all-zero ranges to keep
                       the bowl fixed at its init_state position.
    cube_offset_range: XY offset from the bowl's placed centre.  Z is always the cube's default
                       table-surface height regardless of this range.

    Because bowl and cube are placed in a single function call, ordering is guaranteed —
    no need to split across two EventTerms.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    range_keys = ["x", "y", "z"]

    # ── Bowl ──────────────────────────────────────────────────────────────────
    # default_root_state is env-local; add env_origins to convert to world frame.
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

    # ── Cube (placed relative to bowl) ────────────────────────────────────────
    obj_state = obj.data.default_root_state[env_ids].clone()

    cube_ranges = torch.tensor(
        [cube_offset_range.get(k, (0.0, 0.0)) for k in range_keys], device=env.device
    )
    cube_offsets = math_utils.sample_uniform(
        cube_ranges[:, 0], cube_ranges[:, 1], (len(env_ids), 3), device=env.device
    )

    obj_state[:, 0] = bowl_state[:, 0] + cube_offsets[:, 0]
    obj_state[:, 1] = bowl_state[:, 1] + cube_offsets[:, 1]
    # Z: always the cube's own default height (table surface) — ignore cube_offset z.
    obj_state[:, 2] = obj.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
    obj_state[:, 7:] = 0.0

    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)
