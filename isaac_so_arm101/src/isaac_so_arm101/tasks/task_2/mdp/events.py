# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import random
import torch
from pxr import Gf, Sdf, Usd
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import find_matching_prims, get_current_stage
from isaaclab.utils import math as math_utils

from .._wrist_cam import FOCAL_LENGTH_MM, OFFSET_POS, OFFSET_QUAT_WXYZ
from .._colors import (
    CUBE_BASE_COLOR as _CUBE_BASE_COLOR,
    BOWL_BASE_COLOR as _BOWL_BASE_COLOR,
    TABLE_BASE_COLOR as _TABLE_BASE_COLOR,
    GRIPPER_BASE_COLOR as _GRIPPER_BASE_COLOR,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initialize_two_cube_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> None:
    """Startup event: allocate target-color and initial-position tensors.

    Runs once at environment construction (mode='startup') so that observation
    and reward functions that read these tensors never see an uninitialised
    attribute during the very first obs evaluation that happens before reset.
    Values are randomised (not all-red) so any bug where reset never fires
    would produce a balanced but wrong signal rather than a silent red bias.
    reset_bowl_and_two_cubes overwrites these with correct per-episode values.
    """
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.randint(
            0, 2, (env.num_envs,), dtype=torch.int64, device=env.device
        )
    if not hasattr(env, "_target_is_red"):
        env._target_is_red = torch.randint(
            0, 2, (env.num_envs,), dtype=torch.int64, device=env.device
        ).bool()
    if not hasattr(env, "_initial_red_cube_pos_w"):
        env._initial_red_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_initial_blue_cube_pos_w"):
        env._initial_blue_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)


def reset_bowl_and_cube(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    # Placement point (first revolute joint) in local robot frame
    placement_point: tuple[float, float] = (0.048, 0.0),
    # Bowl: annular ring + axis constraints
    bowl_dist_range: tuple[float, float] = (0.20, 0.40),
    bowl_x_min: float = 0.148,
    bowl_y_max: float = 0.20,
    # Bowl radius used for both the keep-out circle and the occlusion-cone half-width
    bowl_radius: float = 0.12,
    # Cube: annular ring + axis constraints
    cube_dist_range: tuple[float, float] = (0.15, 0.30),
    cube_x_min: float = 0.148,
    cube_y_max: float = 0.20,
    # Two-phase sampling strategy (mirrors debug/placement_constraints.py)
    safe_fallback_after: int = 100,
    max_placement_tries: int = 200,
    safety_positions: list | None = None,
    # Cube orientation
    cube_z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    # Scene entities
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset bowl and cube with annular-ring sampling and exact occlusion-cone constraint.

    Mirrors the logic in debug/placement_constraints.py — run that script to visualise
    valid/invalid regions before tuning parameters.

    All x/y coordinates are in each environment's LOCAL (robot-relative) frame.

    Bowl is placed first by rejection-sampling from an annular ring [bowl_dist_range]
    around placement_point, subject to bowl_x_min and bowl_y_max.

    Cube is then placed in two phases:
      Phase 1 — up to safe_fallback_after random draws from the cube annular ring,
                 checking: ring radius, x_min, bowl exclusion circle, occlusion cone, y_band.
      Phase 2 — for envs still invalid, try safety_positions in order; the first entry
                 that passes all checks (against that env's bowl) is used.
    """
    if safety_positions is None:
        safety_positions = [
            (0.268, +0.000),
            (0.253, +0.143),
            (0.253, -0.143),
            (0.293, +0.114),
            (0.293, -0.114),
            (0.338, +0.000),
            (0.189, +0.169),
            (0.189, -0.169),
        ]

    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    n = len(env_ids)
    px = float(placement_point[0])
    py = float(placement_point[1])
    r_lo_bowl, r_hi_bowl = float(bowl_dist_range[0]), float(bowl_dist_range[1])
    r_lo_cube, r_hi_cube = float(cube_dist_range[0]), float(cube_dist_range[1])

    def _sample_annulus(count: int, r_lo: float, r_hi: float) -> torch.Tensor:
        angle = torch.rand(count, device=env.device) * (2.0 * math.pi)
        r = torch.sqrt(torch.rand(count, device=env.device) * (r_hi**2 - r_lo**2) + r_lo**2)
        return torch.stack([px + r * torch.cos(angle), py + r * torch.sin(angle)], dim=1)

    # ── Bowl: rejection-sample from annular ring ──────────────────────────────
    def _check_bowl(local_xy: torch.Tensor) -> torch.Tensor:
        x, y = local_xy[:, 0], local_xy[:, 1]
        d = torch.sqrt((x - px) ** 2 + (y - py) ** 2)
        return (d >= r_lo_bowl) & (d <= r_hi_bowl) & (x >= bowl_x_min) & (torch.abs(y) <= bowl_y_max)

    bowl_local_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
    needs_resample_bowl = ~_check_bowl(bowl_local_xy)
    for _ in range(max_placement_tries):
        if not needs_resample_bowl.any():
            break
        new_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
        bowl_local_xy[needs_resample_bowl] = new_xy[needs_resample_bowl]
        needs_resample_bowl[needs_resample_bowl.clone()] = ~_check_bowl(bowl_local_xy)[needs_resample_bowl]

    if needs_resample_bowl.any():
        print(
            f"[WARNING] reset_bowl_and_cube: {needs_resample_bowl.sum().item()} env(s) failed to find "
            f"a valid bowl placement after {max_placement_tries} tries."
        )

    bx_local = bowl_local_xy[:, 0]  # (n,) bowl local x per env
    by_local = bowl_local_xy[:, 1]  # (n,) bowl local y per env

    # ── Cube constraint helpers (vectorised, per-env bowl position) ───────────
    def _in_occlusion_cone(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """True where cube at (cx,cy) is occluded behind the bowl from placement_point.

        Exact 2-D line-of-sight check matching debug/placement_constraints.py:
          perp distance from bowl centre to ray P→C < bowl_radius,
          AND bowl is in front of P (proj > 0),
          AND bowl is closer to P than cube is (proj < |P→C|).
        """
        vc_x = cx - px
        vc_y = cy - py
        vb_x = bx_local - px
        vb_y = by_local - py
        d_c = torch.sqrt(vc_x**2 + vc_y**2).clamp(min=1e-9)
        vc_hat_x = vc_x / d_c
        vc_hat_y = vc_y / d_c
        proj = vb_x * vc_hat_x + vb_y * vc_hat_y
        perp = torch.abs(vb_x * vc_hat_y - vb_y * vc_hat_x)
        return (perp < bowl_radius) & (proj > 0.0) & (proj < d_c)

    def _check_cube(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        d = torch.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        in_ring   = (d >= r_lo_cube) & (d <= r_hi_cube)
        x_ok      = cx >= cube_x_min
        excl_ok   = torch.sqrt((cx - bx_local) ** 2 + (cy - by_local) ** 2) > bowl_radius
        cone_ok   = ~_in_occlusion_cone(cx, cy)
        y_band_ok = torch.abs(cy) <= cube_y_max
        return in_ring & x_ok & excl_ok & cone_ok & y_band_ok

    # ── Cube Phase 1: random rejection sampling ───────────────────────────────
    cube_local_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
    needs_resample = ~_check_cube(cube_local_xy[:, 0], cube_local_xy[:, 1])

    for _ in range(safe_fallback_after):
        if not needs_resample.any():
            break
        new_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
        cube_local_xy[needs_resample] = new_xy[needs_resample]
        needs_resample[needs_resample.clone()] = ~_check_cube(
            cube_local_xy[:, 0], cube_local_xy[:, 1]
        )[needs_resample]

    # ── Cube Phase 2: safety positions fallback ───────────────────────────────
    if needs_resample.any():
        for sx, sy in safety_positions:
            if not needs_resample.any():
                break
            sp_x = torch.full((n,), float(sx), device=env.device)
            sp_y = torch.full((n,), float(sy), device=env.device)
            can_fix = needs_resample & _check_cube(sp_x, sp_y)
            if can_fix.any():
                cube_local_xy[can_fix, 0] = float(sx)
                cube_local_xy[can_fix, 1] = float(sy)
                needs_resample[can_fix] = False

    if needs_resample.any():
        print(
            f"[WARNING] reset_bowl_and_cube: {needs_resample.sum().item()} env(s) failed to find "
            f"a valid cube placement after {safe_fallback_after} random tries + "
            f"{len(safety_positions)} safety positions."
        )

    # ── Write bowl state to sim ───────────────────────────────────────────────
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    bowl_state[:, 0] = bowl_local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
    bowl_state[:, 1] = bowl_local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
    bowl_state[:, 2] = bowl.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
    bowl_state[:, 7:] = 0.0

    bowl.write_root_pose_to_sim(bowl_state[:, :7], env_ids=env_ids)
    bowl.write_root_velocity_to_sim(bowl_state[:, 7:], env_ids=env_ids)

    # ── Write cube state to sim ───────────────────────────────────────────────
    obj_state = obj.data.default_root_state[env_ids].clone()
    obj_state[:, 0] = cube_local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
    obj_state[:, 1] = cube_local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
    obj_state[:, 2] = obj.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]

    z_angle = math_utils.sample_uniform(
        cube_z_rotation_range[0], cube_z_rotation_range[1], (n,), device=env.device
    )
    z_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device),
        torch.zeros(n, device=env.device),
        z_angle,
    )
    obj_state[:, 3:7] = math_utils.quat_mul(z_quat, obj.data.default_root_state[env_ids, 3:7])
    obj_state[:, 7:] = 0.0

    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)

    # Store initial cube world position for the asymmetric actor observation.
    if not hasattr(env, "_initial_cube_pos_w"):
        env._initial_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._initial_cube_pos_w[env_ids] = obj_state[:, :3]


def reset_bowl_and_two_cubes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    placement_point: tuple[float, float] = (0.048, 0.0),
    bowl_dist_range: tuple[float, float] = (0.20, 0.40),
    bowl_x_min: float = 0.148,
    bowl_y_max: float = 0.20,
    bowl_radius: float = 0.14,
    cube_dist_range: tuple[float, float] = (0.15, 0.30),
    cube_x_min: float = 0.148,
    cube_y_max: float = 0.20,
    cluster_half_gap: float = 0.011,
    bowl_extra_margin: float = 0.005,
    safe_fallback_after: int = 100,
    max_placement_tries: int = 200,
    safety_positions: list | None = None,
    cube_z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
) -> None:
    """Reset bowl and two cubes (red/blue) adjacent in a cluster, with random target color.

    The two cubes are placed ± cluster_half_gap apart along the y-axis (transverse to the
    arm), centred on a cluster point sampled with the same constraints as a single cube.
    Which side gets the red cube and which gets blue is randomised per env each episode.
    target_color_id is sampled uniformly: 0 = red is target, 1 = blue is target.

    State stored on env each reset:
      env._target_color_id      : (num_envs,) int64   — 0=red, 1=blue
      env._initial_red_cube_pos_w : (num_envs, 3)     — red cube initial pos
      env._initial_blue_cube_pos_w: (num_envs, 3)     — blue cube initial pos
    """
    if safety_positions is None:
        safety_positions = [
            (0.268, +0.000),
            (0.253, +0.143),
            (0.253, -0.143),
            (0.293, +0.114),
            (0.293, -0.114),
            (0.338, +0.000),
            (0.189, +0.169),
            (0.189, -0.169),
        ]

    bowl: RigidObject = env.scene[bowl_cfg.name]
    red_obj: RigidObject = env.scene[red_object_cfg.name]
    blue_obj: RigidObject = env.scene[blue_object_cfg.name]

    n = len(env_ids)
    px = float(placement_point[0])
    py = float(placement_point[1])
    r_lo_bowl, r_hi_bowl = float(bowl_dist_range[0]), float(bowl_dist_range[1])
    r_lo_cube, r_hi_cube = float(cube_dist_range[0]), float(cube_dist_range[1])

    def _sample_annulus(count: int, r_lo: float, r_hi: float) -> torch.Tensor:
        angle = torch.rand(count, device=env.device) * (2.0 * math.pi)
        r = torch.sqrt(torch.rand(count, device=env.device) * (r_hi**2 - r_lo**2) + r_lo**2)
        return torch.stack([px + r * torch.cos(angle), py + r * torch.sin(angle)], dim=1)

    # ── Bowl: rejection-sample ────────────────────────────────────────────────
    def _check_bowl(local_xy: torch.Tensor) -> torch.Tensor:
        x, y = local_xy[:, 0], local_xy[:, 1]
        d = torch.sqrt((x - px) ** 2 + (y - py) ** 2)
        return (d >= r_lo_bowl) & (d <= r_hi_bowl) & (x >= bowl_x_min) & (torch.abs(y) <= bowl_y_max)

    bowl_local_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
    needs_resample_bowl = ~_check_bowl(bowl_local_xy)
    for _ in range(max_placement_tries):
        if not needs_resample_bowl.any():
            break
        new_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
        bowl_local_xy[needs_resample_bowl] = new_xy[needs_resample_bowl]
        needs_resample_bowl[needs_resample_bowl.clone()] = ~_check_bowl(bowl_local_xy)[needs_resample_bowl]

    if needs_resample_bowl.any():
        print(
            f"[WARNING] reset_bowl_and_two_cubes: {needs_resample_bowl.sum().item()} env(s) failed "
            f"bowl placement after {max_placement_tries} tries."
        )

    bx_local = bowl_local_xy[:, 0]
    by_local = bowl_local_xy[:, 1]

    # ── Cluster centre constraint helpers ────────────────────────────────────
    def _in_occlusion_cone(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        vc_x, vc_y = cx - px, cy - py
        vb_x, vb_y = bx_local - px, by_local - py
        d_c = torch.sqrt(vc_x**2 + vc_y**2).clamp(min=1e-9)
        vc_hat_x, vc_hat_y = vc_x / d_c, vc_y / d_c
        proj = vb_x * vc_hat_x + vb_y * vc_hat_y
        perp = torch.abs(vb_x * vc_hat_y - vb_y * vc_hat_x)
        return (perp < bowl_radius) & (proj > 0.0) & (proj < d_c)

    def _check_cluster_center(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        d = torch.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        in_ring = (d >= r_lo_cube) & (d <= r_hi_cube)
        x_ok = cx >= cube_x_min
        # Expand exclusion so both adjacent cubes clear the bowl with extra margin.
        excl_ok = torch.sqrt((cx - bx_local) ** 2 + (cy - by_local) ** 2) > (bowl_radius + cluster_half_gap + bowl_extra_margin)
        cone_ok = ~_in_occlusion_cone(cx, cy)
        y_band_ok = (torch.abs(cy) + cluster_half_gap) <= cube_y_max
        return in_ring & x_ok & excl_ok & cone_ok & y_band_ok

    # ── Phase 1: random cluster-centre sampling ───────────────────────────────
    cluster_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
    needs_resample = ~_check_cluster_center(cluster_xy[:, 0], cluster_xy[:, 1])

    for _ in range(safe_fallback_after):
        if not needs_resample.any():
            break
        new_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
        cluster_xy[needs_resample] = new_xy[needs_resample]
        needs_resample[needs_resample.clone()] = ~_check_cluster_center(
            cluster_xy[:, 0], cluster_xy[:, 1]
        )[needs_resample]

    # ── Phase 2: safety fallback ──────────────────────────────────────────────
    if needs_resample.any():
        for sx, sy in safety_positions:
            if not needs_resample.any():
                break
            sp_x = torch.full((n,), float(sx), device=env.device)
            sp_y = torch.full((n,), float(sy), device=env.device)
            can_fix = needs_resample & _check_cluster_center(sp_x, sp_y)
            if can_fix.any():
                cluster_xy[can_fix, 0] = float(sx)
                cluster_xy[can_fix, 1] = float(sy)
                needs_resample[can_fix] = False

    if needs_resample.any():
        print(
            f"[WARNING] reset_bowl_and_two_cubes: {needs_resample.sum().item()} env(s) failed "
            f"cluster placement after {safe_fallback_after} random tries + {len(safety_positions)} fallbacks."
        )

    # ── Sample one shared cluster rotation and derive cube positions ──────────
    # cluster_angle randomises the pair orientation in world space (uniform 0–2π).
    # swap independently randomises which color goes to which side so color
    # assignment is fully decorrelated from pair orientation.
    cluster_angle = math_utils.sample_uniform(
        cube_z_rotation_range[0], cube_z_rotation_range[1], (n,), device=env.device
    )
    gap = cluster_half_gap
    offset_x = -gap * torch.sin(cluster_angle)
    offset_y =  gap * torch.cos(cluster_angle)

    pos_a = torch.stack([cluster_xy[:, 0] + offset_x, cluster_xy[:, 1] + offset_y], dim=1)
    pos_b = torch.stack([cluster_xy[:, 0] - offset_x, cluster_xy[:, 1] - offset_y], dim=1)

    swap = (torch.rand(n, device=env.device) > 0.5).unsqueeze(1)  # [n, 1]
    red_local_xy  = torch.where(swap, pos_a, pos_b)
    blue_local_xy = torch.where(swap, pos_b, pos_a)

    cluster_z_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device), torch.zeros(n, device=env.device), cluster_angle
    )

    # ── Sample target color ──────────────────────────────────────────────────
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
    env._target_color_id[env_ids] = torch.randint(0, 2, (n,), device=env.device, dtype=torch.int64)

    # ── Write bowl state ──────────────────────────────────────────────────────
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    bowl_state[:, 0] = bowl_local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
    bowl_state[:, 1] = bowl_local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
    bowl_state[:, 2] = bowl.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
    bowl_state[:, 7:] = 0.0
    bowl.write_root_pose_to_sim(bowl_state[:, :7], env_ids=env_ids)
    bowl.write_root_velocity_to_sim(bowl_state[:, 7:], env_ids=env_ids)

    # ── Helper: write cube state (shared rotation) ────────────────────────────
    def _write_cube(obj: RigidObject, local_xy: torch.Tensor) -> torch.Tensor:
        state = obj.data.default_root_state[env_ids].clone()
        state[:, 0] = local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
        state[:, 1] = local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
        state[:, 2] = obj.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]
        state[:, 3:7] = math_utils.quat_mul(cluster_z_quat, obj.data.default_root_state[env_ids, 3:7])
        state[:, 7:] = 0.0
        obj.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        return state[:, :3]

    red_world_pos = _write_cube(red_obj, red_local_xy)
    blue_world_pos = _write_cube(blue_obj, blue_local_xy)

    # ── Store initial positions ───────────────────────────────────────────────
    if not hasattr(env, "_initial_red_cube_pos_w"):
        env._initial_red_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    if not hasattr(env, "_initial_blue_cube_pos_w"):
        env._initial_blue_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._initial_red_cube_pos_w[env_ids] = red_world_pos
    env._initial_blue_cube_pos_w[env_ids] = blue_world_pos


def _set_color_on_subtree(prim_pattern: str, color: tuple[float, float, float]) -> None:
    prims = find_matching_prims(prim_pattern)
    with Sdf.ChangeBlock():
        for root_prim in prims:
            if not root_prim.IsValid():
                continue
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsValid():
                    continue
                for attr_name in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
                    attr = prim.GetAttribute(attr_name)
                    if attr.IsValid():
                        attr.Set(color)
                        break


def set_bowl_color(
    env,
    env_ids: torch.Tensor | None,
    color: tuple[float, float, float] = _BOWL_BASE_COLOR,
) -> None:
    """Set bowl color once at startup (mode='startup'). Defaults to BOWL_BASE_COLOR (#d4be9f)."""
    _set_color_on_subtree("/World/envs/env_.*/Bowl", color)


def _sample_color_around_base(
    num: int,
    base_color: tuple[float, float, float],
    delta: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    """Sample RGB colors uniformly around base_color with per-channel half-width delta.

    Returns a plain list of (r, g, b) tuples in [0, 1] ready for USD attribute writes.
    """
    base = torch.tensor(base_color, dtype=torch.float32)
    d = torch.tensor(delta, dtype=torch.float32)
    noise = torch.rand(num, 3) * 2.0 - 1.0  # uniform in [-1, 1] per channel
    colors = (base + noise * d).clamp(0.0, 1.0)
    return [tuple(row.tolist()) for row in colors]


def randomize_cube_color(
    env,
    env_ids: torch.Tensor,
    prim_path: str = "/World/envs/env_.*/Object",
    base_color: tuple[float, float, float] = _CUBE_BASE_COLOR,
    delta: tuple[float, float, float] = (0.08, 0.03, 0.04),
    palette: list[tuple[float, float, float]] | None = None,
) -> None:
    """Randomize cube color at each episode reset (sim-to-real domain randomization).

    If palette is None, samples uniformly around base_color with per-channel delta.
    If palette is provided, samples uniformly from the discrete color list instead.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        prim_path: USD prim path template containing 'env_.*' for the cube.
        base_color: Centre of the colour band (RGB in [0, 1]).
        delta: Per-channel half-width of the uniform band.
        palette: If set, sample from this discrete list instead of the colour band.
    """
    num = len(env_ids)
    if palette is not None:
        palette_t = torch.tensor(palette, dtype=torch.float32)
        idx = torch.randint(0, len(palette_t), (num,)).tolist()
        colors = [tuple(palette_t[i].tolist()) for i in idx]
    else:
        colors = _sample_color_around_base(num, base_color, delta)

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids.tolist()):
            path = prim_path.replace("env_.*", f"env_{env_id}")
            _set_color_on_subtree(path, colors[i])


def randomize_table_color(
    env,
    env_ids: torch.Tensor,
    prim_path: str = "/World/envs/env_.*/Table",
    base_color: tuple[float, float, float] = _TABLE_BASE_COLOR,
    delta: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> None:
    """Randomize table surface color at each episode reset (sim-to-real domain randomization).

    Simulates variation in table material, wear, and lighting response.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        prim_path: USD prim path template containing 'env_.*' for the table.
        base_color: Centre of the colour band (RGB in [0, 1]).
        delta: Per-channel half-width of the uniform band (default ±5%).
    """
    colors = _sample_color_around_base(len(env_ids), base_color, delta)
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids.tolist()):
            path = prim_path.replace("env_.*", f"env_{env_id}")
            _set_color_on_subtree(path, colors[i])


def randomize_gripper_color(
    env,
    env_ids: torch.Tensor,
    base_color: tuple[float, float, float] = _GRIPPER_BASE_COLOR,
    delta: tuple[float, float, float] = (0.03, 0.03, 0.03),
) -> None:
    """Randomize wrist gripper link colors at each episode reset (sim-to-real domain randomization).

    Applies the same sampled colour to both gripper_link and moving_jaw_so101_v1_link so they
    look cohesive. Uses OmniPBR's inputs:diffuse_color_constant, which _set_color_on_subtree
    already handles. Base colours from the URDF: 3d_printed=(0.05,0.05,0.05),
    sts3215 servo=(0.1,0.1,0.1); GRIPPER_BASE_COLOR splits the difference at 0.07.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        base_color: Centre of the colour band (greyscale, RGB in [0, 1]).
        delta: Per-channel half-width of the uniform band (default ±3%).
    """
    colors = _sample_color_around_base(len(env_ids), base_color, delta)
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids.tolist()):
            _set_color_on_subtree(f"/World/envs/env_{env_id}/Robot/gripper_link", colors[i])
            _set_color_on_subtree(f"/World/envs/env_{env_id}/Robot/moving_jaw_so101_v1_link", colors[i])


def randomize_dome_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str = "/World/light",
    intensity_range: tuple[float, float] = (400.0, 1200.0),
    color_range: tuple[float, float] = (0.65, 0.85),
) -> None:
    """Randomize the dome light intensity and color temperature each episode.

    Args:
        env: The RL environment instance.
        env_ids: Unused (is_global_time=True passes None), but required by event manager signature.
        prim_path:      USD path of the DomeLight prim.
        intensity_range: (min, max) intensity. Default (400, 1200) stays around the 800 baseline.
        color_range:    (min, max) per-channel uniform value for the neutral grey color.
                        Slightly varying this simulates warm/cool ambient shifts.
    """
    intensity = random.uniform(*intensity_range)
    grey = random.uniform(*color_range)

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    with Sdf.ChangeBlock():
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr.IsValid():
            intensity_attr.Set(intensity)
        color_attr = prim.GetAttribute("inputs:color")
        if color_attr.IsValid():
            color_attr.Set(Gf.Vec3f(grey, grey, grey))


def randomize_sphere_light(
    env,
    env_ids: torch.Tensor | None,
    intensity_range: tuple[float, float] = (3000.0, 8000.0),
    color_range: tuple[float, float] = (0.65, 0.85),
    radius_range: tuple[float, float] = (0.1, 0.4),
    pos_x_range: tuple[float, float] = (-0.2, 0.5),
    pos_y_range: tuple[float, float] = (-0.4, 0.4),
    pos_z_range: tuple[float, float] = (0.3, 1.0),
) -> None:
    """Randomize per-env sphere light: intensity, color, radius, and position.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        intensity_range: (min, max) intensity.
        color_range: (min, max) per-channel uniform value for neutral white light.
        radius_range: (min, max) radius in meters.
        pos_x_range: (min, max) x position offset relative to env origin.
        pos_y_range: (min, max) y position offset relative to env origin.
        pos_z_range: (min, max) z position offset relative to env origin.
    """
    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for env_id in env_ids.tolist():
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/SphereLight")
            if not prim.IsValid():
                continue

            # Intensity
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if intensity_attr.IsValid():
                intensity_attr.Set(random.uniform(*intensity_range))

            # Color — sample a single greyscale value for all channels (neutral white light)
            c = random.uniform(*color_range)
            color_attr = prim.GetAttribute("inputs:color")
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(c, c, c))

            # Radius
            radius_attr = prim.GetAttribute("inputs:radius")
            if radius_attr.IsValid():
                radius_attr.Set(random.uniform(*radius_range))

            # Position
            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr.IsValid():
                translate_attr.Set(Gf.Vec3d(
                    random.uniform(*pos_x_range),
                    random.uniform(*pos_y_range),
                    random.uniform(*pos_z_range),
                ))


def randomize_distant_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str = "/World/DistantLight",
    intensity_range: tuple[float, float] = (1000.0, 3000.0),
    angle_range: tuple[float, float] = (30.0, 70.0),
    azimuth_range: tuple[float, float] = (0.0, 360.0),
) -> None:
    """Randomize the scene-wide distant light intensity and direction each episode.

    The distant light casts parallel shadows from all objects (robot arm, cube, bowl)
    regardless of their position — unlike a sphere light which only casts shadows when
    objects happen to lie between it and the surface.

    Elevation is sampled from angle_range (degrees above horizon) and azimuth from
    azimuth_range (degrees around the vertical axis), then converted to a USD xformOp
    rotation that points the light in the chosen direction.

    Args:
        env: The RL environment instance.
        env_ids: Unused (is_global_time=True passes None), required by event manager.
        prim_path: USD path of the DistantLight prim.
        intensity_range: (min, max) light intensity.
        angle_range: (min, max) elevation angle in degrees above the horizon.
        azimuth_range: (min, max) azimuth angle in degrees.
    """
    elevation = random.uniform(*angle_range)
    azimuth = random.uniform(*azimuth_range)

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    with Sdf.ChangeBlock():
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr.IsValid():
            intensity_attr.Set(random.uniform(*intensity_range))

        # DistantLight points along -Z by default; rotate to the chosen direction.
        # Pitch down from zenith by (90 - elevation), then spin around Z by azimuth.
        rot_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rot_attr.IsValid():
            rot_attr.Set(Gf.Vec3f(-(90.0 - elevation), 0.0, azimuth))


##
# Wrist camera domain randomization (sim-to-real)
##

def randomize_wrist_camera_intrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prim_path: str = "/World/envs/env_.*/Robot/gripper_link/wrist_camera",
    focal_length_noise_pct: float = 0.10,
) -> None:
    """Randomize camera intrinsic parameters on each reset (sim-to-real domain randomization).

    Only focal length is perturbed — it is the primary intrinsic that changes the camera FOV.
    Focus distance and f-stop are omitted because F_STOP=100 (pinhole) makes depth-of-field
    effects invisible in rendering.  Aperture is also omitted to avoid uncontrolled FOV
    compounding with the focal-length noise.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        prim_path: USD prim path template containing 'env_.*' for the camera.
        focal_length_noise_pct: Fractional noise for focal length (default ±10%).
    """
    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for env_id in env_ids.tolist():
            path = prim_path.replace("env_.*", f"env_{env_id}")
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            focal = _sample_with_noise(FOCAL_LENGTH_MM, focal_length_noise_pct)
            attr = prim.GetAttribute("focalLength")
            if attr.IsValid():
                attr.Set(focal)


def randomize_wrist_camera_extrinsics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    prim_path: str = "/World/envs/env_.*/Robot/gripper_link/wrist_camera",
    position_noise_m: tuple[float, float] = (-0.002, 0.002),
    rotation_noise_deg: tuple[float, float] = (-2.0, 2.0),
) -> None:
    """Randomize camera extrinsic pose (mount position + angle) on each reset.

    Simulates camera mount tolerance, cable-induced shifts, and mechanical flex.
    The noise is applied relative to the canonical offset defined in _wrist_cam.py
    (OFFSET_POS, OFFSET_QUAT_WXYZ) — the same values used by TiledCameraCfg.OffsetCfg.

    Rotation is composed as base * delta, so the delta axes are expressed in the
    parent (gripper_link) frame.

    Args:
        env: The RL environment instance.
        env_ids: Indices of environments to apply randomization to.
        prim_path: USD prim path template containing 'env_.*' for the camera mount XForm.
        position_noise_m: (min, max) position noise in meters per axis (default ±2mm).
        rotation_noise_deg: (min, max) rotation noise in degrees per axis (default ±2°).
    """
    stage = get_current_stage()

    # Canonical offset from _wrist_cam constants — same values as TiledCameraCfg.OffsetCfg.
    base_pos = OFFSET_POS        # (x, y, z) in gripper_link frame
    base_quat = OFFSET_QUAT_WXYZ  # (w, x, y, z)

    with Sdf.ChangeBlock():
        for env_id in env_ids.tolist():
            path = prim_path.replace("env_.*", f"env_{env_id}")
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue

            # Sample position offsets
            dx = base_pos[0] + random.uniform(*position_noise_m)
            dy = base_pos[1] + random.uniform(*position_noise_m)
            dz = base_pos[2] + random.uniform(*position_noise_m)

            # Sample rotation deltas (degrees → radians); compose base * delta
            roll_rad = math.radians(random.uniform(*rotation_noise_deg))
            pitch_rad = math.radians(random.uniform(*rotation_noise_deg))
            yaw_rad = math.radians(random.uniform(*rotation_noise_deg))

            delta_quat = math_utils.quat_from_euler_xyz(
                torch.tensor([roll_rad]),
                torch.tensor([pitch_rad]),
                torch.tensor([yaw_rad]),
            )[0]

            base_quat_tensor = torch.tensor([base_quat])  # shape (1, 4) wxyz
            final_quat = math_utils.quat_mul(base_quat_tensor, delta_quat.unsqueeze(0))[0]

            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr.IsValid():
                translate_attr.Set(Gf.Vec3d(dx, dy, dz))

            orient_attr = prim.GetAttribute("xformOp:orient")
            if orient_attr.IsValid():
                orient_attr.Set(
                    Gf.Quatd(
                        final_quat[0].item(),  # w
                        final_quat[1].item(),  # x
                        final_quat[2].item(),  # y
                        final_quat[3].item(),  # z
                    )
                )


def _sample_with_noise(baseline: float, noise_pct: float) -> float:
    """Sample a value perturbed by a uniform percentage deviation from baseline.

    Args:
        baseline: The nominal value.
        noise_pct: Fractional noise range (e.g., 0.10 for ±10%).

    Returns:
        Perturbed value in range [baseline * (1 - noise_pct), baseline * (1 + noise_pct)].
    """
    factor = 1.0 + random.uniform(-noise_pct, noise_pct)
    return baseline * factor


##
# Visual-coord color events
##

# Six target colors (RGB in [0, 1]).
# Index MUST match observations.random_target_color_one_hot:
#   0=blue  1=red  2=green  3=yellow  4=purple  5=orange
_TARGET_COLORS: list[tuple[float, float, float]] = [
    (0.12, 0.24, 0.87),  # 0: blue
    (0.87, 0.10, 0.10),  # 1: red
    (0.10, 0.78, 0.22),  # 2: green
    (0.95, 0.88, 0.05),  # 3: yellow
    (0.58, 0.10, 0.78),  # 4: purple
    (0.95, 0.50, 0.05),  # 5: orange
]


def set_two_cube_colors(
    env,
    env_ids: torch.Tensor,
) -> None:
    """Assign two distinct colors from the 6-color palette to the cube pair each reset.

    Picks two different colors, assigns them to object_red and object_blue prims,
    and randomly selects which physical cube is the target for this episode.

    Stores on env (indexed by env_ids):
      env._target_is_red    : (num_envs,) bool   — True when object_red is the target
      env._target_color_id  : (num_envs,) int64  — 0-5 color index of the TARGET cube
    """
    n = len(env_ids)

    # Sample color A for each env; color B is a different index
    color_a = torch.randint(0, 6, (n,), device=env.device)
    shift = torch.randint(1, 6, (n,), device=env.device)
    color_b = (color_a + shift) % 6

    # Randomly assign which physical cube (red/blue entity) gets color A vs B
    swap = torch.rand(n, device=env.device) > 0.5   # True → object_red gets color_a
    red_color_idx  = torch.where(swap, color_a, color_b)
    blue_color_idx = torch.where(swap, color_b, color_a)

    # Randomly choose which physical cube is the target
    target_is_red = torch.rand(n, device=env.device) > 0.5   # True → object_red is target
    target_color_idx = torch.where(target_is_red, red_color_idx, blue_color_idx)

    if not hasattr(env, "_target_is_red"):
        env._target_is_red = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)

    env._target_is_red[env_ids]   = target_is_red
    env._target_color_id[env_ids] = target_color_idx

    # Apply visuals per env
    red_idx_cpu  = red_color_idx.cpu().tolist()
    blue_idx_cpu = blue_color_idx.cpu().tolist()

    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids.tolist()):
            _set_color_on_subtree(
                f"/World/envs/env_{env_id}/Object",
                _TARGET_COLORS[red_idx_cpu[i]],
            )
            _set_color_on_subtree(
                f"/World/envs/env_{env_id}/ObjectBlue",
                _TARGET_COLORS[blue_idx_cpu[i]],
            )
