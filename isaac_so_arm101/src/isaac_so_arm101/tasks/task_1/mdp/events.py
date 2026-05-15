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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_bowl_and_cube(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    bowl_pose_range: dict[str, tuple[float, float]],
    cube_world_range: dict[str, tuple[float, float]],
    exclusion_radius: float = 0.10,
    exclusion_shape: str = "circle",
    y_occlusion_threshold: float = 0.20,
    max_placement_tries: int = 100,
    cube_z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the bowl to a randomised position, then place the cube with constraints.

    ── Coordinate system ────────────────────────────────────────────────────────────────
    Both ``bowl_pose_range`` and ``cube_world_range`` are expressed in each environment's
    LOCAL frame — i.e. relative to the robot base / env origin.  This matches the x/y axes
    in ``debug/cube_placement_constraints.py`` and ``debug/visualize_cube_placements.py``;
    run either script to visualise valid/invalid regions before changing parameters.

        bowl_pose_range  : OFFSET added to the bowl's ``init_state`` position (local frame).
                           {"x": (-0.05, 0.10), "y": (-0.20, 0.20)} with bowl init_state
                           x=0.30 → bowl local x ∈ [0.25, 0.40].  Missing keys → no offset.

        cube_world_range : ABSOLUTE sampling rectangle for the cube in local frame.
                           {"x": (0.10, 0.40), "y": (-0.35, 0.35)} is the current range.
                           Widen if the rejection sampler warns; narrow to reduce diversity.

    ── Cube placement constraints ────────────────────────────────────────────────────────
    Let ox = cube_x − bowl_x,  oy = cube_y − bowl_y  (both in local frame).

    C1  Exclusion zone   controlled by exclusion_radius and exclusion_shape.
        "circle" (default): dist(cube, bowl) > exclusion_radius  (radial, no sqrt at runtime)
        "box":              |ox| > exclusion_radius OR |oy| > exclusion_radius
                            (axis-aligned square, side = 2 × exclusion_radius)
        Bowl physical radius ≈ 0.075 m at scale 1.35; default 0.10 m adds ~2.5 cm margin.
        ↑ Increase exclusion_radius if the cube initialises inside the bowl.
        ↓ Decrease to allow tighter initial separations.
        Switch exclusion_shape to match what is configured in debug/cube_placement_constraints.py.

    C2  X occlusion      (|oy| > y_occlusion_threshold) OR (ox ≤ 0)
        When the cube is within y_occlusion_threshold metres of the bowl in y — i.e. it is
        "in line" with the bowl along x — the cube must not be behind the bowl (ox ≤ 0 means
        cube_x ≤ bowl_x, so the cube is between the robot and the bowl, not beyond it).
        Outside that y band the cube is off to the side and any x is allowed.
        ↑ Larger threshold → x constraint applies over a wider y band (fewer valid positions).
        ↓ Smaller threshold → x constraint lifts sooner; more valid positions at large |oy|.

    C3  Y occlusion      (no free parameter — geometry only)
        The cube must not be further from the robot's centre-line in y than the bowl itself.
        bowl_y < 0  →  cube_y ≥ bowl_y  (cube stays on the positive-y side of the bowl)
        bowl_y > 0  →  cube_y ≤ bowl_y  (cube stays on the negative-y side of the bowl)
        bowl_y = 0  →  no constraint
        To disable C3 entirely, replace ``c3`` with ``torch.ones(n, dtype=torch.bool, device=env.device)``.

    Uses per-env rejection sampling. Emits a warning if placement fails after max_placement_tries.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    range_keys = ["x", "y", "z"]

    # ── Bowl ──────────────────────────────────────────────────────────────────
    # Start from the bowl's default (init_state) pose and add the per-env world origin
    # so that the subsequent offset is applied in local (robot-relative) space.
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    bowl_state[:, :3] += env.scene.env_origins[env_ids]

    # Sample a uniform offset in each configured axis and shift the bowl position.
    # Keys not present in bowl_pose_range default to (0.0, 0.0) → no randomisation.
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

    # Only x and y are sampled; z always equals the object's default height (table surface).
    # cube_ranges shape: (2, 2) — row 0: x [min, max], row 1: y [min, max].
    cube_ranges = torch.tensor(
        [cube_world_range.get(k, (0.0, 0.0)) for k in ["x", "y"]], device=env.device
    )

    n = len(env_ids)

    # Convert bowl world position → local frame so constraint checks and cube_world_range
    # sampling share the same robot-relative coordinate space.
    bx_local = bowl_state[:, 0] - env.scene.env_origins[env_ids, 0]  # (n,) bowl local x
    by_local = bowl_state[:, 1] - env.scene.env_origins[env_ids, 1]  # (n,) bowl local y

    def check_validity(local_xy):
        # Signed displacement of cube FROM bowl centre (local frame).
        # ox > 0 → cube is further from robot in x than the bowl (behind bowl).
        # oy > 0 → cube is to the +y side of the bowl.
        ox = local_xy[:, 0] - bx_local
        oy = local_xy[:, 1] - by_local

        # C1 — Exclusion zone: shape controlled by exclusion_shape parameter.
        # "circle": squared-distance (avoids sqrt) — matches the debug visualisation default.
        # "box":    axis-aligned square of side 2 × exclusion_radius.
        if exclusion_shape == "circle":
            c1 = (ox**2 + oy**2) > exclusion_radius**2
        else:  # "box"
            c1 = (torch.abs(ox) > exclusion_radius) | (torch.abs(oy) > exclusion_radius)

        # C2 — X occlusion: enforce cube_x ≤ bowl_x only when cube is "in line" with bowl in y.
        # If |oy| > y_occlusion_threshold the cube is off to the side → x is unconstrained.
        # Tune y_occlusion_threshold (default 0.20 m); visualise in debug/cube_placement_constraints.py.
        c2 = (torch.abs(oy) > y_occlusion_threshold) | (ox <= 0.0)

        # C3 — Y occlusion: cube must not be on the far y side of the bowl from the robot.
        # Expressed via De Morgan implications (avoids branching over the batch dimension):
        #   ~(by_local < 0) | (oy >= 0)  ≡  if bowl_y < 0: require oy ≥ 0
        #   ~(by_local > 0) | (oy <= 0)  ≡  if bowl_y > 0: require oy ≤ 0
        #   both trivially True when by_local == 0 → no constraint on the centre-line.
        # To disable C3, replace with: torch.ones(n, dtype=torch.bool, device=env.device)
        c3 = (~(by_local < 0) | (oy >= 0)) & (~(by_local > 0) | (oy <= 0))

        return c1 & c2 & c3

    # Initial candidate positions sampled uniformly from cube_world_range (local frame).
    cube_local_xy = math_utils.sample_uniform(
        cube_ranges[:, 0], cube_ranges[:, 1], (n, 2), device=env.device
    )
    needs_resample = ~check_validity(cube_local_xy)

    # Rejection sampling: only invalid envs are resampled each iteration so that already-valid
    # placements are never disturbed.  If warnings appear regularly, widen cube_world_range
    # or relax a constraint rather than increasing max_placement_tries.
    for _ in range(max_placement_tries):
        if not needs_resample.any():
            break
        new_xy = math_utils.sample_uniform(
            cube_ranges[:, 0], cube_ranges[:, 1], (n, 2), device=env.device
        )
        # Write new candidates only into the slots that still need resampling.
        cube_local_xy[needs_resample] = new_xy[needs_resample]
        # Re-evaluate validity for the resampled envs only; .clone() prevents in-place
        # index aliasing when the mask is both read and written in the same expression.
        needs_resample[needs_resample.clone()] = ~check_validity(cube_local_xy)[needs_resample]

    if needs_resample.any():
        print(
            f"[WARNING] reset_bowl_and_cube: {needs_resample.sum().item()} env(s) failed to find "
            f"a valid cube placement after {max_placement_tries} tries. "
            f"Consider widening cube_world_range."
        )

    # Convert local → world frame (add per-env origin) before writing positions to the sim.
    # Z is taken directly from the object's default state — the table surface height.
    obj_state[:, 0] = cube_local_xy[:, 0] + env.scene.env_origins[env_ids, 0]
    obj_state[:, 1] = cube_local_xy[:, 1] + env.scene.env_origins[env_ids, 1]
    obj_state[:, 2] = obj.data.default_root_state[env_ids, 2] + env.scene.env_origins[env_ids, 2]

    # Randomize cube orientation around the z-axis.
    z_angle = math_utils.sample_uniform(
        cube_z_rotation_range[0], cube_z_rotation_range[1], (n,), device=env.device
    )
    z_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device),
        torch.zeros(n, device=env.device),
        z_angle,
    )
    obj_state[:, 3:7] = math_utils.quat_mul(
        z_quat, obj.data.default_root_state[env_ids, 3:7]
    )

    obj_state[:, 7:] = 0.0  # zero initial velocity

    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)

    # Store the initial cube world position so the asymmetric actor can observe it
    # as a fixed reference throughout the episode (read by initial_object_position_in_robot_root_frame).
    if not hasattr(env, "_initial_cube_pos_w"):
        env._initial_cube_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    env._initial_cube_pos_w[env_ids] = obj_state[:, :3]

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
    color: tuple[float, float, float] = (212 / 255, 190 / 255, 159 / 255),
) -> None:
    """Set bowl color once at startup (mode='startup'). HEX #d4be9f by default."""
    _set_color_on_subtree("/World/envs/env_.*/Bowl", color)


def randomize_dome_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str = "/World/light",
    intensity_range: tuple[float, float] = (400.0, 1200.0),
    color_range: tuple[float, float] = (0.65, 0.85),
) -> None:
    """Randomize the dome light intensity and color temperature each episode.

    Args:
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


# def randomize_sphere_light(
#     env,
#     env_ids: torch.Tensor | None,
#     intensity_range: tuple[float, float] = (200.0, 1500.0),
#     pos_x_range: tuple[float, float] = (-0.1, 0.4),
#     pos_y_range: tuple[float, float] = (-0.3, 0.3),
#     pos_z_range: tuple[float, float] = (0.3, 0.8),
# ) -> None:
#     """Randomize per-env sphere light position and intensity independently at every reset.

#     Each resetting environment gets its own independently sampled light — cheap because
#     it writes exactly two attributes (intensity + translate) per env prim.
#     """
#     stage = get_current_stage()
#     with Sdf.ChangeBlock():
#         for env_id in env_ids.tolist():
#             prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/SphereLight")
#             if not prim.IsValid():
#                 continue
#             intensity_attr = prim.GetAttribute("inputs:intensity")
#             if intensity_attr.IsValid():
#                 intensity_attr.Set(random.uniform(*intensity_range))
#             translate_attr = prim.GetAttribute("xformOp:translate")
#             if translate_attr.IsValid():
#                 translate_attr.Set(Gf.Vec3d(
#                     random.uniform(*pos_x_range),
#                     random.uniform(*pos_y_range),
#                     random.uniform(*pos_z_range),
#                 ))

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
    """Randomize per-env sphere light: intensity, color, radius, and position."""
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