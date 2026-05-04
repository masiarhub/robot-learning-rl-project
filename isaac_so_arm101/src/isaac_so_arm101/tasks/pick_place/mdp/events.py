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
    obj_state[:, 7:] = 0.0  # zero initial velocity

    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)


def set_robot_color_black(
    env,
    env_ids: torch.Tensor | None,
    color: tuple[float, float, float] = (0.08, 0.08, 0.08),
) -> None:
    """Set the robot color to black on every reset.

    Traverses all shader prims under the Robot prim and sets whichever diffuse
    attribute exists — ``inputs:diffuse_color_constant`` (OmniPBR) or
    ``inputs:diffuseColor`` (UsdPreviewSurface).
    Prints a warning on the first call if no shader attributes are found.
    """
    robot = env.scene["robot"]
    robot_pattern = robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    robot_prims = find_matching_prims(robot_pattern)

    n_set = 0
    with Sdf.ChangeBlock():
        for robot_prim in robot_prims:
            if not robot_prim.IsValid():
                continue
            for prim in Usd.PrimRange(robot_prim):
                if not prim.IsValid():
                    continue
                for attr_name in ("inputs:diffuse_color_constant", "inputs:diffuseColor"):
                    attr = prim.GetAttribute(attr_name)
                    if attr.IsValid():
                        attr.Set(color)
                        n_set += 1
                        break

    if n_set == 0:
        print("[WARNING] set_robot_color_black: no diffuse shader attributes found under Robot prim.")


def randomize_directional_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str = "/World/light_directional",
    elevation_range: tuple[float, float] = (30.0, 70.0),
    intensity_range: tuple[float, float] = (1500.0, 2500.0),
) -> None:
    """Randomize the direction and intensity of the distant key light each episode.

    Azimuth is sampled uniformly over the full circle so shadows fall from any direction.
    Elevation is sampled within ``elevation_range`` degrees from vertical (90° = horizontal,
    0° = straight down) to keep the light roughly overhead.

    Args:
        prim_path:       USD path of the DistantLight prim.
        elevation_range: (min, max) degrees from vertical. Default (30, 70) keeps light overhead.
        intensity_range: (min, max) light intensity in nits.
    """
    azimuth = random.uniform(0.0, 2.0 * math.pi)  # random full-circle yaw
    elevation_rad = math.radians(random.uniform(*elevation_range))

    # Build quaternion: tilt from vertical (around X), then yaw (around Z).
    # wxyz convention throughout.
    e2, a2 = elevation_rad / 2.0, azimuth / 2.0
    # Elevation quaternion (rotate around X)
    qe = (math.cos(e2), math.sin(e2), 0.0, 0.0)
    # Azimuth quaternion (rotate around Z)
    qa = (math.cos(a2), 0.0, 0.0, math.sin(a2))
    # Compose: qa * qe  (elevation applied first, then yaw)
    w1, x1, y1, z1 = qa
    w2, x2, y2, z2 = qe
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    intensity = random.uniform(*intensity_range)

    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    with Sdf.ChangeBlock():
        orient_attr = prim.GetAttribute("xformOp:orient")
        if orient_attr.IsValid():
            orient_attr.Set(Gf.Quatd(w, x, y, z))
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr.IsValid():
            intensity_attr.Set(intensity)


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
