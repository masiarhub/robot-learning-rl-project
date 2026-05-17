#!/usr/bin/env python3
"""Visualise bowl and cube placement constraints for the SO-ARM101 Isaac Lab task.

Bowl is placed first in an annular ring around the placement point (first revolute
joint).  Cube is then placed subject to: annular ring, x_min, bowl exclusion box,
exact angular occlusion cone (line-of-sight from placement point through bowl disk),
and optional y-band constraints.

Background colour shows valid/invalid constraint regions for the cube.
Dots show rejection-sampled positions: green = valid, red × = failed.

To preview a parameter change:
  1. Edit the CONFIG block below.
  2. Re-run:  python3 debug/placement_constraints.py
  Output:    debug/placement_constraints.png
"""

import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

# ══ CONFIG ════════════════════════════════════════════════════════════════════

# Table extent (metres) — x = depth (robot sits at low-x end), y = lateral width
TABLE_X = (0.0, 0.8)
TABLE_Y = (-0.6, 0.6)

# Robot footprint (visualisation only)
ROBOT_SIZE   = 0.048          # side length (m) of square footprint
ROBOT_CENTER = (0.024, 0.0)   # centre in world frame

# Placement point — first revolute joint; all radial distances measured from here
PLACEMENT_POINT = (0.048, 0.0)

# ── Bowl ──────────────────────────────────────────────────────────────────────
BOWL_PHYSICAL_RADIUS = 0.0775  # real physical bowl radius (m)
CONE_RADIUS_MIN   = 0.12     # minimum cone half-width (diameter 0.24 m) — safety margin even at closest bowl position
BOWL_RADIUS       = 0.16     # maximum cone half-width (m) — at furthest bowl position, full safety margin
BOWL_DIST_RANGE   = (0.20, 0.40)   # annular ring radii from placement point (m)
BOWL_X_MIN        = 0.148    # absolute world-x lower bound (= placement_pt_x + 0.10)
BOWL_Y_CONSTRAINT = True           # optional: |bowl_y| ≤ BOWL_Y_MAX
BOWL_Y_MAX        = 0.20           # (m)

# Inner boundary of bowl ring is an ELLIPSE (not a circle) to prevent the bowl
# from being placed right in front of the robot.
#   At y=0 (directly ahead):  bowl world-x must be > BOWL_INNER_X_MIN
#   As |y| grows the x-constraint relaxes (ellipse shape).
# Ellipse centred at placement point:
#   semi-axis x = BOWL_INNER_X_MIN - px   (0.30 - 0.048 = 0.252 m)
#   semi-axis y = BOWL_INNER_SEMI_Y       (0.30 m — at x=placement_pt, bowl must be ≥ 0.30 m lateral)
# Validity: ((bowl_x - px) / (BOWL_INNER_X_MIN - px))^2 + ((bowl_y - py) / BOWL_INNER_SEMI_Y)^2 >= 1
BOWL_INNER_X_MIN  = 0.30    # (m)  minimum bowl world-x when bowl_y = 0
BOWL_INNER_SEMI_Y = 0.30    # (m)  lateral semi-axis of inner ellipse (relaxation in y)

# ── Cube ──────────────────────────────────────────────────────────────────────
CUBE_HALF_SIZE    = 0.01     # half side-length (m)  [2 cm cube]
CUBE_DIST_RANGE   = (0.15, 0.40)   # annular ring radii from placement point (m)
CUBE_X_MIN        = 0.198    # absolute world-x lower bound (= placement_pt_x + 0.10)
CUBE_Y_CONSTRAINT = True           # optional: linear |cube_y| ≤ y_max(x)

# y-band grows linearly with x: tight near the robot, wider at the far table edge.
#   |y| ≤  Y_MAX_LOW   at  x = CUBE_X_MIN   (close to robot)
#   |y| ≤  Y_MAX_HIGH  at  x = TABLE_X[1]   (far table edge)
# Rationale: camera FOV footprint on the table widens with distance, so positions
# farther away are more likely to be visible even when displaced laterally.
Y_MAX_LOW  = 0.20   # (m)  y limit at CUBE_X_MIN
Y_MAX_HIGH = 0.30   # (m)  y limit at the far table edge  (TABLE_X[1])

# ── Rejection sampling ────────────────────────────────────────────────────────
MAX_PLACEMENT_TRIES  = 200
SAFE_FALLBACK_AFTER  = 100   # after this many random tries, attempt safety positions
N_SAMPLES            = 200   # cube positions sampled per bowl test location
RANDOM_SEED          = 42

# ── Safety positions ──────────────────────────────────────────────────────────
# Tried in order once SAFE_FALLBACK_AFTER random attempts are exhausted.
# Must each satisfy: dist ∈ CUBE_DIST_RANGE, x ≥ CUBE_X_MIN, |y| ≤ y_max(x).
# They are NOT pre-checked against bowl constraints (that varies per episode);
# the first one that passes check_validity() for the current bowl is used.
# Defined as (x, y) world coordinates — edit to match your workspace.
SAFETY_POSITIONS = [
    (0.268, +0.000),   # straight ahead, mid-range
    (0.253, +0.143),   # slight left
    (0.253, -0.143),   # slight right
    (0.293, +0.114),   # left, further
    (0.293, -0.114),   # right, further
    (0.338, +0.000),   # straight ahead, far
    (0.189, +0.169),   # left, close
    (0.189, -0.169),   # right, close
]

# ══ BOWL TEST POSITIONS ═══════════════════════════════════════════════════════
# Direct (x, y) positions in local robot frame — sampled from the actual bowl
# placement rectangle: x ∈ [0.25, 0.40], y ∈ [-0.20, +0.20]
# (bowl init_state x=0.30 + offset (-0.05, +0.10); y=0.0 + offset (-0.20, +0.20))
BOWL_POSITIONS = [
    (0.32,  0.00),   # close x, on axis       — near ellipse boundary at y=0
    (0.44,  0.00),   # far x, on axis          — near outer ring boundary at y=0
    (0.35, +0.20),   # mid x, max +y           — upper y limit
    (0.35, -0.20),   # mid x, max -y           — lower y limit (symmetric)
    (0.38, +0.13),   # far-ish x, moderate +y  — interior of valid zone
    (0.30, +0.18),   # near x, large +y        — corner of ellipse + y-band
]
_px, _py = float(PLACEMENT_POINT[0]), float(PLACEMENT_POINT[1])
_table_edge_x = float(TABLE_X[1])
# Max reachable bowl x along the x-axis (placement_point + outer ring radius).
# Used as the upper anchor for the cone-radius interpolation.
_bowl_x_max = _px + BOWL_DIST_RANGE[1]
# x semi-axis of the elliptical inner bowl boundary (distance from placement point).
_bowl_inner_semi_x = BOWL_INNER_X_MIN - _px   # = 0.30 - 0.048 = 0.252 m

# ══ GRID ══════════════════════════════════════════════════════════════════════
RES = 600

_xs = torch.linspace(TABLE_X[0], TABLE_X[1], RES)
_ys = torch.linspace(TABLE_Y[0], TABLE_Y[1], RES)
_YY_t, _XX_t = torch.meshgrid(_ys, _xs, indexing="ij")   # (RES, RES)

_on_table = (
    (_XX_t >= TABLE_X[0]) & (_XX_t <= TABLE_X[1]) &
    (_YY_t >= TABLE_Y[0]) & (_YY_t <= TABLE_Y[1])
)

# ══ CONSTRAINT LOGIC ══════════════════════════════════════════════════════════

def _bowl_inner_valid(bowl_x: float, bowl_y: float) -> bool:
    """True when the bowl centre is OUTSIDE the elliptical inner exclusion zone.

    Ellipse centred at placement point:
        semi-axis x = _bowl_inner_semi_x = BOWL_INNER_X_MIN - px  (0.252 m)
            → at y=0, bowl world-x must exceed BOWL_INNER_X_MIN = 0.30 m
        semi-axis y = BOWL_INNER_SEMI_Y  (0.30 m)
            → relaxation: as |y| grows, required x decreases
    """
    dx = bowl_x - _px
    dy = bowl_y - _py
    return (dx / _bowl_inner_semi_x) ** 2 + (dy / BOWL_INNER_SEMI_Y) ** 2 >= 1.0


def _bowl_cone_radius(bowl_x: float) -> float:
    """Effective occlusion-cone half-width, linear in bowl x.

    CONE_RADIUS_MIN at bowl_x = BOWL_X_MIN (close bowl, diameter 0.20 m minimum safety).
    BOWL_RADIUS at bowl_x = _bowl_x_max (far bowl, full safety margin).
    """
    t = (bowl_x - BOWL_X_MIN) / (_bowl_x_max - BOWL_X_MIN)
    t = max(0.0, min(1.0, t))
    return CONE_RADIUS_MIN + (BOWL_RADIUS - CONE_RADIUS_MIN) * t


def _in_occlusion_cone(
    cx: torch.Tensor,
    cy: torch.Tensor,
    bowl_x: float,
    bowl_y: float,
    cone_radius: float,
) -> torch.Tensor:
    """True where the cube position is occluded behind the bowl from the placement
    point.

    Exact 2-D line-of-sight check: the cube at C is occluded when the ray from
    placement point P through C passes through the bowl disk of radius cone_radius.
    Three conditions must all hold:
      1. The perpendicular distance from bowl centre B to the ray P→C < cone_radius.
      2. The scalar projection of P→B onto the unit direction P→C is positive
         (bowl is in front of P, not behind).
      3. That projection is less than |P→C| (bowl is closer to P than cube is).

    Handles any shape of cx/cy via broadcasting (grid or batch).
    """
    vc_x = cx - _px          # vector P → C  (shape of cx)
    vc_y = cy - _py

    vb_x = bowl_x - _px      # vector P → B  (scalar)
    vb_y = bowl_y - _py

    d_c = torch.sqrt(vc_x**2 + vc_y**2).clamp(min=1e-9)
    vc_hat_x = vc_x / d_c    # unit P → C
    vc_hat_y = vc_y / d_c

    # Scalar projection of P→B onto unit P→C
    proj = vb_x * vc_hat_x + vb_y * vc_hat_y

    # Perpendicular distance from B to the ray P→C  (2-D cross-product magnitude)
    perp = torch.abs(vb_x * vc_hat_y - vb_y * vc_hat_x)

    return (perp < cone_radius) & (proj > 0.0) & (proj < d_c)


def _cube_y_max_at(cx: torch.Tensor) -> torch.Tensor:
    """Linear y limit: Y_MAX_LOW at CUBE_X_MIN, Y_MAX_HIGH at the far table edge."""
    t = ((cx - CUBE_X_MIN) / (_table_edge_x - CUBE_X_MIN)).clamp(0.0, 1.0)
    return Y_MAX_LOW + (Y_MAX_HIGH - Y_MAX_LOW) * t


def _cube_constraints(
    cx: torch.Tensor,
    cy: torch.Tensor,
    bowl_x: float,
    bowl_y: float,
) -> dict[str, torch.Tensor]:
    """Evaluate each cube constraint individually.

    Returns a dict of bool tensors with the same shape as cx/cy:
        radius  — inside annular ring [CUBE_DIST_RANGE]
        x_min   — cube_x ≥ CUBE_X_MIN
        box     — outside bowl exclusion circle
        cone    — NOT in occlusion cone
        y_band  — |cube_y| ≤ y_max(x)  (always True when CUBE_Y_CONSTRAINT=False)
    """
    cone_r = _bowl_cone_radius(bowl_x)
    d = torch.sqrt((cx - _px)**2 + (cy - _py)**2)
    return {
        "radius": (d >= CUBE_DIST_RANGE[0]) & (d <= CUBE_DIST_RANGE[1]),
        "x_min":  cx >= CUBE_X_MIN,
        "box":    torch.sqrt((cx - bowl_x)**2 + (cy - bowl_y)**2) > cone_r,
        "cone":   ~_in_occlusion_cone(cx, cy, bowl_x, bowl_y, cone_r),
        "y_band": (torch.abs(cy) <= _cube_y_max_at(cx)) if CUBE_Y_CONSTRAINT
                  else torch.ones_like(cx, dtype=torch.bool),
    }


def bowl_valid_grid() -> torch.Tensor:
    """(RES, RES) bool: where the bowl centre can legally be placed.

    Constraints (independent of cube position):
        outer ring  — distance from placement point ≤ BOWL_DIST_RANGE[1]
        inner ellipse — outside the elliptical exclusion zone
        x_min       — world-x ≥ BOWL_X_MIN
        y_band      — |world-y| ≤ BOWL_Y_MAX  (if BOWL_Y_CONSTRAINT)
    """
    dx = _XX_t - _px
    dy = _YY_t - _py
    r  = torch.sqrt(dx**2 + dy**2)
    in_outer = r <= BOWL_DIST_RANGE[1]
    outside_ellipse = (dx / _bowl_inner_semi_x) ** 2 + (dy / BOWL_INNER_SEMI_Y) ** 2 >= 1.0
    x_ok = _XX_t >= BOWL_X_MIN
    y_ok = (torch.abs(_YY_t) <= BOWL_Y_MAX) if BOWL_Y_CONSTRAINT \
           else torch.ones_like(dx, dtype=torch.bool)
    return _on_table & in_outer & outside_ellipse & x_ok & y_ok


def check_validity_grid(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate cube constraints on the (RES, RES) grid.

    Returns:
        valid    (RES, RES) bool — all constraints pass
        in_box   (RES, RES) bool — inside bowl exclusion box
        in_cone  (RES, RES) bool — in occlusion cone AND NOT in_box
    """
    c = _cube_constraints(_XX_t, _YY_t, bowl_x, bowl_y)
    in_box  = ~c["box"]
    in_cone = c["box"] & ~c["cone"]
    valid   = c["radius"] & c["x_min"] & c["box"] & c["cone"] & c["y_band"]
    return valid, in_box, in_cone


def sample_cube_positions(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rejection-sample N_SAMPLES cube positions for one bowl location.

    Two-phase strategy:
      Phase 1 — up to SAFE_FALLBACK_AFTER random draws from the annulus.
      Phase 2 — for any still-invalid samples, try each entry in SAFETY_POSITIONS
                 in order until one passes check_validity() for the current bowl.
                 Because the bowl is the same for all samples, one valid safety
                 position resolves all remaining failures at once.

    Returns:
        xy           (N_SAMPLES, 2)  final positions in world frame
        failed       (N_SAMPLES,)   True where even safety positions did not help
        used_safety  (N_SAMPLES,)   True where a safety position was used
    """
    n = N_SAMPLES
    r_lo, r_hi = CUBE_DIST_RANGE

    def _sample_annulus(count: int) -> torch.Tensor:
        angle = torch.rand(count) * (2.0 * math.pi)
        r = torch.sqrt(torch.rand(count) * (r_hi**2 - r_lo**2) + r_lo**2)
        return torch.stack([_px + r * torch.cos(angle),
                            _py + r * torch.sin(angle)], dim=1)

    def check_validity(xy: torch.Tensor) -> torch.Tensor:
        c = _cube_constraints(xy[:, 0], xy[:, 1], bowl_x, bowl_y)
        return c["radius"] & c["x_min"] & c["box"] & c["cone"] & c["y_band"]

    xy = _sample_annulus(n)
    needs_resample = ~check_validity(xy)
    used_safety = torch.zeros(n, dtype=torch.bool)

    # Phase 1: random rejection sampling
    for _ in range(SAFE_FALLBACK_AFTER):
        if not needs_resample.any():
            break
        new_xy = _sample_annulus(n)
        xy[needs_resample] = new_xy[needs_resample]
        # .clone() prevents in-place index aliasing when updating needs_resample
        needs_resample[needs_resample.clone()] = ~check_validity(xy)[needs_resample]

    # Phase 2: safety position fallback for remaining failures
    if needs_resample.any():
        for sx, sy in SAFETY_POSITIONS:
            sp = torch.tensor([[sx, sy]])
            if check_validity(sp)[0]:
                # Same bowl → same validity for every remaining failed sample
                count = int(needs_resample.sum().item())
                xy[needs_resample] = sp.expand(count, 2)
                used_safety[needs_resample] = True
                needs_resample[:] = False
                break

    return xy, needs_resample, used_safety


# ══ COLOUR SCHEME ════════════════════════════════════════════════════════════
# 0=off-table | 1=valid | 2=out-of-range | 3=occluded | 4=bowl-exclusion-box
CMAP = ListedColormap(["#d4d4d4", "#81c784", "#b0bec5", "#ffb74d", "#e57373"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CMAP.N)

# ══ PLOT ══════════════════════════════════════════════════════════════════════
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Cube Placement Constraints  |  "
    f"bowl_dist={BOWL_DIST_RANGE} m   cube_dist={CUBE_DIST_RANGE} m   "
    f"physical_r={BOWL_PHYSICAL_RADIUS} m   cone_r={CONE_RADIUS_MIN}→{BOWL_RADIUS} m (linear in bowl_x)\n"
    f"placement_pt={PLACEMENT_POINT}   bowl inner ellipse: x>{BOWL_INNER_X_MIN} m at y=0  semi_y={BOWL_INNER_SEMI_Y} m   "
    f"cube_x_min={CUBE_X_MIN} m   "
    f"y_band={'ON' if CUBE_Y_CONSTRAINT else 'OFF'} ±({Y_MAX_LOW}→{Y_MAX_HIGH}) m linear   "
    f"N={N_SAMPLES}  max_tries={MAX_PLACEMENT_TRIES}",
    fontsize=10, fontweight="bold",
)

_xs_np = _xs.numpy()
_ys_np = _ys.numpy()
_bowl_valid_np = bowl_valid_grid().numpy().astype(float)   # precomputed once

for ax, (bowl_x, bowl_y) in zip(axes.flat, BOWL_POSITIONS):

    # ── Background constraint regions ────────────────────────────────────────
    valid_grid, in_box_grid, in_cone_grid = check_validity_grid(bowl_x, bowl_y)

    region = torch.zeros(RES, RES, dtype=torch.int32)
    region[_on_table]                    = 2   # out-of-range (default for on-table)
    region[_on_table & in_cone_grid]     = 3   # occluded by bowl
    region[_on_table & in_box_grid]      = 4   # bowl exclusion box (overrides cone)
    region[_on_table & valid_grid]       = 1   # valid

    ax.pcolormesh(_xs_np, _ys_np, region.numpy(),
                  cmap=CMAP, norm=NORM, shading="auto", rasterized=True, alpha=0.55)

    # ── Bowl valid zone overlay ───────────────────────────────────────────────
    ax.contourf(_xs_np, _ys_np, _bowl_valid_np,
                levels=[0.5, 1.5], colors=["#b2dfdb"], alpha=0.30, zorder=1)
    ax.contour(_xs_np, _ys_np, _bowl_valid_np,
               levels=[0.5], colors=["#00695c"], linewidths=[1.8], zorder=3)

    # ── Sampled cube positions ────────────────────────────────────────────────
    xy, failed, used_safety = sample_cube_positions(bowl_x, bowl_y)
    n_failed  = int(failed.sum().item())
    n_safety  = int(used_safety.sum().item())
    xy_np     = xy.numpy()
    random_ok  = (~failed & ~used_safety).numpy()
    safety_ok  = used_safety.numpy()

    if random_ok.any():
        ax.scatter(xy_np[random_ok, 0], xy_np[random_ok, 1],
                   c="#1b5e20", s=18, zorder=7, alpha=0.80, linewidths=0)
    if safety_ok.any():
        ax.scatter(xy_np[safety_ok, 0], xy_np[safety_ok, 1],
                   c="#f57f17", s=40, marker="D", zorder=8, alpha=0.90,
                   linewidths=0, label="Safety fallback")
    if failed.any():
        ax.scatter(xy_np[failed.numpy(), 0], xy_np[failed.numpy(), 1],
                   c="#b71c1c", s=55, marker="x", linewidths=1.8, zorder=9)

    # ── Safety position candidates ────────────────────────────────────────────
    for sx, sy in SAFETY_POSITIONS:
        ax.plot(sx, sy, marker="D", color="#f57f17", markersize=5,
                markeredgecolor="#bf360c", markeredgewidth=0.8,
                zorder=6, linestyle="None", alpha=0.70)

    # ── Bowl circle and exclusion box ─────────────────────────────────────────
    cone_r = _bowl_cone_radius(bowl_x)
    ax.add_patch(plt.Circle((bowl_x, bowl_y), cone_r,
                             color="#1565c0", alpha=0.85, zorder=5))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), cone_r,
                             fill=False, edgecolor="#e57373",
                             linestyle="--", linewidth=1.6, zorder=6))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_PHYSICAL_RADIUS,
                             fill=False, edgecolor="#ffffff",
                             linestyle="-", linewidth=1.8, zorder=7))
    ax.text(bowl_x, bowl_y - cone_r - 0.025, f"r={cone_r:.3f}",
            ha="center", va="top", fontsize=6.5, color="#aac8ff", zorder=8)

    # ── Annular rings ─────────────────────────────────────────────────────────
    for r in CUBE_DIST_RANGE:
        ax.add_patch(plt.Circle(PLACEMENT_POINT, r,
                                 fill=False, edgecolor="#1b5e20",
                                 linestyle="--", linewidth=1.2, alpha=0.55, zorder=4))
    # Outer bowl ring — circle
    ax.add_patch(plt.Circle(PLACEMENT_POINT, BOWL_DIST_RANGE[1],
                             fill=False, edgecolor="#1565c0",
                             linestyle=":", linewidth=1.0, alpha=0.40, zorder=4))
    # Inner bowl boundary — ELLIPSE (semi-x = _bowl_inner_semi_x, semi-y = BOWL_INNER_SEMI_Y)
    ax.add_patch(mpatches.Ellipse(
        PLACEMENT_POINT,
        width=2 * _bowl_inner_semi_x,
        height=2 * BOWL_INNER_SEMI_Y,
        fill=False,
        edgecolor="#1565c0",
        linestyle=":",
        linewidth=1.0,
        alpha=0.40,
        zorder=4,
    ))

    # ── x_min boundaries ──────────────────────────────────────────────────────
    ax.axvline(x=CUBE_X_MIN, color="#e65100", linestyle="-.", linewidth=1.2, alpha=0.70)
    ax.axvline(x=BOWL_X_MIN, color="#1565c0", linestyle="-.", linewidth=1.2, alpha=0.55)

    # ── y-band boundaries (linear: tight near robot, wide at far edge) ────────
    if CUBE_Y_CONSTRAINT:
        import numpy as np
        x_line = np.linspace(CUBE_X_MIN, _table_edge_x, 200)
        y_bound = Y_MAX_LOW + (Y_MAX_HIGH - Y_MAX_LOW) * (x_line - CUBE_X_MIN) / (_table_edge_x - CUBE_X_MIN)
        for sign in (+1, -1):
            ax.plot(x_line, sign * y_bound, color="#7b1fa2",
                    linestyle=":", linewidth=1.2, alpha=0.70)

    # ── Occlusion cone tangent lines from placement point ─────────────────────
    bvx = bowl_x - _px
    bvy = bowl_y - _py
    d_b = math.sqrt(bvx**2 + bvy**2)
    if d_b > cone_r:
        half_angle  = math.asin(cone_r / d_b)
        bowl_angle  = math.atan2(bvy, bvx)
        for sign in (+1, -1):
            tang = bowl_angle + sign * half_angle
            ax.plot(
                [_px, _px + 1.5 * math.cos(tang)],
                [_py, _py + 1.5 * math.sin(tang)],
                color="#e65100", linestyle="--", linewidth=1.0, alpha=0.55, zorder=4,
            )

    # ── Placement point marker ────────────────────────────────────────────────
    ax.plot(*PLACEMENT_POINT, marker="*", color="#e65100",
            markersize=10, zorder=9, linestyle="None")

    # ── Robot footprint ───────────────────────────────────────────────────────
    _half_r = ROBOT_SIZE / 2
    ax.add_patch(mpatches.Rectangle(
        (ROBOT_CENTER[0] - _half_r, ROBOT_CENTER[1] - _half_r),
        ROBOT_SIZE, ROBOT_SIZE,
        linewidth=1.8, edgecolor="#4a148c", facecolor="#ce93d8",
        alpha=0.85, zorder=5,
    ))

    # ── Table border ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (TABLE_X[0], TABLE_Y[0]),
        TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0],
        linewidth=2, edgecolor="black", facecolor="none", zorder=4,
    ))

    ax.set_xlim(TABLE_X)
    ax.set_ylim(TABLE_Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.20, linewidth=0.5)

    if n_failed > 0:
        status_str = f"{n_failed}/{N_SAMPLES} FAILED"
        title_color = "#b71c1c"
    elif n_safety > 0:
        status_str = f"{n_safety}/{N_SAMPLES} via safety pos"
        title_color = "#e65100"
    else:
        status_str = f"all {N_SAMPLES} valid"
        title_color = "#1b5e20"
    ax.set_title(
        f"bowl  x={bowl_x:.3f}  y={bowl_y:+.3f}   [{status_str}]",
        fontsize=9, color=title_color,
    )

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#81c784", alpha=0.6,
                   label="Valid cube region"),
    mpatches.Patch(color="#e57373", alpha=0.6,
                   label=f"Bowl excl. circle  (cone_r={CONE_RADIUS_MIN}→{BOWL_RADIUS} m)"),
    mpatches.Patch(color="#ffb74d", alpha=0.6,
                   label="Occlusion cone  (shadow behind bowl)"),
    mpatches.Patch(color="#b0bec5", alpha=0.6,
                   label="Outside annular ring / x_min / y_band"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#1b5e20", markersize=7,
               label=f"Valid sample  (n={N_SAMPLES})"),
    plt.Line2D([0], [0], marker="x", color="#b71c1c", lw=0,
               markersize=8, markeredgewidth=1.8,
               label="Failed sample (all fallbacks exhausted)"),
    plt.Line2D([0], [0], marker="D", color="#f57f17", lw=0,
               markersize=7, markeredgecolor="#bf360c", markeredgewidth=0.8,
               label=f"Safety fallback sample / candidate  (n={len(SAFETY_POSITIONS)})"),
    mpatches.Patch(color="#1565c0", alpha=0.85,
                   label=f"Bowl keep-out  (cone_r={CONE_RADIUS_MIN}→{BOWL_RADIUS} m, linear in bowl_x)"),
    plt.Line2D([0], [0], color="#e57373", linestyle="--", linewidth=1.6,
               label=f"Bowl excl. boundary  (cone_r, varies per subplot)"),
    plt.Line2D([0], [0], color="#ffffff", linestyle="-", linewidth=1.8,
               label=f"Bowl physical radius  (r={BOWL_PHYSICAL_RADIUS} m)"),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="--", linewidth=1.2,
               label=f"Cube annular ring  {CUBE_DIST_RANGE} m"),
    mpatches.Patch(facecolor="#b2dfdb", edgecolor="#00695c", linewidth=1.8, alpha=0.6,
                   label=f"Valid bowl zone  (outer r={BOWL_DIST_RANGE[1]} m, inner ellipse "
                         f"x>{BOWL_INNER_X_MIN} m @ y=0, semi_y={BOWL_INNER_SEMI_Y} m, "
                         f"|y|≤{BOWL_Y_MAX} m)"),
    plt.Line2D([0], [0], color="#e65100", linestyle="-.", linewidth=1.2,
               label=f"Cube x_min = {CUBE_X_MIN} m"),
    plt.Line2D([0], [0], color="#1565c0", linestyle="-.", linewidth=1.2,
               label=f"Bowl x_min = {BOWL_X_MIN} m"),
    plt.Line2D([0], [0], color="#7b1fa2", linestyle=":", linewidth=1.0,
               label=f"Cube y_band  ±({Y_MAX_LOW}→{Y_MAX_HIGH}) m  linear in x"
                     + ("" if CUBE_Y_CONSTRAINT else "  (OFF)")),
    plt.Line2D([0], [0], color="#e65100", linestyle="--", linewidth=1.0,
               label="Occlusion cone tangent lines"),
    plt.Line2D([0], [0], marker="*", color="#e65100", lw=0, markersize=10,
               label=f"Placement point  {PLACEMENT_POINT}"),
    mpatches.Patch(facecolor="#ce93d8", edgecolor="#4a148c", linewidth=1.8,
                   label=f"Robot  ({ROBOT_SIZE*100:.1f}×{ROBOT_SIZE*100:.1f} cm)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.003))

plt.tight_layout(rect=[0, 0.11, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "placement_constraints.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
