#!/usr/bin/env python3
"""Visualise bowl and four-cube cluster placement constraints for Task 3.

Four 2×2 cm cubes are arranged in a 2×2 grid (4×4 cm block) around the cluster
centre, all sharing the same random orientation angle.  The sampling logic is
the same annular-ring / occlusion-cone approach as Task 2, but the bowl exclusion
radius, the y-band, and the x-band are all expanded by the cluster half-diagonal
(gap × √2 ≈ 1.56 cm) so that every cube in the grid clears the bowl and stays
within the workspace bounds regardless of cluster orientation.

Cube layout relative to the cluster centre, rotated by angle θ:
  A (red)    = centre + gap·(+cos θ − sin θ,  +sin θ + cos θ)
  B (blue)   = centre + gap·(−cos θ − sin θ,  −sin θ + cos θ)
  C (green)  = centre + gap·(+cos θ + sin θ,  +sin θ − cos θ)
  D (orange) = centre + gap·(−cos θ + sin θ,  −sin θ − cos θ)

Key difference vs Task 2 cluster:
  - Task 2 worst-case extent from centre: CLUSTER_HALF_GAP (cubes offset in y only)
  - Task 3 worst-case extent from centre: CLUSTER_HALF_GAP × √2  (cubes offset along both axes)
  - x_min constraint is now active for the cluster centre (Task 2 had no x expansion).

To preview a parameter change:
  1. Edit the CONFIG block below.
  2. Re-run:  python3 debug/task_3/placement_constraints.py
  Output:    debug/task_3/placement_constraints.png
"""

import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

# ══ CONFIG ════════════════════════════════════════════════════════════════════

# Table extent (metres) — x = depth, y = lateral width
TABLE_X = (0.0, 0.8)
TABLE_Y = (-0.6, 0.6)

# Robot footprint (visualisation only)
ROBOT_SIZE   = 0.048
ROBOT_CENTER = (0.024, 0.0)

# Placement point — first revolute joint; all radial distances measured from here
PLACEMENT_POINT = (0.048, 0.0)

# ── Bowl ──────────────────────────────────────────────────────────────────────
BOWL_PHYSICAL_RADIUS = 0.0775
BOWL_RADIUS       = 0.14
BOWL_DIST_RANGE   = (0.20, 0.40)
BOWL_X_MIN        = 0.148
BOWL_Y_CONSTRAINT = True
BOWL_Y_MAX        = 0.20

# ── Four-cube cluster (2×2 grid) ──────────────────────────────────────────────
CUBE_DIST_RANGE   = (0.15, 0.30)
CUBE_X_MIN        = 0.148
CUBE_Y_CONSTRAINT = True
CUBE_Y_MAX        = 0.20
# Half-separation between adjacent cube centres along each axis (m).
# = cube half-width (0.010) + 1 mm physics clearance.
CLUSTER_HALF_GAP  = 0.011
BOWL_EXTRA_MARGIN = 0.005

# Worst-case distance from cluster centre to any cube centre (at 45° orientation).
CLUSTER_HALF_DIAG = CLUSTER_HALF_GAP * math.sqrt(2)   # ≈ 0.01556 m

# Effective bowl exclusion radius for the cluster centre.
# Ensures every cube in the 2×2 grid clears the bowl keep-out circle.
CLUSTER_EXCL_RADIUS = BOWL_RADIUS + CLUSTER_HALF_DIAG + BOWL_EXTRA_MARGIN

# ── Rejection sampling ────────────────────────────────────────────────────────
MAX_PLACEMENT_TRIES  = 200
SAFE_FALLBACK_AFTER  = 100
N_SAMPLES            = 200
RANDOM_SEED          = 42

# ── Safety positions (cluster centres) ────────────────────────────────────────
SAFETY_POSITIONS = [
    (0.268, +0.000),
    (0.253, +0.143),
    (0.253, -0.143),
    (0.293, +0.114),
    (0.293, -0.114),
    (0.338, +0.000),
    (0.189, +0.169),
    (0.189, -0.169),
]

# ══ BOWL TEST POSITIONS ═══════════════════════════════════════════════════════
BOWL_POSITIONS = [
    (0.30,  0.00),
    (0.25,  0.00),
    (0.40,  0.00),
    (0.30, -0.20),
    (0.30, +0.20),
    (0.35, -0.15),
]

_px, _py = float(PLACEMENT_POINT[0]), float(PLACEMENT_POINT[1])

# ══ GRID ══════════════════════════════════════════════════════════════════════
RES = 600

_xs = torch.linspace(TABLE_X[0], TABLE_X[1], RES)
_ys = torch.linspace(TABLE_Y[0], TABLE_Y[1], RES)
_YY_t, _XX_t = torch.meshgrid(_ys, _xs, indexing="ij")

_on_table = (
    (_XX_t >= TABLE_X[0]) & (_XX_t <= TABLE_X[1]) &
    (_YY_t >= TABLE_Y[0]) & (_YY_t <= TABLE_Y[1])
)

# ══ CONSTRAINT LOGIC ══════════════════════════════════════════════════════════

def _in_occlusion_cone(
    cx: torch.Tensor,
    cy: torch.Tensor,
    bowl_x: float,
    bowl_y: float,
) -> torch.Tensor:
    """True where (cx, cy) is occluded behind the bowl from the placement point."""
    vc_x = cx - _px
    vc_y = cy - _py
    vb_x = bowl_x - _px
    vb_y = bowl_y - _py
    d_c = torch.sqrt(vc_x**2 + vc_y**2).clamp(min=1e-9)
    vc_hat_x = vc_x / d_c
    vc_hat_y = vc_y / d_c
    proj = vb_x * vc_hat_x + vb_y * vc_hat_y
    perp = torch.abs(vb_x * vc_hat_y - vb_y * vc_hat_x)
    return (perp < BOWL_RADIUS) & (proj > 0.0) & (proj < d_c)


def _cluster_center_constraints(
    cx: torch.Tensor,
    cy: torch.Tensor,
    bowl_x: float,
    bowl_y: float,
) -> dict[str, torch.Tensor]:
    """Evaluate each constraint for the 4-cube cluster centre.

    All spatial bounds are expanded by CLUSTER_HALF_DIAG — the worst-case
    distance from the cluster centre to any individual cube centre (achieved
    at 45° cluster orientation).  This guarantees all four cubes satisfy the
    original workspace constraints regardless of orientation.

    New vs Task 2: x_min is also expanded (in Task 2 only y was expanded,
    because cubes were offset purely along the y axis).
    """
    d = torch.sqrt((cx - _px)**2 + (cy - _py)**2)
    return {
        "radius":  (d >= CUBE_DIST_RANGE[0]) & (d <= CUBE_DIST_RANGE[1]),
        # x_min: cluster centre must be far enough from the near edge so the
        # cube with the most negative x offset still clears CUBE_X_MIN.
        "x_min":   cx >= CUBE_X_MIN + CLUSTER_HALF_DIAG,
        # Bowl exclusion expanded by CLUSTER_HALF_DIAG + extra margin.
        "box":     torch.sqrt((cx - bowl_x)**2 + (cy - bowl_y)**2) > CLUSTER_EXCL_RADIUS,
        "cone":    ~_in_occlusion_cone(cx, cy, bowl_x, bowl_y),
        # y-band: |cy| + CLUSTER_HALF_DIAG ≤ CUBE_Y_MAX ensures all cubes stay in band.
        "y_band":  (torch.abs(cy) + CLUSTER_HALF_DIAG <= CUBE_Y_MAX) if CUBE_Y_CONSTRAINT
                   else torch.ones_like(cx, dtype=torch.bool),
    }


def check_validity_grid(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = _cluster_center_constraints(_XX_t, _YY_t, bowl_x, bowl_y)
    in_box  = ~c["box"]
    in_cone = c["box"] & ~c["cone"]
    valid   = c["radius"] & c["x_min"] & c["box"] & c["cone"] & c["y_band"]
    return valid, in_box, in_cone


def sample_cluster_positions(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, ...]:
    """Rejection-sample N_SAMPLES cluster centres, derive 4 cube positions each.

    The 4 cubes sit at the corners of a square of side 2×CLUSTER_HALF_GAP,
    rotated by a shared cluster angle θ:
      A (red)    = centre + gap·(ax + pe)
      B (blue)   = centre + gap·(−ax + pe)
      C (green)  = centre + gap·(ax − pe)
      D (orange) = centre + gap·(−ax − pe)
    where ax = (cos θ, sin θ) and pe = (−sin θ, cos θ).

    Returns:
        centers   (N, 2) cluster centre positions
        a_xy      (N, 2) red   cube positions
        b_xy      (N, 2) blue  cube positions
        c_xy      (N, 2) green cube positions
        d_xy      (N, 2) orange cube positions
        angles    (N,)   shared cluster rotation (radians)
        failed    (N,)   True where all fallbacks exhausted
        used_safety (N,) True where a safety position was used
    """
    n = N_SAMPLES
    r_lo, r_hi = CUBE_DIST_RANGE

    def _sample_annulus(count: int) -> torch.Tensor:
        angle = torch.rand(count) * (2.0 * math.pi)
        r = torch.sqrt(torch.rand(count) * (r_hi**2 - r_lo**2) + r_lo**2)
        return torch.stack([_px + r * torch.cos(angle),
                            _py + r * torch.sin(angle)], dim=1)

    def check_validity(xy: torch.Tensor) -> torch.Tensor:
        c = _cluster_center_constraints(xy[:, 0], xy[:, 1], bowl_x, bowl_y)
        return c["radius"] & c["x_min"] & c["box"] & c["cone"] & c["y_band"]

    centers = _sample_annulus(n)
    needs_resample = ~check_validity(centers)
    used_safety = torch.zeros(n, dtype=torch.bool)

    # Phase 1: random rejection sampling
    for _ in range(SAFE_FALLBACK_AFTER):
        if not needs_resample.any():
            break
        new_xy = _sample_annulus(n)
        centers[needs_resample] = new_xy[needs_resample]
        needs_resample[needs_resample.clone()] = ~check_validity(centers)[needs_resample]

    # Phase 2: safety position fallback
    if needs_resample.any():
        for sx, sy in SAFETY_POSITIONS:
            sp = torch.tensor([[sx, sy]])
            if check_validity(sp)[0]:
                count = int(needs_resample.sum().item())
                centers[needs_resample] = sp.expand(count, 2)
                used_safety[needs_resample] = True
                needs_resample[:] = False
                break

    # Build 4 cube positions from a shared cluster angle
    theta = torch.rand(n) * (2.0 * math.pi)
    gap   = CLUSTER_HALF_GAP
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # Main axis: (cos θ, sin θ).  Perpendicular: (−sin θ, cos θ).
    ax_x, ax_y =  cos_t,  sin_t
    pe_x, pe_y = -sin_t,  cos_t

    def _cube_pos(sx: float, sy: float) -> torch.Tensor:
        return torch.stack([
            centers[:, 0] + gap * (sx * ax_x + sy * pe_x),
            centers[:, 1] + gap * (sx * ax_y + sy * pe_y),
        ], dim=1)

    a_xy = _cube_pos(+1.0, +1.0)   # red
    b_xy = _cube_pos(-1.0, +1.0)   # blue
    c_xy = _cube_pos(+1.0, -1.0)   # green
    d_xy = _cube_pos(-1.0, -1.0)   # orange

    return centers, a_xy, b_xy, c_xy, d_xy, theta, needs_resample, used_safety


def _oriented_rect_corners(cx: float, cy: float, angle: float, half: float = 0.01) -> np.ndarray:
    """Return (4, 2) corners for a 2*half × 2*half square at (cx, cy) rotated by angle."""
    local = np.array([[-half, -half], [+half, -half], [+half, +half], [-half, +half]])
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


# ══ COLOUR SCHEME ════════════════════════════════════════════════════════════
# 0=off-table | 1=valid cluster centre | 2=out-of-range | 3=cone | 4=bowl-excl
CMAP = ListedColormap(["#d4d4d4", "#81c784", "#b0bec5", "#ffb74d", "#e57373"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CMAP.N)

# Full-saturation cube face colours for valid samples
CUBE_FACE_COLORS  = ["#c0392b", "#1565c0", "#27ae60", "#f39c12"]  # R, B, G, O
# Washed-out versions for safety-fallback samples
CUBE_FACE_SAFETY  = ["#ffb74d", "#81d4fa", "#a5d6a7", "#ffe082"]
# Even more washed-out for failed samples
CUBE_FACE_FAILED  = ["#ef9a9a", "#90caf9", "#c8e6c9", "#fff9c4"]

_EFFECTIVE_X_MIN = CUBE_X_MIN + CLUSTER_HALF_DIAG

# ══ PLOT ══════════════════════════════════════════════════════════════════════
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Four-Cube Cluster Placement Constraints (Task 3)  |  "
    f"bowl_dist={BOWL_DIST_RANGE} m   cube_dist={CUBE_DIST_RANGE} m\n"
    f"cluster_half_gap={CLUSTER_HALF_GAP*100:.1f} cm   "
    f"cluster_half_diag={CLUSTER_HALF_DIAG*100:.2f} cm   "
    f"bowl_extra_margin={BOWL_EXTRA_MARGIN*100:.1f} cm   "
    f"effective_excl_r={CLUSTER_EXCL_RADIUS*100:.2f} cm   "
    f"physical_r={BOWL_PHYSICAL_RADIUS*100:.1f} cm   N={N_SAMPLES}",
    fontsize=10, fontweight="bold",
)

_xs_np = _xs.numpy()
_ys_np = _ys.numpy()

for ax, (bowl_x, bowl_y) in zip(axes.flat, BOWL_POSITIONS):

    # ── Background constraint regions (for cluster centre) ────────────────────
    valid_grid, in_box_grid, in_cone_grid = check_validity_grid(bowl_x, bowl_y)

    region = torch.zeros(RES, RES, dtype=torch.int32)
    region[_on_table]                    = 2
    region[_on_table & in_cone_grid]     = 3
    region[_on_table & in_box_grid]      = 4
    region[_on_table & valid_grid]       = 1

    ax.pcolormesh(_xs_np, _ys_np, region.numpy(),
                  cmap=CMAP, norm=NORM, shading="auto", rasterized=True, alpha=0.55)

    # ── Sample cluster centres and derive 4 cube positions ───────────────────
    centers, a_xy, b_xy, c_xy, d_xy, angles, failed, used_safety = \
        sample_cluster_positions(bowl_x, bowl_y)
    n_failed = int(failed.sum().item())
    n_safety = int(used_safety.sum().item())

    centers_np = centers.numpy()
    cube_nps   = [a_xy.numpy(), b_xy.numpy(), c_xy.numpy(), d_xy.numpy()]
    angles_np  = angles.numpy()
    valid_mask  = (~failed & ~used_safety).numpy()
    safety_mask = used_safety.numpy()
    failed_mask = failed.numpy()

    for i in range(N_SAMPLES):
        if valid_mask[i]:
            edge_color  = "#444444"
            alpha       = 0.55
            face_colors = CUBE_FACE_COLORS
        elif safety_mask[i]:
            edge_color  = "#bf360c"
            alpha       = 0.70
            face_colors = CUBE_FACE_SAFETY
        else:
            edge_color  = "#b71c1c"
            alpha       = 0.50
            face_colors = CUBE_FACE_FAILED

        angle = angles_np[i]

        # Thin lines from cluster centre to each cube centre
        for cube_np in cube_nps:
            ax.plot([centers_np[i, 0], cube_np[i, 0]],
                    [centers_np[i, 1], cube_np[i, 1]],
                    color=edge_color, linewidth=0.3, alpha=alpha * 0.5, zorder=5)

        # Oriented rectangle for each cube
        for cube_np, face_color in zip(cube_nps, face_colors):
            corners = _oriented_rect_corners(cube_np[i, 0], cube_np[i, 1], angle)
            ax.add_patch(mpatches.Polygon(corners, closed=True,
                                          facecolor=face_color, edgecolor=edge_color,
                                          linewidth=0.4, alpha=alpha, zorder=6))

    # Cluster centres (small dots on top)
    if valid_mask.any():
        ax.scatter(centers_np[valid_mask, 0], centers_np[valid_mask, 1],
                   c="#222222", s=4, zorder=8, alpha=0.50, linewidths=0)
    if safety_mask.any():
        ax.scatter(centers_np[safety_mask, 0], centers_np[safety_mask, 1],
                   c="#f57f17", s=12, marker="D", zorder=9, alpha=0.85, linewidths=0)
    if failed_mask.any():
        ax.scatter(centers_np[failed_mask, 0], centers_np[failed_mask, 1],
                   c="#b71c1c", s=28, marker="x", linewidths=1.5, zorder=9)

    # ── Safety position candidates ────────────────────────────────────────────
    for sx, sy in SAFETY_POSITIONS:
        ax.plot(sx, sy, marker="D", color="#f57f17", markersize=5,
                markeredgecolor="#bf360c", markeredgewidth=0.8,
                zorder=6, linestyle="None", alpha=0.70)

    # ── Bowl circles ──────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((bowl_x, bowl_y), CLUSTER_EXCL_RADIUS,
                             color="#1565c0", alpha=0.25, zorder=4))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), CLUSTER_EXCL_RADIUS,
                             fill=False, edgecolor="#e57373",
                             linestyle="--", linewidth=1.6, zorder=5))
    # Outer-cube worst-case distance (bowl_r + diag, without extra margin)
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS + CLUSTER_HALF_DIAG,
                             fill=False, edgecolor="#ff8f00",
                             linestyle=":", linewidth=1.2, zorder=5))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             fill=False, edgecolor="#1565c0",
                             linestyle="-", linewidth=1.4, zorder=5))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_PHYSICAL_RADIUS,
                             fill=False, edgecolor="#ffffff",
                             linestyle="-", linewidth=1.8, zorder=6))

    # ── Annular rings ─────────────────────────────────────────────────────────
    for r in CUBE_DIST_RANGE:
        ax.add_patch(plt.Circle(PLACEMENT_POINT, r,
                                fill=False, edgecolor="#1b5e20",
                                linestyle="--", linewidth=1.2, alpha=0.55, zorder=4))
    for r in BOWL_DIST_RANGE:
        ax.add_patch(plt.Circle(PLACEMENT_POINT, r,
                                fill=False, edgecolor="#1565c0",
                                linestyle=":", linewidth=1.0, alpha=0.40, zorder=4))

    # ── x_min boundaries ──────────────────────────────────────────────────────
    # Original x_min (thin, faded)
    ax.axvline(x=CUBE_X_MIN, color="#e65100", linestyle="-.",
               linewidth=0.9, alpha=0.45)
    # Effective x_min for cluster centre (bold)
    ax.axvline(x=_EFFECTIVE_X_MIN, color="#e65100", linestyle="-.",
               linewidth=1.5, alpha=0.85)
    ax.axvline(x=BOWL_X_MIN, color="#1565c0", linestyle="-.",
               linewidth=1.2, alpha=0.55)

    # ── y-band boundaries ────────────────────────────────────────────────────
    if CUBE_Y_CONSTRAINT:
        for sign in (+1, -1):
            # Original y_max
            ax.axhline(y=sign * CUBE_Y_MAX, color="#7b1fa2",
                       linestyle=":", linewidth=1.0, alpha=0.60)
            # Effective y_max for cluster centre (shrunk by diag)
            ax.axhline(y=sign * (CUBE_Y_MAX - CLUSTER_HALF_DIAG), color="#9c27b0",
                       linestyle="--", linewidth=0.8, alpha=0.50)

    # ── Occlusion cone tangent lines ──────────────────────────────────────────
    bvx = bowl_x - _px
    bvy = bowl_y - _py
    d_b = math.sqrt(bvx**2 + bvy**2)
    if d_b > BOWL_RADIUS:
        half_angle = math.asin(BOWL_RADIUS / d_b)
        bowl_angle = math.atan2(bvy, bvx)
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
        status_str  = f"{n_failed}/{N_SAMPLES} FAILED"
        title_color = "#b71c1c"
    elif n_safety > 0:
        status_str  = f"{n_safety}/{N_SAMPLES} via safety pos"
        title_color = "#e65100"
    else:
        status_str  = f"all {N_SAMPLES} valid"
        title_color = "#1b5e20"
    ax.set_title(
        f"bowl  x={bowl_x:.3f}  y={bowl_y:+.3f}   [{status_str}]",
        fontsize=9, color=title_color,
    )

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color="#81c784", alpha=0.6,
                   label="Valid cluster-centre region"),
    mpatches.Patch(color="#e57373", alpha=0.6,
                   label=f"Cluster excl. zone  (r={CLUSTER_EXCL_RADIUS*100:.2f} cm = bowl_r + diag + extra)"),
    mpatches.Patch(color="#ffb74d", alpha=0.6,
                   label="Occlusion cone  (shadow behind bowl)"),
    mpatches.Patch(color="#b0bec5", alpha=0.6,
                   label="Outside annular ring / x_min / y_band"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#222222", markersize=5,
               label="Cluster centre (valid)"),
    mpatches.Patch(facecolor=CUBE_FACE_COLORS[0], edgecolor="#444444", linewidth=0.8,
                   label="Cube A — red"),
    mpatches.Patch(facecolor=CUBE_FACE_COLORS[1], edgecolor="#444444", linewidth=0.8,
                   label="Cube B — blue"),
    mpatches.Patch(facecolor=CUBE_FACE_COLORS[2], edgecolor="#444444", linewidth=0.8,
                   label="Cube C — green"),
    mpatches.Patch(facecolor=CUBE_FACE_COLORS[3], edgecolor="#444444", linewidth=0.8,
                   label="Cube D — orange"),
    plt.Line2D([0], [0], color="#444444", linewidth=0.5,
               label="Centre–cube links"),
    plt.Line2D([0], [0], marker="x", color="#b71c1c", lw=0,
               markersize=8, markeredgewidth=1.8,
               label="Failed sample (all fallbacks exhausted)"),
    plt.Line2D([0], [0], marker="D", color="#f57f17", lw=0,
               markersize=7, markeredgecolor="#bf360c", markeredgewidth=0.8,
               label=f"Safety fallback candidate  (n={len(SAFETY_POSITIONS)})"),
    plt.Line2D([0], [0], color="#e57373", linestyle="--", linewidth=1.6,
               label=f"Cluster excl. boundary  r={CLUSTER_EXCL_RADIUS*100:.2f} cm"),
    plt.Line2D([0], [0], color="#ff8f00", linestyle=":", linewidth=1.2,
               label=f"Outer-cube worst-case  r={(BOWL_RADIUS + CLUSTER_HALF_DIAG)*100:.2f} cm"),
    plt.Line2D([0], [0], color="#1565c0", linestyle="-", linewidth=1.4,
               label=f"Bowl keep-out  r={BOWL_RADIUS*100:.1f} cm"),
    plt.Line2D([0], [0], color="#ffffff", linestyle="-", linewidth=1.8,
               label=f"Bowl physical  r={BOWL_PHYSICAL_RADIUS*100:.1f} cm"),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="--", linewidth=1.2,
               label=f"Cluster-centre annular ring  {CUBE_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#1565c0", linestyle=":", linewidth=1.0,
               label=f"Bowl annular ring  {BOWL_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#e65100", linestyle="-.", linewidth=0.9,
               label=f"Cube x_min = {CUBE_X_MIN} m  (original)"),
    plt.Line2D([0], [0], color="#e65100", linestyle="-.", linewidth=1.5,
               label=f"Effective x_min for centre = {_EFFECTIVE_X_MIN:.3f} m  (+diag)"),
    plt.Line2D([0], [0], color="#7b1fa2", linestyle=":", linewidth=1.0,
               label=f"Cube y_band  ±{CUBE_Y_MAX} m"),
    plt.Line2D([0], [0], color="#9c27b0", linestyle="--", linewidth=0.8,
               label=f"Effective y_band for centre  ±{CUBE_Y_MAX - CLUSTER_HALF_DIAG:.3f} m  (−diag)"),
    plt.Line2D([0], [0], color="#e65100", linestyle="--", linewidth=1.0,
               label="Occlusion cone tangent lines"),
    plt.Line2D([0], [0], marker="*", color="#e65100", lw=0, markersize=10,
               label=f"Placement point  {PLACEMENT_POINT}"),
    mpatches.Patch(facecolor="#ce93d8", edgecolor="#4a148c", linewidth=1.8,
                   label=f"Robot  ({ROBOT_SIZE*100:.1f}×{ROBOT_SIZE*100:.1f} cm)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=8.0, frameon=True, bbox_to_anchor=(0.5, 0.002))

plt.tight_layout(rect=[0, 0.14, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "placement_constraints.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
