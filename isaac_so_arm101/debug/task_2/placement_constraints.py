#!/usr/bin/env python3
"""Visualise bowl and two-cube cluster placement constraints for Task 2.

The cluster centre is sampled with the same annular-ring / occlusion-cone approach
as the single-cube version (see debug/task_1/placement_constraints.py), but with
two key extra checks so that both adjacent cubes are guaranteed to clear the bowl:

  Bowl exclusion radius for the cluster centre:
      BOWL_RADIUS + CLUSTER_HALF_GAP + BOWL_EXTRA_MARGIN
  y-band check for the cluster centre:
      |cy| + CLUSTER_HALF_GAP ≤ CUBE_Y_MAX

After a valid centre is found, red and blue cubes are placed at:
    red  → (cx,  cy ± gap)   (random L/R flip per env)
    blue → (cx,  cy ∓ gap)

Background colour shows valid/invalid cluster-centre regions.
Cluster-centre samples are drawn as small grey circles; the derived red/blue cube
positions are shown as filled squares connected by a thin line.

To preview a parameter change:
  1. Edit the CONFIG block below.
  2. Re-run:  python3 debug/task_2/placement_constraints.py
  Output:    debug/task_2/placement_constraints.png
"""

import math
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

# ══ CONFIG ════════════════════════════════════════════════════════════════════

# Table extent (metres) — x = depth (robot sits at low-x end), y = lateral width
TABLE_X = (0.0, 0.8)
TABLE_Y = (-0.6, 0.6)

# Robot footprint (visualisation only)
ROBOT_SIZE   = 0.048          # side length (m)
ROBOT_CENTER = (0.024, 0.0)

# Placement point — first revolute joint; all radial distances measured from here
PLACEMENT_POINT = (0.048, 0.0)

# ── Bowl ──────────────────────────────────────────────────────────────────────
BOWL_PHYSICAL_RADIUS = 0.0775  # real physical bowl radius (m)
BOWL_RADIUS       = 0.14     # keep-out + cone half-width (m)
BOWL_DIST_RANGE   = (0.20, 0.40)
BOWL_X_MIN        = 0.148
BOWL_Y_CONSTRAINT = True
BOWL_Y_MAX        = 0.20

# ── Two-cube cluster ───────────────────────────────────────────────────────────
CUBE_DIST_RANGE   = (0.15, 0.30)
CUBE_X_MIN        = 0.148
CUBE_Y_CONSTRAINT = True
CUBE_Y_MAX        = 0.20
CLUSTER_HALF_GAP  = 0.011   # half-separation between the two cubes along y (m) — cube half-width + 1 mm
BOWL_EXTRA_MARGIN = 0.005   # extra safety buffer on top of bowl_radius + gap (m)

# Effective bowl exclusion radius for the cluster centre:
#   outer cube worst-case distance from bowl = excl_r - gap = bowl_r + extra_margin
CLUSTER_EXCL_RADIUS = BOWL_RADIUS + CLUSTER_HALF_GAP + BOWL_EXTRA_MARGIN

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
    """True where the point (cx, cy) is occluded behind the bowl from placement_point."""
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
    """Evaluate each constraint for the cluster centre position.

    The bowl exclusion and y-band are expanded by CLUSTER_HALF_GAP (+extra margin)
    so both derived cube positions are guaranteed to satisfy the original constraints.
    """
    d = torch.sqrt((cx - _px)**2 + (cy - _py)**2)
    return {
        "radius":  (d >= CUBE_DIST_RANGE[0]) & (d <= CUBE_DIST_RANGE[1]),
        "x_min":   cx >= CUBE_X_MIN,
        # Expanded exclusion: cluster_excl_radius ensures outer cube clears the bowl
        "box":     torch.sqrt((cx - bowl_x)**2 + (cy - bowl_y)**2) > CLUSTER_EXCL_RADIUS,
        "cone":    ~_in_occlusion_cone(cx, cy, bowl_x, bowl_y),
        # Expanded y-band: |cy| + gap ≤ y_max ensures both cubes stay within band
        "y_band":  (torch.abs(cy) + CLUSTER_HALF_GAP <= CUBE_Y_MAX) if CUBE_Y_CONSTRAINT
                   else torch.ones_like(cx, dtype=torch.bool),
    }


def check_validity_grid(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate cluster-centre constraints on the (RES, RES) grid."""
    c = _cluster_center_constraints(_XX_t, _YY_t, bowl_x, bowl_y)
    in_box  = ~c["box"]
    in_cone = c["box"] & ~c["cone"]
    valid   = c["radius"] & c["x_min"] & c["box"] & c["cone"] & c["y_band"]
    return valid, in_box, in_cone


def sample_cluster_positions(
    bowl_x: float, bowl_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rejection-sample N_SAMPLES cluster centres, then derive red/blue cube positions.

    Returns:
        centers        (N, 2)  cluster centre positions
        red_xy         (N, 2)  red cube positions
        blue_xy        (N, 2)  blue cube positions
        cluster_angles (N,)    shared z-rotation for each pair (radians)
        failed         (N,)    True where even safety positions did not help
        used_safety    (N,)    True where a safety position was used
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

    # One shared cluster rotation per sample; offset axis is rotated by that angle
    cluster_angles = torch.rand(n) * (2.0 * math.pi)
    gap = CLUSTER_HALF_GAP
    flip = torch.rand(n) > 0.5
    sign = torch.where(flip, torch.ones(n), -torch.ones(n))
    offset_x = -sign * gap * torch.sin(cluster_angles)
    offset_y =  sign * gap * torch.cos(cluster_angles)
    red_xy  = torch.stack([centers[:, 0] + offset_x, centers[:, 1] + offset_y], dim=1)
    blue_xy = torch.stack([centers[:, 0] - offset_x, centers[:, 1] - offset_y], dim=1)

    return centers, red_xy, blue_xy, cluster_angles, needs_resample, used_safety


def _oriented_rect_corners(cx: float, cy: float, angle: float, half: float = 0.01) -> np.ndarray:
    """Return (4, 2) array of corners for a square of side 2*half centred at (cx,cy), rotated by angle."""
    local = np.array([[-half, -half], [+half, -half], [+half, +half], [-half, +half]])
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


# ══ COLOUR SCHEME ════════════════════════════════════════════════════════════
# 0=off-table | 1=valid cluster centre | 2=out-of-range | 3=cone | 4=bowl-excl
CMAP = ListedColormap(["#d4d4d4", "#81c784", "#b0bec5", "#ffb74d", "#e57373"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], CMAP.N)

# ══ PLOT ══════════════════════════════════════════════════════════════════════
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    f"Two-Cube Cluster Placement Constraints (Task 2)  |  "
    f"bowl_dist={BOWL_DIST_RANGE} m   cube_dist={CUBE_DIST_RANGE} m\n"
    f"cluster_half_gap={CLUSTER_HALF_GAP*100:.1f} cm   bowl_extra_margin={BOWL_EXTRA_MARGIN*100:.1f} cm   "
    f"effective_excl_r={CLUSTER_EXCL_RADIUS*100:.1f} cm   physical_r={BOWL_PHYSICAL_RADIUS*100:.1f} cm   "
    f"N={N_SAMPLES}",
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

    # ── Sample cluster centres and derive cube positions ──────────────────────
    centers, red_xy, blue_xy, angles, failed, used_safety = sample_cluster_positions(bowl_x, bowl_y)
    n_failed  = int(failed.sum().item())
    n_safety  = int(used_safety.sum().item())

    centers_np = centers.numpy()
    red_np     = red_xy.numpy()
    blue_np    = blue_xy.numpy()
    angles_np  = angles.numpy()
    valid_mask  = (~failed & ~used_safety).numpy()
    safety_mask = used_safety.numpy()
    failed_mask = failed.numpy()

    # Draw oriented cube rectangles + connecting line for each sample
    for i in range(N_SAMPLES):
        if valid_mask[i]:
            edge_color = "#444444"
            r_face, b_face = "#c0392b", "#1565c0"
            alpha = 0.55
        elif safety_mask[i]:
            edge_color = "#bf360c"
            r_face, b_face = "#ffb74d", "#81d4fa"
            alpha = 0.70
        else:
            edge_color = "#b71c1c"
            r_face, b_face = "#ef9a9a", "#90caf9"
            alpha = 0.50

        angle = angles_np[i]
        # Connecting line through cluster centre
        ax.plot([red_np[i, 0], blue_np[i, 0]],
                [red_np[i, 1], blue_np[i, 1]],
                color=edge_color, linewidth=0.4, alpha=alpha * 0.6, zorder=5)
        # Oriented rectangles
        r_corners = _oriented_rect_corners(red_np[i, 0],  red_np[i, 1],  angle)
        b_corners = _oriented_rect_corners(blue_np[i, 0], blue_np[i, 1], angle)
        ax.add_patch(mpatches.Polygon(r_corners, closed=True,
                                      facecolor=r_face, edgecolor=edge_color,
                                      linewidth=0.4, alpha=alpha, zorder=6))
        ax.add_patch(mpatches.Polygon(b_corners, closed=True,
                                      facecolor=b_face, edgecolor=edge_color,
                                      linewidth=0.4, alpha=alpha, zorder=6))

    # Cluster centres (small grey dots on top)
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
    # Effective exclusion radius for cluster centre
    ax.add_patch(plt.Circle((bowl_x, bowl_y), CLUSTER_EXCL_RADIUS,
                             color="#1565c0", alpha=0.25, zorder=4))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), CLUSTER_EXCL_RADIUS,
                             fill=False, edgecolor="#e57373",
                             linestyle="--", linewidth=1.6, zorder=5))
    # Inner: bowl_radius + gap (outer cube worst-case reach)
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS + CLUSTER_HALF_GAP,
                             fill=False, edgecolor="#ff8f00",
                             linestyle=":", linewidth=1.2, zorder=5))
    # Bowl keep-out radius
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             fill=False, edgecolor="#1565c0",
                             linestyle="-", linewidth=1.4, zorder=5))
    # Physical bowl radius
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
    ax.axvline(x=CUBE_X_MIN, color="#e65100", linestyle="-.", linewidth=1.2, alpha=0.70)
    ax.axvline(x=BOWL_X_MIN, color="#1565c0", linestyle="-.", linewidth=1.2, alpha=0.55)

    # ── y-band boundaries (effective ± gap, as seen by the cluster centre) ────
    if CUBE_Y_CONSTRAINT:
        for sign in (+1, -1):
            # Original y_max line
            ax.axhline(y=sign * CUBE_Y_MAX, color="#7b1fa2",
                       linestyle=":", linewidth=1.0, alpha=0.60)
            # Effective y_max for cluster centre (shrunk by gap)
            ax.axhline(y=sign * (CUBE_Y_MAX - CLUSTER_HALF_GAP), color="#9c27b0",
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
                   label="Valid cluster-centre region"),
    mpatches.Patch(color="#e57373", alpha=0.6,
                   label=f"Cluster excl. zone  (r={CLUSTER_EXCL_RADIUS*100:.1f} cm = bowl_r + gap + extra)"),
    mpatches.Patch(color="#ffb74d", alpha=0.6,
                   label="Occlusion cone  (shadow behind bowl)"),
    mpatches.Patch(color="#b0bec5", alpha=0.6,
                   label="Outside annular ring / x_min / y_band"),
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#222222", markersize=5,
               label="Cluster centre (valid)"),
    mpatches.Patch(facecolor="#c0392b", edgecolor="#444444", linewidth=0.8,
                   label="Red cube (oriented rectangle)"),
    mpatches.Patch(facecolor="#1565c0", edgecolor="#444444", linewidth=0.8,
                   label="Blue cube (oriented rectangle)"),
    plt.Line2D([0], [0], color="#444444", linewidth=0.8,
               label="Red–centre–blue link"),
    plt.Line2D([0], [0], marker="x", color="#b71c1c", lw=0,
               markersize=8, markeredgewidth=1.8,
               label="Failed sample (all fallbacks exhausted)"),
    plt.Line2D([0], [0], marker="D", color="#f57f17", lw=0,
               markersize=7, markeredgecolor="#bf360c", markeredgewidth=0.8,
               label=f"Safety fallback candidate  (n={len(SAFETY_POSITIONS)})"),
    plt.Line2D([0], [0], color="#e57373", linestyle="--", linewidth=1.6,
               label=f"Cluster excl. boundary  r={CLUSTER_EXCL_RADIUS*100:.1f} cm"),
    plt.Line2D([0], [0], color="#ff8f00", linestyle=":", linewidth=1.2,
               label=f"Outer-cube worst-case  r={( BOWL_RADIUS+CLUSTER_HALF_GAP)*100:.1f} cm"),
    plt.Line2D([0], [0], color="#1565c0", linestyle="-", linewidth=1.4,
               label=f"Bowl keep-out  r={BOWL_RADIUS*100:.1f} cm"),
    plt.Line2D([0], [0], color="#ffffff", linestyle="-", linewidth=1.8,
               label=f"Bowl physical  r={BOWL_PHYSICAL_RADIUS*100:.1f} cm"),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="--", linewidth=1.2,
               label=f"Cluster-centre annular ring  {CUBE_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#1565c0", linestyle=":", linewidth=1.0,
               label=f"Bowl annular ring  {BOWL_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#e65100", linestyle="-.", linewidth=1.2,
               label=f"Cube x_min = {CUBE_X_MIN} m"),
    plt.Line2D([0], [0], color="#7b1fa2", linestyle=":", linewidth=1.0,
               label=f"Cube y_band  ±{CUBE_Y_MAX} m"),
    plt.Line2D([0], [0], color="#9c27b0", linestyle="--", linewidth=0.8,
               label=f"Effective y_band for centre  ±{CUBE_Y_MAX - CLUSTER_HALF_GAP:.3f} m"),
    plt.Line2D([0], [0], color="#e65100", linestyle="--", linewidth=1.0,
               label="Occlusion cone tangent lines"),
    plt.Line2D([0], [0], marker="*", color="#e65100", lw=0, markersize=10,
               label=f"Placement point  {PLACEMENT_POINT}"),
    mpatches.Patch(facecolor="#ce93d8", edgecolor="#4a148c", linewidth=1.8,
                   label=f"Robot  ({ROBOT_SIZE*100:.1f}×{ROBOT_SIZE*100:.1f} cm)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=8.0, frameon=True, bbox_to_anchor=(0.5, 0.002))

plt.tight_layout(rect=[0, 0.13, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "placement_constraints.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
