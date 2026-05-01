#!/usr/bin/env python3
"""Visualise cube placement constraints AND sampled positions for 6 bowl positions.

Background colour shows valid/invalid constraint regions (same as before).
Dots show actual rejection-sampled placements: green = valid, red × = failed
(budget exhausted).  Failure counts are shown in each subplot title.

Constraint logic and rejection sampling are an exact port of check_validity()
and the rejection loop inside reset_bowl_and_cube() in
tasks/pick_place/mdp/events.py — torch, same variable names, same formulas.

To preview the effect of a parameter change:
  1. Edit the CONFIG block below to match your EventTerm kwargs.
  2. Re-run:  python3 debug/cube_placement_constraints.py
  Output:    debug/cube_placement_constraints.png
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

# ══ CONFIG — edit this block to match your reset_bowl_and_cube EventTerm kwargs ══
# Every name maps 1-to-1 to a parameter of reset_bowl_and_cube() in events.py.

# Bowl default pose from RigidObjectCfg init_state (local / robot-relative frame).
BOWL_INIT_X = 0.3
BOWL_INIT_Y = 0.00

# bowl_pose_range — uniform offset applied to bowl init_state each reset.
# Bowl local x ∈ [BOWL_INIT_X + x[0], BOWL_INIT_X + x[1]], same for y.
BOWL_POSE_RANGE = {"x": (-0.05, 0.10), "y": (-0.20, 0.20)}

# cube_world_range — absolute XY sampling rectangle for the cube in local frame.
CUBE_WORLD_RANGE = {"x": (0.1, 0.3), "y": (-0.3, 0.3)}

# exclusion_radius — minimum distance (m) between cube and bowl centres.
EXCLUSION_RADIUS = 0.10

# exclusion_shape — shape of the keep-out zone around the bowl centre.
#   "circle" : radial check  dist(cube, bowl) > EXCLUSION_RADIUS  (default, matches events.py)
#   "box"    : axis-aligned square with side length = 2 * EXCLUSION_RADIUS
#              (same area as the bounding square of the circle; valid when |ox| > R OR |oy| > R)
EXCLUSION_SHAPE = "box"

# y_occlusion_threshold — half-width (m) of the y band in which C2 is enforced.
Y_OCCLUSION_THRESHOLD = 0.20

# max_placement_tries — rejection sampling budget per reset (matches events.py default).
MAX_PLACEMENT_TRIES = 100

# ── Debug-only sampling config (not EventTerm params) ─────────────────────────
N_SAMPLES   = 200   # cube positions to draw per bowl location
RANDOM_SEED = 42    # set to None for a new draw each run

# ── Physical / scene constants (change only if the asset changes) ─────────────
BOWL_RADIUS = 0.075

ROBOT_BOX_CENTER = (0.025, 0.0)   # robot base centre in local frame
ROBOT_BOX_SIZE   = 0.05           # side length (m) of the robot footprint indicator

TABLE_X = (0.0, 0.8)
TABLE_Y = (-0.6, 0.6)

# ══ BOWL TEST POSITIONS (derived from CONFIG) ══════════════════════════════════
_bx_lo  = BOWL_INIT_X + BOWL_POSE_RANGE["x"][0]   # 0.15
_bx_mid = BOWL_INIT_X                              # 0.25
_bx_hi  = BOWL_INIT_X + BOWL_POSE_RANGE["x"][1]   # 0.35
_by_lo  = BOWL_INIT_Y + BOWL_POSE_RANGE["y"][0]   # -0.20
_by_hi  = BOWL_INIT_Y + BOWL_POSE_RANGE["y"][1]   # +0.20

BOWL_POSITIONS = [
    (_bx_lo,  _by_lo),          # x=0.15, y=-0.20
    (_bx_mid, _by_lo),          # x=0.25, y=-0.20
    (_bx_hi,  _by_lo),          # x=0.35, y=-0.20
    (_bx_lo,   0.0),            # x=0.15, y= 0.00  (no C3 constraint)
    (_bx_mid,  _by_hi / 2),     # x=0.25, y=+0.10
    (_bx_hi,   _by_hi),         # x=0.35, y=+0.20
]

# ══ GRID (background constraint visualisation) ════════════════════════════════
# Plot covers the full table surface — edit TABLE_X / TABLE_Y above to resize.
PLOT_XLIM = TABLE_X
PLOT_YLIM = TABLE_Y

RES = 500

_xs  = torch.linspace(PLOT_XLIM[0], PLOT_XLIM[1], RES)
_ys  = torch.linspace(PLOT_YLIM[0], PLOT_YLIM[1], RES)
# indexing="ij" with (_ys, _xs): _YY_t[row,col]=_ys[row], _XX_t[row,col]=_xs[col]
# — matches numpy meshgrid(xs,ys) layout expected by pcolormesh.
_YY_t, _XX_t = torch.meshgrid(_ys, _xs, indexing="ij")

_on_table = (
    (_XX_t >= TABLE_X[0]) & (_XX_t <= TABLE_X[1]) &
    (_YY_t >= TABLE_Y[0]) & (_YY_t <= TABLE_Y[1])
)

_cube_lo = torch.tensor([CUBE_WORLD_RANGE["x"][0], CUBE_WORLD_RANGE["y"][0]])
_cube_hi = torch.tensor([CUBE_WORLD_RANGE["x"][1], CUBE_WORLD_RANGE["y"][1]])

# ══ CONSTRAINT LOGIC — exact port from events.py ═══════════════════════════════

def _c1_valid(ox: torch.Tensor, oy: torch.Tensor) -> torch.Tensor:
    """C1: True when (ox, oy) is outside the exclusion zone.

    Shared by both check_validity_grid and the closure in sample_cube_positions
    so that changing EXCLUSION_SHAPE affects all constraint checks and the
    background region map consistently.
    """
    if EXCLUSION_SHAPE == "circle":
        return (ox**2 + oy**2) > EXCLUSION_RADIUS**2
    else:  # "box" — axis-aligned square, side = 2 * EXCLUSION_RADIUS
        return (torch.abs(ox) > EXCLUSION_RADIUS) | (torch.abs(oy) > EXCLUSION_RADIUS)


def check_validity_grid(bowl_x: float, bowl_y: float) -> torch.Tensor:
    """Evaluate C1 & C2 & C3 for every grid cell. Returns (RES, RES) bool tensor.

    Uses broadcasting from scalar tensors — same formulas as events.py.
    """
    bx_local = torch.tensor(bowl_x)   # scalar, mirrors bx_local (n,) in events.py
    by_local = torch.tensor(bowl_y)

    ox = _XX_t - bx_local   # (RES, RES) — signed x displacement from bowl
    oy = _YY_t - by_local   # (RES, RES) — signed y displacement from bowl

    c1 = _c1_valid(ox, oy)
    c2 = (torch.abs(oy) > Y_OCCLUSION_THRESHOLD) | (ox <= 0.0)
    c3 = (~(by_local < 0) | (oy >= 0)) & (~(by_local > 0) | (oy <= 0))

    return c1 & c2 & c3


def sample_cube_positions(bowl_x: float, bowl_y: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rejection-sample N_SAMPLES cube positions for one bowl location.

    Mirrors reset_bowl_and_cube() in events.py exactly: same closure structure,
    same update rule, same .clone() guard.

    Returns:
        xy      (N_SAMPLES, 2)  final positions in local frame
        failed  (N_SAMPLES,)    True where budget was exhausted without a valid position
    """
    n = N_SAMPLES

    # Shape (n,) tensors — mirrors bx_local / by_local in events.py.
    bx_local = torch.full((n,), bowl_x)
    by_local = torch.full((n,), bowl_y)

    def check_validity(local_xy: torch.Tensor) -> torch.Tensor:
        ox = local_xy[:, 0] - bx_local
        oy = local_xy[:, 1] - by_local
        c1 = _c1_valid(ox, oy)
        c2 = (torch.abs(oy) > Y_OCCLUSION_THRESHOLD) | (ox <= 0.0)
        c3 = (~(by_local < 0) | (oy >= 0)) & (~(by_local > 0) | (oy <= 0))
        return c1 & c2 & c3

    # Initial uniform sample — mirrors math_utils.sample_uniform in events.py.
    xy = torch.rand(n, 2) * (_cube_hi - _cube_lo) + _cube_lo
    needs_resample = ~check_validity(xy)

    for _ in range(MAX_PLACEMENT_TRIES):
        if not needs_resample.any():
            break
        new_xy = torch.rand(n, 2) * (_cube_hi - _cube_lo) + _cube_lo
        xy[needs_resample] = new_xy[needs_resample]
        # .clone() prevents in-place index aliasing — mirrors events.py exactly.
        needs_resample[needs_resample.clone()] = ~check_validity(xy)[needs_resample]

    return xy, needs_resample


# ══ COLOUR SCHEME ═════════════════════════════════════════════════════════════
# 0=off-table | 1=valid | 2=invalid (occlusion) | 3=invalid (exclusion zone)
CMAP = ListedColormap(["#d4d4d4", "#81c784", "#ffb74d", "#e57373"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CMAP.N)

# ══ PLOT ══════════════════════════════════════════════════════════════════════
if RANDOM_SEED is not None:
    torch.manual_seed(RANDOM_SEED)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f"Cube Placement Constraints + {N_SAMPLES} Sampled Positions  |  "
    f"excl_r={EXCLUSION_RADIUS} m   y_occl_thresh={Y_OCCLUSION_THRESHOLD} m\n"
    f"cube_range  x={CUBE_WORLD_RANGE['x']}   y={CUBE_WORLD_RANGE['y']}   "
    f"max_tries={MAX_PLACEMENT_TRIES}\n"
    "(green = valid region / sample,  orange = occluded,  red = exclusion zone / failed sample)",
    fontsize=10, fontweight="bold",
)

_xs_np = _xs.numpy()
_ys_np = _ys.numpy()

for ax, (bowl_x, bowl_y) in zip(axes.flat, BOWL_POSITIONS):
    # ── Background constraint regions ────────────────────────────────────────
    valid_grid = check_validity_grid(bowl_x, bowl_y)
    excl_grid  = ~_c1_valid(_XX_t - bowl_x, _YY_t - bowl_y)

    region = torch.zeros(RES, RES, dtype=torch.int32)  # 0=off-table
    region[_on_table]                 = 2              # default: occluded
    region[_on_table & valid_grid]    = 1              # passes C1 & C2 & C3
    region[_on_table & excl_grid]     = 3              # exclusion zone (overrides)

    ax.pcolormesh(_xs_np, _ys_np, region.numpy(),
                  cmap=CMAP, norm=NORM, shading="auto", rasterized=True, alpha=0.55)

    # ── Sampled positions ─────────────────────────────────────────────────────
    xy, failed = sample_cube_positions(bowl_x, bowl_y)
    n_failed = int(failed.sum().item())

    xy_np     = xy.numpy()
    valid_pts = ~failed.numpy()

    if valid_pts.any():
        ax.scatter(xy_np[valid_pts, 0], xy_np[valid_pts, 1],
                   c="#1b5e20", s=20, zorder=7, alpha=0.80, linewidths=0)
    if failed.any():
        ax.scatter(xy_np[~valid_pts, 0], xy_np[~valid_pts, 1],
                   c="#b71c1c", s=55, marker="x", linewidths=1.8, zorder=8)

    # ── Decorations ──────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             color="#1565c0", alpha=0.9, zorder=5))
    # Exclusion boundary — shape matches EXCLUSION_SHAPE
    _er = EXCLUSION_RADIUS
    if EXCLUSION_SHAPE == "circle":
        ax.add_patch(plt.Circle((bowl_x, bowl_y), _er,
                                 fill=False, edgecolor="#b71c1c",
                                 linestyle="--", linewidth=1.8, zorder=6))
    else:  # "box"
        ax.add_patch(mpatches.Rectangle(
            (bowl_x - _er, bowl_y - _er), 2 * _er, 2 * _er,
            fill=False, edgecolor="#b71c1c", linestyle="--", linewidth=1.8, zorder=6,
        ))

    ax.axhline(y=bowl_y + Y_OCCLUSION_THRESHOLD, color="#1b5e20",
               linestyle="-.", linewidth=1.2, alpha=0.70)
    ax.axhline(y=bowl_y - Y_OCCLUSION_THRESHOLD, color="#1b5e20",
               linestyle="-.", linewidth=1.2, alpha=0.70)
    ax.axvline(x=bowl_x, color="#e65100", linestyle=":", linewidth=1.4, alpha=0.75)
    if bowl_y != 0:
        ax.axhline(y=bowl_y, color="#6a1b9a", linestyle=":", linewidth=1.4, alpha=0.75)

    rx, ry = CUBE_WORLD_RANGE["x"], CUBE_WORLD_RANGE["y"]
    ax.add_patch(mpatches.Rectangle(
        (rx[0], ry[0]), rx[1] - rx[0], ry[1] - ry[0],
        linewidth=1.5, edgecolor="#1565c0", facecolor="none", linestyle="--", zorder=4,
    ))
    ax.add_patch(mpatches.Rectangle(
        (TABLE_X[0], TABLE_Y[0]), TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0],
        linewidth=2, edgecolor="black", facecolor="none", zorder=4,
    ))

    # Robot footprint indicator (0.05 × 0.05 box centred at ROBOT_BOX_CENTER)
    _half = ROBOT_BOX_SIZE / 2
    ax.add_patch(mpatches.Rectangle(
        (ROBOT_BOX_CENTER[0] - _half, ROBOT_BOX_CENTER[1] - _half),
        ROBOT_BOX_SIZE, ROBOT_BOX_SIZE,
        linewidth=1.8, edgecolor="#4a148c", facecolor="#ce93d8", alpha=0.75, zorder=5,
    ))

    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.20, linewidth=0.5)

    fail_str  = f"{n_failed}/{N_SAMPLES} FAILED" if n_failed > 0 else f"all {N_SAMPLES} valid"
    ax.set_title(
        f"bowl  x={bowl_x:.2f}  y={bowl_y:+.2f}   [{fail_str}]",
        fontsize=10,
        color="#b71c1c" if n_failed > 0 else "#1b5e20",
    )

# Shared legend
legend_handles = [
    mpatches.Patch(color="#81c784", alpha=0.6,
                   label="Valid region"),
    mpatches.Patch(color="#e57373", alpha=0.6,
                   label=(f"Exclusion zone (dist < {EXCLUSION_RADIUS} m)"
                          if EXCLUSION_SHAPE == "circle" else
                          f"Exclusion zone (|ox|,|oy| < {EXCLUSION_RADIUS} m, "
                          f"side = {2*EXCLUSION_RADIUS} m)")),
    mpatches.Patch(color="#ffb74d", alpha=0.6,
                   label="Occluded region (C2 or C3 fail)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b5e20",
               markersize=7, label=f"Valid sample  (n={N_SAMPLES})"),
    plt.Line2D([0], [0], marker="x", color="#b71c1c", lw=0,
               markersize=8, markeredgewidth=1.8, label="Failed sample (budget exhausted)"),
    mpatches.Patch(color="#1565c0", alpha=0.9,
                   label=f"Bowl  (r = {BOWL_RADIUS:.3f} m)"),
    plt.Line2D([0], [0], color="#b71c1c", linestyle="--", linewidth=1.8,
               label=(f"Exclusion boundary  (r = {EXCLUSION_RADIUS} m)"
                      if EXCLUSION_SHAPE == "circle" else
                      f"Exclusion boundary  (side = {2*EXCLUSION_RADIUS} m)")),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="-.", linewidth=1.2,
               label=f"Y-occlusion threshold  (±{Y_OCCLUSION_THRESHOLD} m)"),
    plt.Line2D([0], [0], color="#e65100", linestyle=":", linewidth=1.4,
               label="X occlusion boundary  (bowl_x)"),
    plt.Line2D([0], [0], color="#6a1b9a", linestyle=":", linewidth=1.4,
               label="Y occlusion boundary  (bowl_y, C3)"),
    plt.Line2D([0], [0], color="#1565c0", linestyle="--", linewidth=1.5,
               label="cube_world_range rectangle"),
    mpatches.Patch(facecolor="#ce93d8", edgecolor="#4a148c", linewidth=1.8,
                   label=f"Robot base  ({ROBOT_BOX_SIZE*100:.0f}×{ROBOT_BOX_SIZE*100:.0f} cm "
                         f"@ x={ROBOT_BOX_CENTER[0]}, y={ROBOT_BOX_CENTER[1]})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.003))

plt.tight_layout(rect=[0, 0.09, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "cube_placement_constraints.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
