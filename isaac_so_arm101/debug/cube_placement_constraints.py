#!/usr/bin/env python3
"""Visualise valid/invalid cube placement regions for 6 different bowl positions.

Constraints enforced:
  1. Exclusion zone   – cube centre must be > EXCLUSION_RADIUS (0.10 m) from bowl centre
                        (bowl radius ~0.075 m  +  2.5 cm safety margin)
  2. X occlusion      – cube_x <= bowl_x  ONLY when |cube_y - bowl_y| <= Y_OCCLUSION_THRESHOLD
                        If the cube is further than 0.2 m in y from the bowl it is off to the
                        side and no longer occluded, so any x is allowed.
  3. Y occlusion      – if bowl_y < 0 → cube_y >= bowl_y
                        if bowl_y > 0 → cube_y <= bowl_y
                        if bowl_y == 0 → no y constraint (bowl on robot centre-line)

Run from any directory:
    python3 debug/cube_placement_constraints.py
Output: debug/cube_placement_constraints.png
"""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# ── Physical constants ─────────────────────────────────────────────────────────
BOWL_RADIUS = 0.075          # ~15 cm diameter at scale 1.35 → radius 7.5 cm
EXCLUSION_RADIUS = 0.1      # exclusion zone radius → 2.5 cm margin beyond bowl edge
Y_OCCLUSION_THRESHOLD = 0.6 # if |cube_y - bowl_y| > this, x occlusion is lifted -> 0.2 could make sense

TABLE_X = (0.0, 0.8)        # table surface bounds in world frame (robot at origin)
TABLE_Y = (-0.6, 0.6)

# ── 6 bowl positions spanning the configured randomisation range ───────────────
# bowl x ∈ [0.15, 0.25],  bowl y ∈ [-0.2, 0.2]
BOWL_POSITIONS = [
    (0.15, -0.20),
    (0.20, -0.20),
    (0.25, -0.20),
    (0.15,  0.00),
    (0.20,  0.10),
    (0.25,  0.20),
]

# ── Dynamic plot limits: max/min bowl coords + 0.1 m margin ───────────────────
PLOT_MARGIN = 0.10
all_bx = [p[0] for p in BOWL_POSITIONS]
all_by = [p[1] for p in BOWL_POSITIONS]
PLOT_XLIM = (max(TABLE_X[0], min(all_bx) - PLOT_MARGIN),
             min(TABLE_X[1], max(all_bx) + PLOT_MARGIN))
PLOT_YLIM = (max(TABLE_Y[0], min(all_by) - PLOT_MARGIN),
             min(TABLE_Y[1], max(all_by) + PLOT_MARGIN))

# ── Grid (covers the full plot area) ─────────────────────────────────────────
RES = 600
xs = np.linspace(PLOT_XLIM[0], PLOT_XLIM[1], RES)
ys = np.linspace(PLOT_YLIM[0], PLOT_YLIM[1], RES)
XX, YY = np.meshgrid(xs, ys)

on_table = (
    (XX >= TABLE_X[0]) & (XX <= TABLE_X[1]) &
    (YY >= TABLE_Y[0]) & (YY <= TABLE_Y[1])
)

# ── Colour scheme ──────────────────────────────────────────────────────────────
# 0 = off-table  1 = valid  2 = invalid occlusion  3 = invalid exclusion zone
CMAP = ListedColormap(["#d4d4d4", "#81c784", "#ffb74d", "#e57373"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], CMAP.N)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "Cube Placement Constraints for 6 Bowl Positions\n"
    "(green = valid,  orange = occluded,  red = exclusion zone)",
    fontsize=13, fontweight="bold",
)

for ax, (bowl_x, bowl_y) in zip(axes.flat, BOWL_POSITIONS):
    dist = np.sqrt((XX - bowl_x) ** 2 + (YY - bowl_y) ** 2)

    # Constraint 1 – exclusion zone
    c_proximity = dist > EXCLUSION_RADIUS

    # Constraint 2 – x occlusion (only active when cube is close in y to bowl)
    y_close_to_bowl = np.abs(YY - bowl_y) <= Y_OCCLUSION_THRESHOLD
    c_x = (~y_close_to_bowl) | (XX <= bowl_x)
    # reads as: x is ok if cube is far in y  OR  cube_x is not behind bowl

    # Constraint 3 – y occlusion
    if bowl_y < 0:
        c_y = YY >= bowl_y
    elif bowl_y > 0:
        c_y = YY <= bowl_y
    else:
        c_y = np.ones_like(XX, dtype=bool)

    # Classify every grid cell
    region = np.zeros_like(XX, dtype=int)          # 0 = off-table
    region[on_table] = 2                            # default: occlusion-invalid
    region[on_table & c_x & c_y] = 3               # passed occlusion, check proximity
    region[on_table & ~c_proximity] = 3             # exclusion zone overrides
    region[on_table & c_proximity & c_x & c_y] = 1 # fully valid

    ax.pcolormesh(xs, ys, region, cmap=CMAP, norm=NORM, shading="auto", rasterized=True)

    # Bowl disc and exclusion-zone ring
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             color="#1565c0", alpha=0.9, zorder=5))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), EXCLUSION_RADIUS,
                             fill=False, edgecolor="#b71c1c",
                             linestyle="--", linewidth=1.8, zorder=6))

    # Y-occlusion threshold lines (where x constraint turns on/off)
    ax.axhline(y=bowl_y + Y_OCCLUSION_THRESHOLD, color="#1b5e20",
               linestyle="-.", linewidth=1.4, alpha=0.75)
    ax.axhline(y=bowl_y - Y_OCCLUSION_THRESHOLD, color="#1b5e20",
               linestyle="-.", linewidth=1.4, alpha=0.75)

    # X occlusion boundary (only meaningful inside the y threshold band)
    ax.axvline(x=bowl_x, color="#e65100", linestyle=":", linewidth=1.6, alpha=0.8)

    # Y occlusion boundary
    if bowl_y != 0:
        ax.axhline(y=bowl_y, color="#6a1b9a", linestyle=":", linewidth=1.6, alpha=0.8)

    # Table border
    ax.add_patch(mpatches.Rectangle(
        (TABLE_X[0], TABLE_Y[0]),
        TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0],
        linewidth=2, edgecolor="black", facecolor="none", zorder=4,
    ))

    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.set_aspect("equal")
    ax.set_title(f"bowl  x={bowl_x:.2f}  y={bowl_y:+.2f}", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.20, linewidth=0.5)

# Shared legend
legend_handles = [
    mpatches.Patch(color="#81c784", label="Valid placement"),
    mpatches.Patch(color="#e57373", label="Invalid – exclusion zone (dist < 0.10 m)"),
    mpatches.Patch(color="#ffb74d", label="Invalid – occlusion (x or y constraint)"),
    mpatches.Patch(color="#1565c0", alpha=0.9, label=f"Bowl (r = {BOWL_RADIUS:.3f} m)"),
    plt.Line2D([0], [0], color="#b71c1c", linestyle="--", linewidth=1.8,
               label=f"Exclusion boundary (r = {EXCLUSION_RADIUS:.2f} m)"),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="-.", linewidth=1.4,
               label=f"Y-occlusion threshold (±{Y_OCCLUSION_THRESHOLD:.2f} m)"),
    plt.Line2D([0], [0], color="#e65100", linestyle=":", linewidth=1.6,
               label="X occlusion boundary (bowl_x)"),
    plt.Line2D([0], [0], color="#6a1b9a", linestyle=":", linewidth=1.6,
               label="Y occlusion boundary (bowl_y)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.005))

plt.tight_layout(rect=[0, 0.09, 1, 0.97])

out_path = os.path.join(os.path.dirname(__file__), "cube_placement_constraints.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
