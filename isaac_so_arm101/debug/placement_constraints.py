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
BOWL_RADIUS       = 0.14     # keep-out + cone half-width (m) — wider than physical to account for 3D camera perspective
BOWL_DIST_RANGE   = (0.20, 0.40)   # annular ring radii from placement point (m)
BOWL_X_MIN        = 0.148    # absolute world-x lower bound (= placement_pt_x + 0.10)
BOWL_Y_CONSTRAINT = True           # optional: |bowl_y| ≤ BOWL_Y_MAX
BOWL_Y_MAX        = 0.20           # (m)

# ── Cube ──────────────────────────────────────────────────────────────────────
CUBE_HALF_SIZE    = 0.01     # half side-length (m)  [2 cm cube]
CUBE_DIST_RANGE   = (0.15, 0.30)   # annular ring radii from placement point (m)
CUBE_X_MIN        = 0.148    # absolute world-x lower bound (= placement_pt_x + 0.10)
CUBE_Y_CONSTRAINT = True           # optional: |cube_y| ≤ CUBE_Y_MAX
CUBE_Y_MAX        = 0.20           # (m)

# ── Rejection sampling ────────────────────────────────────────────────────────
MAX_PLACEMENT_TRIES  = 200
SAFE_FALLBACK_AFTER  = 100   # after this many random tries, attempt safety positions
N_SAMPLES            = 200   # cube positions sampled per bowl test location
RANDOM_SEED          = 42

# ── Safety positions ──────────────────────────────────────────────────────────
# Tried in order once SAFE_FALLBACK_AFTER random attempts are exhausted.
# Must each satisfy: dist ∈ CUBE_DIST_RANGE, x ≥ CUBE_X_MIN, |y| ≤ CUBE_Y_MAX.
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
    (0.30,  0.00),   # centre of range (default position)
    (0.25,  0.00),   # near centre     (x minimum)
    (0.40,  0.00),   # far centre      (x maximum)
    (0.30, -0.20),   # mid x, far -y boundary
    (0.30, +0.20),   # mid x, far +y boundary
    (0.35, -0.15),   # off-centre: far x, moderate -y
]
_px, _py = float(PLACEMENT_POINT[0]), float(PLACEMENT_POINT[1])

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

def _in_occlusion_cone(
    cx: torch.Tensor,
    cy: torch.Tensor,
    bowl_x: float,
    bowl_y: float,
) -> torch.Tensor:
    """True where the cube position is occluded behind the bowl from the placement
    point.

    Exact 2-D line-of-sight check: the cube at C is occluded when the ray from
    placement point P through C passes through the bowl disk of radius BOWL_RADIUS.
    Three conditions must all hold:
      1. The perpendicular distance from bowl centre B to the ray P→C < BOWL_RADIUS.
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

    return (perp < BOWL_RADIUS) & (proj > 0.0) & (proj < d_c)


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
        y_band  — |cube_y| ≤ CUBE_Y_MAX  (always True when CUBE_Y_CONSTRAINT=False)
    """
    d = torch.sqrt((cx - _px)**2 + (cy - _py)**2)
    return {
        "radius": (d >= CUBE_DIST_RANGE[0]) & (d <= CUBE_DIST_RANGE[1]),
        "x_min":  cx >= CUBE_X_MIN,
        "box":    torch.sqrt((cx - bowl_x)**2 + (cy - bowl_y)**2) > BOWL_RADIUS,
        "cone":   ~_in_occlusion_cone(cx, cy, bowl_x, bowl_y),
        "y_band": (torch.abs(cy) <= CUBE_Y_MAX) if CUBE_Y_CONSTRAINT
                  else torch.ones_like(cx, dtype=torch.bool),
    }


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
    f"physical_r={BOWL_PHYSICAL_RADIUS} m   excl_r={BOWL_RADIUS} m\n"
    f"placement_pt={PLACEMENT_POINT}   cube_x_min={CUBE_X_MIN} m   "
    f"y_constraint={'ON' if CUBE_Y_CONSTRAINT else 'OFF'} ±{CUBE_Y_MAX} m   "
    f"N={N_SAMPLES}  max_tries={MAX_PLACEMENT_TRIES}",
    fontsize=10, fontweight="bold",
)

_xs_np = _xs.numpy()
_ys_np = _ys.numpy()

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
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             color="#1565c0", alpha=0.85, zorder=5))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_RADIUS,
                             fill=False, edgecolor="#e57373",
                             linestyle="--", linewidth=1.6, zorder=6))
    ax.add_patch(plt.Circle((bowl_x, bowl_y), BOWL_PHYSICAL_RADIUS,
                             fill=False, edgecolor="#ffffff",
                             linestyle="-", linewidth=1.8, zorder=7))

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

    # ── y-band boundaries ─────────────────────────────────────────────────────
    if CUBE_Y_CONSTRAINT:
        for sign in (+1, -1):
            ax.axhline(y=sign * CUBE_Y_MAX, color="#7b1fa2",
                       linestyle=":", linewidth=1.0, alpha=0.60)

    # ── Occlusion cone tangent lines from placement point ─────────────────────
    bvx = bowl_x - _px
    bvy = bowl_y - _py
    d_b = math.sqrt(bvx**2 + bvy**2)
    if d_b > BOWL_RADIUS:
        half_angle  = math.asin(BOWL_RADIUS / d_b)
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
                   label=f"Bowl excl. circle  (r={BOWL_RADIUS} m)"),
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
                   label=f"Bowl  (r={BOWL_RADIUS:.4f} m, keep-out)"),
    plt.Line2D([0], [0], color="#e57373", linestyle="--", linewidth=1.6,
               label=f"Bowl excl. boundary  (r={BOWL_RADIUS} m)"),
    plt.Line2D([0], [0], color="#ffffff", linestyle="-", linewidth=1.8,
               label=f"Bowl physical radius  (r={BOWL_PHYSICAL_RADIUS} m)"),
    plt.Line2D([0], [0], color="#1b5e20", linestyle="--", linewidth=1.2,
               label=f"Cube annular ring  {CUBE_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#1565c0", linestyle=":", linewidth=1.0,
               label=f"Bowl annular ring  {BOWL_DIST_RANGE} m"),
    plt.Line2D([0], [0], color="#e65100", linestyle="-.", linewidth=1.2,
               label=f"Cube x_min = {CUBE_X_MIN} m"),
    plt.Line2D([0], [0], color="#1565c0", linestyle="-.", linewidth=1.2,
               label=f"Bowl x_min = {BOWL_X_MIN} m"),
    plt.Line2D([0], [0], color="#7b1fa2", linestyle=":", linewidth=1.0,
               label=f"Cube y_band  ±{CUBE_Y_MAX} m"
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