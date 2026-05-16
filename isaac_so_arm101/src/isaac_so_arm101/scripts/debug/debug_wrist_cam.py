"""Debug script: capture what the wrist camera sees at the robot's initial joint configuration.

Saves one PNG per step for the first N steps (zero actions = robot stays at init pose),
plus a top-down workspace diagram with the cube/bowl spawn regions and camera FOV info.

Usage (from isaac_so_arm101/):
    python src/isaac_so_arm101/scripts/debug/debug_wrist_cam.py \\
        --task Isaac-SO-ARM101-Task-One-Distill-v0 \\
        --num_envs 1 --headless --enable_cameras

Output:
    ~/robot-learning/debug_wrist_cam/<TIMESTAMP>/
        step_000_wrist.png   ← image right after reset (init pose)
        step_001_wrist.png
        ...
        workspace_topdown.png ← annotated top-down view of spawn regions
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wrist camera debug view at robot init pose.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-Task-One-Distill-v0",
                    help="Task name. Must be a camera-enabled variant (Distill / CamPPO / PostTrain).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_steps", type=int, default=15,
                    help="Number of zero-action steps to capture.")
parser.add_argument("--out_dir", type=str, default="~/robot-learning/debug_wrist_cam",
                    help="Root output directory.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera rendering requires --enable_cameras; set it automatically if missing.
if not getattr(args_cli, "enable_cameras", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""---- everything below runs inside the Isaac Sim process ----"""

import datetime
import os

import numpy as np
import torch

import gymnasium as gym

import isaac_so_arm101.tasks  # noqa: F401  — registers all gym envs
from isaaclab_tasks.utils import parse_env_cfg
from isaac_so_arm101.tasks.task_1._wrist_cam import (
    FOCAL_LENGTH_MM as _CAM_FOCAL_MM,
    HORIZONTAL_APERTURE_MM as _CAM_H_APERTURE_MM,
    IMAGE_WIDTH as _CAM_IMG_W,
    IMAGE_HEIGHT as _CAM_IMG_H,
)


def _fov_deg() -> tuple[float, float]:
    v_aperture = _CAM_H_APERTURE_MM * (_CAM_IMG_H / _CAM_IMG_W)
    hfov = 2.0 * np.degrees(np.arctan(_CAM_H_APERTURE_MM / (2.0 * _CAM_FOCAL_MM)))
    vfov = 2.0 * np.degrees(np.arctan(v_aperture / (2.0 * _CAM_FOCAL_MM)))
    return hfov, vfov


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # ── output folder ──────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.expanduser(os.path.join(args_cli.out_dir, ts))
    os.makedirs(out_dir, exist_ok=True)

    hfov, vfov = _fov_deg()
    print(f"\n[wrist-cam-debug] Task          : {args_cli.task}")
    print(f"[wrist-cam-debug] Camera FOV    : HFOV={hfov:.1f}°  VFOV={vfov:.1f}°")
    print(f"[wrist-cam-debug] Output dir    : {out_dir}\n")

    env.reset()

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            obs, _, terminated, truncated, _ = env.step(zero_action)

        # ── grab raw camera image ──────────────────────────────────────────
        img_np = _grab_camera_rgb(env, "wrist_camera")
        if img_np is not None:
            import imageio
            path = os.path.join(out_dir, f"step_{step:03d}_wrist.png")
            imageio.imwrite(path, img_np)
        else:
            path = None

        # ── camera world position ─────────────────────────────────────────
        cam_pos, cam_quat = _get_camera_world_pose(env)
        cam_str = f"[{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]" if cam_pos is not None else "N/A"
        print(f"  step {step:02d}  cam_world_pos={cam_str}  → {os.path.basename(path) if path else '(no image)'}")

        if (terminated | truncated).any():
            print("  [episode ended — resetting]")
            env.reset()

    # ── top-down workspace diagram ─────────────────────────────────────────
    cam_pos_at_init, cam_quat_at_init = _get_camera_world_pose(env)
    _save_topdown_diagram(out_dir, cam_pos_at_init, cam_quat_at_init, hfov, vfov)

    env.close()
    print(f"\n[wrist-cam-debug] Done → {out_dir}")
    print(f"[wrist-cam-debug] Open step_000_wrist.png for the init-pose camera view.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grab_camera_rgb(env, sensor_name: str) -> np.ndarray | None:
    """Return a uint8 (H, W, 3) array from env 0, or None on failure."""
    try:
        scene = env.unwrapped.scene
        # Try dict access first (sensors stored in scene.sensors dict)
        sensor = scene.sensors.get(sensor_name) if hasattr(scene, "sensors") else None
        if sensor is None:
            sensor = scene[sensor_name]          # fallback: direct scene access
        raw = sensor.data.output.get("rgb")      # (num_envs, H, W, 3 or 4)
        if raw is None:
            return None
        return raw[0, ..., :3].cpu().clamp(0, 255).to(torch.uint8).numpy()
    except Exception as exc:
        print(f"  [wrist-cam-debug] Could not read '{sensor_name}': {exc}")
        return None


def _get_camera_world_pose(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute wrist camera world position and orientation from gripper_link FK.

    Returns:
        cam_pos_w:       (3,) camera position in world frame.
        cam_quat_wxyz:   (4,) camera orientation in world frame, wxyz convention.
    """
    try:
        from scipy.spatial.transform import Rotation
        from isaaclab.utils.math import quat_apply
        from isaac_so_arm101.tasks.task_1._wrist_cam import OFFSET_POS, OFFSET_QUAT_WXYZ

        robot = env.unwrapped.scene["robot"]
        body_idx = robot.find_bodies(["gripper_link"])[0][0]

        gripper_pos_w  = robot.data.body_pos_w[0, body_idx, :].cpu()   # (3,)
        gripper_quat_w = robot.data.body_quat_w[0, body_idx, :].cpu()  # (4,) wxyz

        # Camera position: rotate offset into world frame and add to gripper origin.
        cam_offset = torch.tensor(OFFSET_POS)
        cam_pos_w = gripper_pos_w + quat_apply(
            gripper_quat_w.unsqueeze(0), cam_offset.unsqueeze(0)
        ).squeeze(0)

        # Camera orientation: compose gripper world quat with local camera tilt.
        # scipy uses xyzw; our convention is wxyz.
        w, x, y, z = gripper_quat_w.tolist()
        R_gripper = Rotation.from_quat([x, y, z, w])
        ow, ox, oy, oz = OFFSET_QUAT_WXYZ
        R_cam_local = Rotation.from_quat([ox, oy, oz, ow])
        R_cam_w = R_gripper * R_cam_local
        x, y, z, w = R_cam_w.as_quat()   # scipy returns xyzw
        cam_quat_wxyz = np.array([w, x, y, z])

        return cam_pos_w.numpy(), cam_quat_wxyz
    except Exception as exc:
        print(f"  [wrist-cam-debug] Could not compute camera pose: {exc}")
        return None, None


def _project_frustum_corners_to_z0(
    cam_pos: np.ndarray,
    cam_quat_wxyz: np.ndarray,
    hfov_deg: float,
    vfov_deg: float,
) -> np.ndarray | None:
    """Project the 4 image corner rays from the camera onto the z=0 table plane.

    Returns an (4, 2) array of (x, y) intersection points, or None if any ray
    points away from the table (camera looking up).
    """
    from scipy.spatial.transform import Rotation

    hfov = np.radians(hfov_deg)
    vfov = np.radians(vfov_deg)

    # Corner directions in OpenGL camera frame (camera looks along -Z)
    # OpenGL: X=right, Y=up, Z=backward
    # Image corners: (±hfov/2, ±vfov/2)
    corners_cam = np.array([
        [ np.tan(hfov / 2),  np.tan(vfov / 2), -1.0],  # top-right
        [-np.tan(hfov / 2),  np.tan(vfov / 2), -1.0],  # top-left
        [-np.tan(hfov / 2), -np.tan(vfov / 2), -1.0],  # bottom-left
        [ np.tan(hfov / 2), -np.tan(vfov / 2), -1.0],  # bottom-right
    ])
    corners_cam /= np.linalg.norm(corners_cam, axis=1, keepdims=True)

    # Convert wxyz quaternion → scipy Rotation
    w, x, y, z = cam_quat_wxyz
    rot = Rotation.from_quat([x, y, z, w])  # scipy uses xyzw

    # Rotate corners into world frame
    corners_world = rot.apply(corners_cam)  # (4, 3)

    # Intersect each ray with z=0 plane: P = cam_pos + t * direction, P.z = 0
    # t = -cam_pos.z / direction.z
    results = []
    for d in corners_world:
        if abs(d[2]) < 1e-6 or d[2] > 0:   # ray horizontal or pointing up
            return None
        t = -cam_pos[2] / d[2]
        if t < 0:
            return None
        pt = cam_pos[:2] + t * d[:2]
        results.append(pt)
    return np.array(results)


def _save_topdown_diagram(
    out_dir: str,
    cam_pos: np.ndarray | None,
    cam_quat_wxyz: np.ndarray | None,
    hfov: float,
    vfov: float,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon as MplPolygon
    except ImportError:
        print("[wrist-cam-debug] matplotlib not available — skipping top-down diagram.")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    theta = np.linspace(0, 2 * np.pi, 360)
    px, py = 0.048, 0.0  # placement point

    # ── Cube spawn region ───────────────────────────────────────────────────
    for r in [0.15, 0.30]:
        ax.plot(px + r * np.cos(theta), py + r * np.sin(theta), "--", color="steelblue", alpha=0.5)
    ax.fill_between(np.linspace(0.148, 0.35, 200), -0.20, 0.20,
                    color="steelblue", alpha=0.15, label="Cube spawn region")

    # ── Bowl spawn region ───────────────────────────────────────────────────
    for r in [0.20, 0.40]:
        ax.plot(px + r * np.cos(theta), py + r * np.sin(theta), "-.", color="seagreen", alpha=0.5)
    ax.fill_between(np.linspace(0.148, 0.45, 200), -0.20, 0.20,
                    color="seagreen", alpha=0.08, label="Bowl spawn region")

    # ── Robot base ──────────────────────────────────────────────────────────
    ax.plot(0, 0, "k+", markersize=18, linewidth=3, zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.04, color="black", alpha=0.5, zorder=5))
    ax.text(0.01, 0.05, "robot\nbase", fontsize=8, ha="center")

    # ── Placement point ─────────────────────────────────────────────────────
    ax.plot(px, py, "r*", markersize=13, zorder=5, label=f"Placement point ({px:.3f}, {py:.3f})")

    # ── Table ───────────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle((-0.4, -0.6), 0.8, 1.2,
                                    fill=False, edgecolor="saddlebrown",
                                    linewidth=2.5, label="Table (≈0.8×1.2 m)"))

    # ── Camera position and frustum footprint on table ──────────────────────
    if cam_pos is not None:
        ax.plot(cam_pos[0], cam_pos[1], "m^", markersize=12, zorder=6,
                label=f"Camera pos ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, z={cam_pos[2]:.3f})")

        if cam_quat_wxyz is not None:
            footprint = _project_frustum_corners_to_z0(cam_pos, cam_quat_wxyz, hfov, vfov)
            if footprint is not None:
                # Reorder corners: top-right, top-left, bottom-left, bottom-right → convex hull order
                poly = MplPolygon(
                    footprint[[0, 1, 2, 3]],
                    closed=True,
                    facecolor="magenta", alpha=0.20,
                    edgecolor="magenta", linewidth=1.5,
                    label="Camera FOV footprint (z=0 plane)",
                    zorder=4,
                )
                ax.add_patch(poly)
                # Lines from camera ground-projection to each corner
                for pt in footprint:
                    ax.plot([cam_pos[0], pt[0]], [cam_pos[1], pt[1]],
                            color="magenta", linewidth=0.8, alpha=0.5, zorder=4)
            else:
                ax.text(cam_pos[0], cam_pos[1] + 0.03, "frustum\nnot on table",
                        ha="center", fontsize=7, color="magenta")

    # ── Annotation ─────────────────────────────────────────────────────────
    ax.text(0.97, 0.97,
            f"Wrist camera intrinsics\nHFOV ≈ {hfov:.0f}°  VFOV ≈ {vfov:.0f}°\n"
            f"Sensor: {_CAM_IMG_W}×{_CAM_IMG_H} px\nfocal: {_CAM_FOCAL_MM} mm",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))

    ax.text(0.03, 0.03,
            "▲ Check step_000_wrist.png for the actual\n  camera image at init pose.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="#ffe0e0", alpha=0.8))

    ax.set_xlim(-0.5, 0.70)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]  (forward from robot base)", fontsize=11)
    ax.set_ylabel("Y [m]  (left)", fontsize=11)
    ax.set_title("Workspace top-down view — Task 1  (init pose)\n"
                 "Magenta polygon = camera FOV footprint on table", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)

    path = os.path.join(out_dir, "workspace_topdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrist-cam-debug] Top-down diagram → {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
