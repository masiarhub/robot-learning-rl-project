"""Debug script: capture what the wrist camera sees at the robot's initial joint configuration.

Saves one PNG per step for the first N steps (zero actions = robot stays at init pose),
plus a top-down workspace diagram with the cube/bowl spawn regions and camera FOV info.

Usage (from isaac_so_arm101/):
    python src/isaac_so_arm101/scripts/debug/debug_wrist_cam.py \\
        --task Isaac-SO-ARM101-Task-One-Distill-v0 \\
        --num_envs 1 --headless --enable_cameras

    # Override individual joint angles (rad):
    python src/isaac_so_arm101/scripts/debug/debug_wrist_cam.py \\
        --task Isaac-SO-ARM101-Task-One-Distill-v0 --headless --enable_cameras \\
        --shoulder_lift -1.0 --elbow_flex 1.2 --wrist_flex 0.8

    # Place cube at custom (x, y) in robot frame:
    python src/isaac_so_arm101/scripts/debug/debug_wrist_cam.py \\
        --task Isaac-SO-ARM101-Task-One-Distill-v0 --headless --enable_cameras \\
        --cube_xy 0.25 0.05

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

# ── Joint position overrides ───────────────────────────────────────────────
parser.add_argument("--shoulder_pan",  type=float, default=None,
                    help="Override shoulder_pan initial joint angle (rad).")
parser.add_argument("--shoulder_lift", type=float, default=None,
                    help="Override shoulder_lift initial joint angle (rad).")
parser.add_argument("--elbow_flex",    type=float, default=None,
                    help="Override elbow_flex initial joint angle (rad).")
parser.add_argument("--wrist_flex",    type=float, default=None,
                    help="Override wrist_flex initial joint angle (rad).")
parser.add_argument("--wrist_roll",    type=float, default=None,
                    help="Override wrist_roll initial joint angle (rad).")

# ── Optional cube placement ────────────────────────────────────────────────
parser.add_argument("--cube_xy", nargs=2, type=float, default=None, metavar=("X", "Y"),
                    help="Force the cube to (X, Y) in the robot root frame after reset "
                         "(e.g. --cube_xy 0.25 0.05). Keeps the cube's default Z height.")

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

    # ── Apply joint position overrides before env creation ─────────────────
    joint_overrides = {
        "shoulder_pan":  args_cli.shoulder_pan,
        "shoulder_lift": args_cli.shoulder_lift,
        "elbow_flex":    args_cli.elbow_flex,
        "wrist_flex":    args_cli.wrist_flex,
        "wrist_roll":    args_cli.wrist_roll,
    }
    default_joints = dict(env_cfg.scene.robot.init_state.joint_pos)
    for name, val in joint_overrides.items():
        if val is not None:
            env_cfg.scene.robot.init_state.joint_pos[name] = val

    env = gym.make(args_cli.task, cfg=env_cfg)

    # ── output folder ──────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.expanduser(os.path.join(args_cli.out_dir, ts))
    os.makedirs(out_dir, exist_ok=True)

    hfov, vfov = _fov_deg()
    print(f"\n[wrist-cam-debug] Task          : {args_cli.task}")
    print(f"[wrist-cam-debug] Camera FOV    : HFOV={hfov:.1f}°  VFOV={vfov:.1f}°")
    print(f"[wrist-cam-debug] Output dir    : {out_dir}")

    # Print joint config
    print("\n[wrist-cam-debug] Joint positions (rad):")
    final_joints = dict(env_cfg.scene.robot.init_state.joint_pos)
    for name, default in default_joints.items():
        final = final_joints.get(name, default)
        tag = f"  ← overridden (was {default:.4f})" if final != default else ""
        print(f"    {name:<20s}: {final:.4f}{tag}")

    if args_cli.cube_xy is not None:
        print(f"\n[wrist-cam-debug] Cube override  : x={args_cli.cube_xy[0]:.3f}  y={args_cli.cube_xy[1]:.3f}")
    print()

    env.reset()

    # ── Override cube position after reset ─────────────────────────────────
    cube_pos_override = None
    if args_cli.cube_xy is not None:
        cx, cy = args_cli.cube_xy
        cube_pos_override = _force_cube_xy(env, cx, cy)

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
            if args_cli.cube_xy is not None:
                cx, cy = args_cli.cube_xy
                _force_cube_xy(env, cx, cy)

    # ── top-down workspace diagram ─────────────────────────────────────────
    cam_pos_at_init, cam_quat_at_init = _get_camera_world_pose(env)
    _save_topdown_diagram(out_dir, cam_pos_at_init, cam_quat_at_init, hfov, vfov,
                          cube_override_xy=args_cli.cube_xy)

    env.close()
    print(f"\n[wrist-cam-debug] Done → {out_dir}")
    print(f"[wrist-cam-debug] Open step_000_wrist.png for the init-pose camera view.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _force_cube_xy(env, cx: float, cy: float) -> tuple[float, float] | None:
    """Move the cube to (cx, cy) in the robot root frame, keeping its default z.

    Returns the (cx, cy) pair on success, None on failure.
    """
    try:
        obj = env.unwrapped.scene["object"]
        env_ids = torch.tensor([0], device=env.unwrapped.device)
        env_origin = env.unwrapped.scene.env_origins[0]  # (3,)

        state = obj.data.default_root_state[0:1].clone()  # (1, 13)
        state[0, 0] = cx + env_origin[0]
        state[0, 1] = cy + env_origin[1]
        state[0, 2] = obj.data.default_root_state[0, 2] + env_origin[2]
        state[0, 7:] = 0.0  # zero velocity

        obj.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)
        print(f"[wrist-cam-debug] Cube placed at robot-frame ({cx:.3f}, {cy:.3f}), "
              f"world z={state[0, 2].item():.3f}")
        return (cx, cy)
    except Exception as exc:
        print(f"[wrist-cam-debug] Could not place cube: {exc}")
        return None


def _grab_camera_rgb(env, sensor_name: str) -> np.ndarray | None:
    """Return a uint8 (H, W, 3) array from env 0, or None on failure."""
    try:
        scene = env.unwrapped.scene
        sensor = scene.sensors.get(sensor_name) if hasattr(scene, "sensors") else None
        if sensor is None:
            sensor = scene[sensor_name]
        raw = sensor.data.output.get("rgb")      # (num_envs, H, W, 3 or 4)
        if raw is None:
            return None
        return raw[0, ..., :3].cpu().clamp(0, 255).to(torch.uint8).numpy()
    except Exception as exc:
        print(f"  [wrist-cam-debug] Could not read '{sensor_name}': {exc}")
        return None


def _get_camera_world_pose(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute wrist camera world position and orientation from gripper_link FK."""
    try:
        from scipy.spatial.transform import Rotation
        from isaaclab.utils.math import quat_apply
        from isaac_so_arm101.tasks.task_1._wrist_cam import OFFSET_POS, OFFSET_QUAT_WXYZ

        robot = env.unwrapped.scene["robot"]
        body_idx = robot.find_bodies(["gripper_link"])[0][0]

        gripper_pos_w  = robot.data.body_pos_w[0, body_idx, :].cpu()
        gripper_quat_w = robot.data.body_quat_w[0, body_idx, :].cpu()

        cam_offset = torch.tensor(OFFSET_POS)
        cam_pos_w = gripper_pos_w + quat_apply(
            gripper_quat_w.unsqueeze(0), cam_offset.unsqueeze(0)
        ).squeeze(0)

        w, x, y, z = gripper_quat_w.tolist()
        R_gripper = Rotation.from_quat([x, y, z, w])
        ow, ox, oy, oz = OFFSET_QUAT_WXYZ
        R_cam_local = Rotation.from_quat([ox, oy, oz, ow])
        R_cam_w = R_gripper * R_cam_local
        x, y, z, w = R_cam_w.as_quat()
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
    """Project the 4 image corner rays from the camera onto the z=0 table plane."""
    from scipy.spatial.transform import Rotation

    hfov = np.radians(hfov_deg)
    vfov = np.radians(vfov_deg)

    corners_cam = np.array([
        [ np.tan(hfov / 2),  np.tan(vfov / 2), -1.0],
        [-np.tan(hfov / 2),  np.tan(vfov / 2), -1.0],
        [-np.tan(hfov / 2), -np.tan(vfov / 2), -1.0],
        [ np.tan(hfov / 2), -np.tan(vfov / 2), -1.0],
    ])
    corners_cam /= np.linalg.norm(corners_cam, axis=1, keepdims=True)

    w, x, y, z = cam_quat_wxyz
    rot = Rotation.from_quat([x, y, z, w])
    corners_world = rot.apply(corners_cam)

    results = []
    for d in corners_world:
        if abs(d[2]) < 1e-6 or d[2] > 0:
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
    cube_override_xy: list[float] | None = None,
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
    px, py = 0.048, 0.0

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

    # ── Forced cube position ────────────────────────────────────────────────
    if cube_override_xy is not None:
        cx, cy = cube_override_xy
        ax.plot(cx, cy, "s", color="darkorange", markersize=14, zorder=7,
                label=f"Cube override ({cx:.3f}, {cy:.3f})")
        ax.add_patch(plt.Circle((cx, cy), 0.015, color="darkorange", alpha=0.6, zorder=7))

    # ── Camera position and frustum footprint on table ──────────────────────
    if cam_pos is not None:
        ax.plot(cam_pos[0], cam_pos[1], "m^", markersize=12, zorder=6,
                label=f"Camera pos ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, z={cam_pos[2]:.3f})")

        if cam_quat_wxyz is not None:
            footprint = _project_frustum_corners_to_z0(cam_pos, cam_quat_wxyz, hfov, vfov)
            if footprint is not None:
                poly = MplPolygon(
                    footprint[[0, 1, 2, 3]],
                    closed=True,
                    facecolor="magenta", alpha=0.20,
                    edgecolor="magenta", linewidth=1.5,
                    label="Camera FOV footprint (z=0 plane)",
                    zorder=4,
                )
                ax.add_patch(poly)
                for pt in footprint:
                    ax.plot([cam_pos[0], pt[0]], [cam_pos[1], pt[1]],
                            color="magenta", linewidth=0.8, alpha=0.5, zorder=4)

                # Mark whether the forced cube is inside the frustum footprint
                if cube_override_xy is not None:
                    from matplotlib.path import Path as MplPath
                    poly_path = MplPath(footprint[[0, 1, 2, 3]])
                    cx, cy = cube_override_xy
                    inside = poly_path.contains_point((cx, cy))
                    color = "green" if inside else "red"
                    label = "Cube IN frustum" if inside else "Cube NOT in frustum"
                    ax.text(cx + 0.02, cy + 0.02, label, fontsize=9, color=color,
                            fontweight="bold", zorder=8)
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
