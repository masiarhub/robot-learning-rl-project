"""Interactive wrist camera viewer.

Spawns SO-ARM101 directly (no RL env, no GPU direct API) so the Isaac Sim
physics inspector works normally alongside keyboard joint control.

A secondary viewport is opened showing the wrist camera's live POV.

Usage (from isaac_so_arm101/, do NOT use --headless):
    python src/isaac_so_arm101/scripts/debug/interactive_wrist_cam.py

Optional: start from a custom pose (radians):
    python src/isaac_so_arm101/scripts/debug/interactive_wrist_cam.py \\
        --shoulder_lift -1.0 --elbow_flex 1.2 --wrist_flex 0.8

Keyboard controls (click Isaac Sim viewport first to give it focus):
    Q / A  —  shoulder_pan   + / -   (held = continuous)
    W / S  —  shoulder_lift  + / -
    E / D  —  elbow_flex     + / -
    R / F  —  wrist_flex     + / -
    T / G  —  wrist_roll     + / -
    Y / H  —  gripper open / close
    Space  —  reset to starting pose
    P      —  print current joint targets
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive wrist camera viewer for SO-ARM101.")
parser.add_argument("--shoulder_pan",  type=float, default=None, metavar="RAD")
parser.add_argument("--shoulder_lift", type=float, default=None, metavar="RAD")
parser.add_argument("--elbow_flex",    type=float, default=None, metavar="RAD")
parser.add_argument("--wrist_flex",    type=float, default=None, metavar="RAD")
parser.add_argument("--wrist_roll",    type=float, default=None, metavar="RAD")
parser.add_argument("--gripper",       type=float, default=None, metavar="RAD")
parser.add_argument("--step_size",     type=float, default=0.008,
                    help="Joint delta per sim step while a key is held (rad, default 0.008).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""---- everything below runs inside the Isaac Sim process ----"""

import torch
from pxr import UsdGeom, Gf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from isaac_so_arm101.robots.trs_so101.so_arm101 import SO_ARM101_CFG
from isaac_so_arm101.tasks.task_1._wrist_cam import (
    FOCAL_LENGTH_MM,
    HORIZONTAL_APERTURE_MM,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    OFFSET_POS,
    OFFSET_QUAT_WXYZ,
)

# ── Keyboard state ─────────────────────────────────────────────────────────────
_keys_held: set = set()
_reset_flag:  list[bool] = [False]
_print_flag:  list[bool] = [False]

def _subscribe_keyboard():
    import carb.input
    import omni.appwindow
    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    ET = carb.input.KeyboardEventType

    def _on_key(event, *_, **__):
        if event.type == ET.KEY_PRESS:
            _keys_held.add(event.input)
            K = carb.input.KeyboardInput
            if event.input == K.SPACE:
                _reset_flag[0] = True
            elif event.input == K.P:
                _print_flag[0] = True
        elif event.type == ET.KEY_RELEASE:
            _keys_held.discard(event.input)
        return True

    sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_key)
    return input_iface, keyboard, sub


def _apply_keys(targets: dict[str, float], step: float) -> dict[str, float]:
    import carb.input
    K = carb.input.KeyboardInput
    mapping = [
        (K.Q, "shoulder_pan",  +step),
        (K.A, "shoulder_pan",  -step),
        (K.W, "shoulder_lift", +step),
        (K.S, "shoulder_lift", -step),
        (K.E, "elbow_flex",    +step),
        (K.D, "elbow_flex",    -step),
        (K.R, "wrist_flex",    +step),
        (K.F, "wrist_flex",    -step),
        (K.T, "wrist_roll",    +step),
        (K.G, "wrist_roll",    -step),
        (K.Y, "gripper",       +step),
        (K.H, "gripper",       -step),
    ]
    out = dict(targets)
    for key, joint, delta in mapping:
        if key in _keys_held and joint in out:
            out[joint] += delta
    return out


# ── Camera helpers ─────────────────────────────────────────────────────────────

def _create_camera_prim(gripper_prim_path: str) -> str:
    """Create a USD Camera as a child of gripper_link with the wrist-cam offset."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    cam_path = f"{gripper_prim_path}/wrist_camera"

    camera = UsdGeom.Camera.Define(stage, cam_path)
    xform  = UsdGeom.Xformable(camera.GetPrim())

    # Local translation
    tx, ty, tz = OFFSET_POS
    xform.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))

    # Local rotation (wxyz → Gf.Quatf)
    w, x, y, z = OFFSET_QUAT_WXYZ
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(float(w), float(x), float(y), float(z))
    )

    # Intrinsics
    v_aperture = HORIZONTAL_APERTURE_MM * (IMAGE_HEIGHT / IMAGE_WIDTH)
    camera.GetFocalLengthAttr().Set(float(FOCAL_LENGTH_MM))
    camera.GetHorizontalApertureAttr().Set(float(HORIZONTAL_APERTURE_MM))
    camera.GetVerticalApertureAttr().Set(float(v_aperture))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.005, 10.0))

    return cam_path


def _open_camera_viewport(cam_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vp_util
        vp_win = vp_util.create_viewport_window("Wrist Camera", width=512, height=288)
        try:
            vp_win.viewport_api.set_active_camera(cam_path)
        except AttributeError:
            vp_win.viewport_api.camera_path = cam_path
        print(f"[interactive] Wrist camera viewport opened  (camera prim: {cam_path})")
    except Exception as exc:
        print(f"[interactive] Could not auto-open camera viewport: {exc}")
        print(f"[interactive] → Open a new viewport manually and set its camera to: {cam_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # device="cpu" keeps PhysX on the CPU pipeline, which does NOT set
    # PxSceneFlag::eENABLE_DIRECT_GPU_API → physics inspector works normally.
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.2, 0.8, 0.9], target=[0.3, 0.0, 0.1])

    # ── Scene ──────────────────────────────────────────────────────────────────
    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/GroundPlane", plane_cfg)

    light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=600.0)
    light_cfg.func("/World/Light", light_cfg)

    # Table — same geometry as the task env
    table_cfg = sim_utils.CuboidCfg(
        size=(0.8, 1.2, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.40, 0.25)),
    )
    table_cfg.func("/World/Table", table_cfg, translation=(0.4, 0.0, -0.5))

    # Robot — apply any CLI joint overrides to the init pose
    robot_cfg = SO_ARM101_CFG.replace(prim_path="/World/Robot")
    init_joints: dict[str, float] = dict(robot_cfg.init_state.joint_pos)
    for name in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"):
        val = getattr(args_cli, name, None)
        if val is not None:
            init_joints[name] = float(val)
    robot_cfg.init_state.joint_pos = init_joints
    robot = Articulation(cfg=robot_cfg)

    # ── Init sim ───────────────────────────────────────────────────────────────
    sim.reset()
    sim_dt = sim.get_physics_dt()

    # Initial robot state read-back
    robot.update(sim_dt)
    joint_names: list[str] = list(robot.data.joint_names)

    # Print starting configuration
    print("\n[interactive] Joint order from robot:", joint_names)
    print("[interactive] Starting joint targets (rad):")
    for name, val in init_joints.items():
        print(f"    {name:<20s}: {val:.4f}")
    print()
    print("[interactive] Controls:")
    print("    Q/A  shoulder_pan  W/S  shoulder_lift  E/D  elbow_flex")
    print("    R/F  wrist_flex   T/G  wrist_roll      Y/H  gripper")
    print("    Space=reset   P=print joints\n")

    # ── Wrist camera prim (child of gripper_link → follows it automatically) ──
    cam_path = _create_camera_prim("/World/Robot/gripper_link")
    _open_camera_viewport(cam_path)

    # ── Keyboard ───────────────────────────────────────────────────────────────
    input_iface, keyboard, kbd_sub = _subscribe_keyboard()

    # Current joint targets (ordered to match robot.data.joint_names)
    joint_targets = {name: init_joints.get(name, 0.0) for name in joint_names}

    # ── Simulation loop ────────────────────────────────────────────────────────
    while simulation_app.is_running():
        # Reset
        if _reset_flag[0]:
            joint_targets = {name: init_joints.get(name, 0.0) for name in joint_names}
            _reset_flag[0] = False
            print("[interactive] Reset to starting pose.")

        # Keyboard deltas
        joint_targets = _apply_keys(joint_targets, args_cli.step_size)

        # Print on request
        if _print_flag[0]:
            actual = robot.data.joint_pos[0].tolist()  # actual positions from sim
            print(f"\n[interactive] {'Joint':<20s}  {'target':>10s}  {'actual':>10s}  {'error':>10s}")
            print(f"[interactive] {'-'*56}")
            for i, name in enumerate(joint_names):
                tgt = joint_targets[name]
                act = actual[i]
                print(f"[interactive] {name:<20s}  {tgt:>10.4f}  {act:>10.4f}  {act - tgt:>+10.4f}")
            print()
            _print_flag[0] = False

        # Build target tensor in robot's joint order and send to sim
        target_tensor = torch.tensor(
            [[joint_targets[n] for n in joint_names]], dtype=torch.float32
        )
        robot.set_joint_position_target(target_tensor)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    input_iface.unsubscribe_to_keyboard_events(keyboard, kbd_sub)


if __name__ == "__main__":
    main()
    simulation_app.close()
