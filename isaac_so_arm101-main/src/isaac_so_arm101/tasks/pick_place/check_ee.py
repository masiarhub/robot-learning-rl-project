import argparse
import torch
import PIL.Image as im
import numpy as np
import traceback
import math

# 1. Launcher Setup
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--ee-align",
    action="store_true",
    default=False,
    help="If set, cube tracks the EE. Otherwise spawns randomly on the floor with physics.",
)
parser.add_argument(
    "--cube-rot",
    type=float,
    default=None,
    help="Fixed Z rotation for the cube in degrees. If not set, randomized in floor mode.",
)
args_cli = parser.parse_args()
app_launcher = AppLauncher({"headless": args_cli.headless, "enable_cameras": True})
simulation_app = app_launcher.app

# 2. Imports
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import RigidObjectCfg, ArticulationCfg, AssetBaseCfg 
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, CameraCfg
from isaaclab.utils import configclass
from isaac_so_arm101.robots import SO_ARM101_CFG
import omni.usd
from pxr import UsdPhysics

@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    def __init__(self):
        super().__init__(num_envs=1, env_spacing=2.0)

        # ── Ground Plane ────────────────────────────────────────────────────
        self.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )
        
        # ── Background Structure (Blue Table matching Robot Base Height) ────
        self.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 1.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.6), roughness=0.5)
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.1))
        )
        
        # ── Robot Config ────────────────────────────────────────────────────
        self.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=SO_ARM101_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.2),  
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan":  0.0,
                    "shoulder_lift": -0.4,
                    "elbow_flex":    -0.3,
                    "wrist_flex":    1.57,
                    "wrist_roll":    -1.57,
                    "gripper":       0.2,
                },
            )
        )
        
        # ── 2cm Target Cube (FIXED: collision_props=None prevents physics ejection) ──
        self.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,   # ← makes it fully kinematic
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
        )

        # ── Crosshair Markers ───────────────────────────────────────────────
        m_len = 0.06
        self.ee_x = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marker_X",
            spawn=sim_utils.CuboidCfg(
                size=(m_len, 0.002, 0.002), 
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            ),
        )
        self.ee_y = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marker_Y",
            spawn=sim_utils.CuboidCfg(
                size=(0.002, m_len, 0.002), 
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            ),
        )
        self.ee_z = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Marker_Z",
            spawn=sim_utils.CuboidCfg(
                size=(0.002, 0.002, m_len), 
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
            ),
        )

        # ── End Effector Frame Transformer ──────────────────────────────────
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[-0.018, 0.0, 0.0]),
                ),
            ]
        )

        # ── High-Resolution Wrist Camera ────────────────────────────────────
        self.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
            update_period=0.0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=9.8,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 5.0),
            ),
            width=1280, height=720,
            offset=CameraCfg.OffsetCfg(
                pos=(-0.0049, 0.0498, -0.0591),
                rot=(0.9537, -0.3035, 0.0, 0.0),
                convention="opengl",
            ),
        )
def quat_mul(q1, q2):
    """Hamilton product: both [w, x, y, z]"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def z_rot_quat(angle_rad: float, device="cuda:0"):
    """Returns [w, x, y, z] quaternion for rotation around global Z."""
    half = angle_rad / 2.0
    return torch.tensor([
        math.cos(half), 0.0, 0.0, math.sin(half)
    ], device=device)

def set_object_kinematic(scene, kinematic: bool):
    from pxr import UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
    body = UsdPhysics.RigidBodyAPI(prim)
    body.GetKinematicEnabledAttr().Set(kinematic)
    physx_body = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_body.GetDisableGravityAttr().Set(kinematic)

def run_check(ee_alignment_mode: bool, cube_rot_deg: float | None):
    try:
        sim_context = sim_utils.SimulationContext(SimulationCfg(device="cuda:0"))
        scene = InteractiveScene(DebugSceneCfg())
        
        # Global Light Configuration
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/SunLight", light_cfg)

        sim_context.play()
        print("[INFO] Warming up renderer...")
        for _ in range(100): sim_context.step()
            
        scene.reset()
        scene.update(0.0)
        
        # Extract target positions to settle joint values
        robot = scene["robot"]
        sim_joint_names = robot.data.joint_names
        targets = robot.data.default_joint_pos.clone()
        for joint_name, angle_value in robot.cfg.init_state.joint_pos.items():
            if joint_name in sim_joint_names:
                targets[0, sim_joint_names.index(joint_name)] = angle_value
        
        # Step once to let Forward Kinematics process world positions
        robot.write_joint_state_to_sim(targets, torch.zeros_like(targets))
        sim_context.step(render=False)
        scene.update(0.0)
        
        # ── FIXED: Extract both tracking position AND true orientation orientation ──
        ee_pos_w = scene["ee_frame"].data.target_pos_w[0, 0]
        ee_quat_w = scene["ee_frame"].data.target_quat_w[0, 0]
        print(f"[INFO] Frame Transformer Resolved Position: {ee_pos_w.tolist()}")
        print(f"[INFO] Frame Transformer Resolved Orientation: {ee_quat_w.tolist()}")
        teleport_pose = torch.cat([ee_pos_w, ee_quat_w], dim=-1).unsqueeze(0)
        # Combine true position and true rotation together
        if ee_alignment_mode:
            # Kinematic: lock to EE with Z rotation
            set_object_kinematic(scene, True)

            z_angle = math.radians(cube_rot_deg) if cube_rot_deg is not None else 0.0
            z_quat = z_rot_quat(z_angle, device=ee_quat_w.device)
            rotated_quat = quat_mul(z_quat, ee_quat_w)
            object_pose = torch.cat([ee_pos_w, rotated_quat], dim=-1).unsqueeze(0)
            scene["object"].write_root_pose_to_sim(object_pose)

        else:
            # Physics: random XY on floor, random Z rotation, gravity enabled
            set_object_kinematic(scene, False)

            rand_x = 0.2 + torch.rand(1).item() * 0.3        # 0.2–0.5m in front
            rand_y = (torch.rand(1).item() - 0.5) * 0.3      # ±0.15m lateral
            #rand_z_angle = math.pi / 4 
            rand_z_angle = math.radians(cube_rot_deg) if cube_rot_deg is not None else torch.rand(1).item() * 2 * math.pi
            rand_quat = z_rot_quat(rand_z_angle, device=ee_quat_w.device)

            floor_pos = torch.tensor([rand_x, rand_y, 0.21], device=ee_quat_w.device)  # 0.2 table + half cube
            floor_pose = torch.cat([floor_pos, rand_quat], dim=-1).unsqueeze(0)
            scene["object"].write_root_pose_to_sim(floor_pose)
            scene["object"].write_root_velocity_to_sim(
                torch.zeros(1, 6, device=ee_quat_w.device)
            )
        # Teleport assets onto the exact position matching the gripper's rotation orientation
        scene["ee_x"].write_root_pose_to_sim(teleport_pose)
        scene["ee_y"].write_root_pose_to_sim(teleport_pose)
        scene["ee_z"].write_root_pose_to_sim(teleport_pose)
        
        # Run render sequence loop maintaining joint assignments
        for _ in range(50): 
            robot.write_joint_state_to_sim(targets, torch.zeros_like(targets))
            sim_context.step()
            
        scene.update(0.0)
        
        # Extract visual sensor payload matrix
        rgb = scene["wrist_cam"].data.output["rgb"][0].cpu().numpy()
        output_filename = "check_ee_alignment.png"
        im.fromarray(rgb).save(output_filename)
        print(f"[SUCCESS] High-resolution diagnostic frame exported to: {output_filename}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    run_check(ee_alignment_mode=args_cli.ee_align, cube_rot_deg=args_cli.cube_rot)
    simulation_app.close()
