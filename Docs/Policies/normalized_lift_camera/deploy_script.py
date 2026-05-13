"""
deploy_script.py  –  SO-ARM101 LiftCamera Policy Deployment
=============================================================
Runs an RSL-RL PPO policy (Isaac-SO-ARM101-LiftCamera) on the real SO-101 robot.

Policy was exported from Isaac Lab via play.py → export_policy_as_jit():
  logs/rsl_rl/lift_camera/<timestamp>/exported/policy.pt
The JIT file contains the actor MLP + normalizer (Identity if not trained with one).
Interface: policy(obs: Tensor[1, 536]) → action: Tensor[1, 6]

Observation layout (536-dim):
  [0:6]    joint_pos_rel         – joint pos - default pos   (rad)
  [6:12]   joint_vel_rel         – joint vel                 (rad/s)
  [12:15]  gripper_link_position – gripper FK in robot frame (m)
  [15:18]  bowl_position         – bowl pos + 0.12 m z       (m)
  [18:530] wrist_image           – frozen ResNet18 features  (512-dim)
  [530:536] last_action          – previous policy output    (6-dim)

Action layout (6-dim):
  [0:5]  JointPosition targets, scale=0.5, use_default_offset=True
         joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
  [5]    gripper binary: > 0 → open (0.5 rad), ≤ 0 → close (-0.1 rad)

Isaac Lab default joint positions:
  shoulder_pan=0.0, shoulder_lift=-0.6, elbow_flex=-0.6,
  wrist_flex=1.57, wrist_roll=-1.57, gripper=0.0  (rad)

Usage
-----
# Export the JIT policy first (run once in the IsaacLab venv):
  python src/isaac_so_arm101/scripts/rsl_rl/play.py \\
      --task Isaac-SO-ARM101-LiftCamera-Play-v0 --num_envs 1 \\
      --checkpoint logs/rsl_rl/lift_camera/<run>/model_2999.pt --headless
  # → exports to logs/rsl_rl/lift_camera/<run>/exported/policy.pt

# Measure LEROBOT_DEFAULT_DEG before first run:
  python check_joints.py --robot_id follower_arm
  # Move arm to Isaac Lab default pose, copy read_deg column into LEROBOT_DEFAULT_DEG below.

# Deploy:
  python deploy_script.py \\
      --checkpoint policy.pt \\
      --robot_port /dev/ttyACM0 --robot_id follower_arm \\
      --bowl_pos 0.20 0.10 0.00 \\
      --camera_device 4
"""

import argparse
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

from lerobot.robots.so_follower import SO100Follower, SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Control constants ──────────────────────────────────────────────────────────

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
NUM_ARM_JOINTS  = 5
NUM_ACTIONS     = 6   # 5 arm + 1 gripper
NUM_OBS         = 536  # 6+6+3+3+512+6

# Isaac Lab default joint positions (rad)
DEFAULT_JOINT_POS_RAD = np.array([0.0, -0.6, -0.6, 1.57, -1.57, 0.0], dtype=np.float64)

CONTROL_HZ       = 50      # sim.dt=0.01 s, decimation=2
BOWL_HOVER_HEIGHT = 0.12   # z-offset added to bowl pos in training
ARM_ACTION_SCALE  = 0.5    # JointPositionActionCfg scale
MAX_DELTA_DEG     = 3.0    # software safety cap per step (degrees)

GRIPPER_OPEN_CMD_RAD  =  0.5
GRIPPER_CLOSE_CMD_RAD = -0.1
GRIPPER_THRESHOLD     =  0.0   # raw policy output > 0 → open
GRIPPER_RAD_MIN       = GRIPPER_CLOSE_CMD_RAD
GRIPPER_RAD_MAX       = GRIPPER_OPEN_CMD_RAD

CAM_H      = 72
CAM_W      = 128
RESNET_DIM = 512

# ── Joint mapping: LeRobot ↔ Isaac Lab ────────────────────────────────────────
#
# JOINT_SIGN[i] = +1  when LeRobot and Isaac Lab share the same positive direction.
#               = -1  when they are opposite.
#
# Verified by movement (move joint +θ in LeRobot, check direction of isaac_rad change):
#   elbow_flex  → −1  (Isaac default −34.4°, LeRobot reads +43.47° → opposite → −1)
#   wrist_flex  → +1  (LeRobot moved −145° from default; sign=+1 gives −55° valid,
#                       sign=−1 gives +235° which exceeds joint limit → +1)
#
# NOT yet verified by movement — confirm before trusting:
#   shoulder_pan   → assumed +1  (both ≈ 0° at default; MUST verify by moving joint)
#   shoulder_lift  → assumed +1  (both negative at default; MUST verify by moving joint)
#   wrist_roll     → assumed +1  (both ≈ −84°/−90° at default; MUST verify by moving joint)
#
# To verify a sign: with the robot connected, run check_joints.py live, manually
# rotate the joint ~20° in the POSITIVE LeRobot direction, then check if
# isaac_rad_estimated increases (sign=+1) or decreases (sign=−1).
JOINT_SIGN = np.array([1.0, 1.0, -1.0, 1.0, 1.0])

# What LeRobot reads (degrees) when the arm is physically at the Isaac Lab default pose.
# Measured 2026-05-13 with check_joints.py (arm at default pose):
LEROBOT_DEFAULT_DEG = np.array([0.62, -49.32, 43.47, 88.44, -83.38])

def busy_wait(dt: float) -> None:
    t_end = time.perf_counter() + dt
    while time.perf_counter() < t_end:
        pass


# ── Policy loading ─────────────────────────────────────────────────────────────

def load_policy(path: str, device: str) -> torch.jit.ScriptModule:
    """Load a TorchScript policy exported by play.py → export_policy_as_jit().

    Interface: policy(obs: Tensor[1, 536]) → action: Tensor[1, 6]
    The normalizer is baked in (Identity if training used no normalizer).
    """
    model = torch.jit.load(path, map_location=device)
    model.eval()
    log.info(f"JIT policy loaded from {path} (device={device})")
    return model


# ── ResNet18 encoder (frozen, identical to training) ──────────────────────────

class FrozenResNet18Encoder(nn.Module):
    _mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        FrozenResNet18Encoder._mean = FrozenResNet18Encoder._mean.to(device)
        FrozenResNet18Encoder._std  = FrozenResNet18Encoder._std.to(device)
        return self

    @torch.no_grad()
    def encode_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        """(H, W, 3) BGR uint8 → (1, 512) feature tensor."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _center_crop_16_9(rgb)
        rgb = cv2.resize(rgb, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(rgb).float() / 255.0           # [H, W, 3]
        img = img.permute(2, 0, 1).unsqueeze(0)               # [1, 3, H, W]
        img = img.to(next(self.encoder.parameters()).device)
        img = (img - FrozenResNet18Encoder._mean) / FrozenResNet18Encoder._std
        return self.encoder(img).flatten(start_dim=1)          # [1, 512]


# ── Camera helpers ─────────────────────────────────────────────────────────────

_TRAIN_ASPECT = CAM_W / CAM_H  # 128/72 ≈ 1.778  (16:9)

def _center_crop_16_9(img: np.ndarray) -> np.ndarray:
    """Center-crop an HxWx3 image to 16:9 without squashing."""
    h, w = img.shape[:2]
    target_w = int(h * _TRAIN_ASPECT)
    target_h = int(w / _TRAIN_ASPECT)
    if target_w <= w:
        x0 = (w - target_w) // 2
        return img[:, x0:x0 + target_w]
    else:
        y0 = (h - target_h) // 2
        return img[y0:y0 + target_h, :]


# ── Camera ─────────────────────────────────────────────────────────────────────

class WristCamera:
    def __init__(self, camera_type: str = "usb", device_id: int = 0):
        self._type      = camera_type
        self._device_id = device_id
        self._cap       = None
        self._pipeline  = None

    def connect(self):
        if self._type == "usb":
            self._cap = cv2.VideoCapture(self._device_id)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self._device_id}. Check: ls /dev/video*")
            # MJPG lets most cameras reach 1280x720 even when raw YUV tops out at 640x480.
            # Must be set before requesting the resolution.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w != 1280 or actual_h != 720:
                log.warning(f"Camera {self._device_id}: requested 1280x720 MJPG but got {actual_w}x{actual_h} — center-crop will correct aspect ratio")
            log.info(f"USB camera {self._device_id} connected at {actual_w}x{actual_h}")
        elif self._type == "realsense":
            import pyrealsense2 as rs
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
            self._pipeline.start(cfg)
            log.info("RealSense connected")
        else:
            raise ValueError(f"Unknown camera type: {self._type}")

    def read_bgr(self) -> np.ndarray:
        if self._type == "usb":
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Camera read failed.")
            return frame
        else:
            import pyrealsense2 as rs
            frames = self._pipeline.wait_for_frames()
            f = frames.get_color_frame()
            if not f:
                raise RuntimeError("RealSense: no color frame.")
            return np.asanyarray(f.get_data())

    def disconnect(self):
        if self._cap is not None:
            self._cap.release()
        if self._pipeline is not None:
            self._pipeline.stop()
        log.info("Camera disconnected")


# ── Mock robot (for --mock mode) ───────────────────────────────────────────────

class _MockRobot:
    def connect(self):
        pass

    def disconnect(self):
        pass

    def get_observation(self) -> dict:
        # Return Isaac Lab default pose expressed as LeRobot degrees
        arm_lerobot = LEROBOT_DEFAULT_DEG.copy()
        return {
            **{f"{m}.pos": float(arm_lerobot[i]) for i, m in enumerate(ARM_JOINT_NAMES)},
            "gripper.pos": 50.0,
        }

    def send_action(self, action: dict):
        pass


# ── Forward kinematics (gripper_link position) ─────────────────────────────────

_URDF_PATH  = Path(__file__).parent / "so_arm101.urdf"
_FK_JOINTS  = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

def _parse_urdf_chain(urdf_path: Path, joint_names: list) -> list:
    """Return [(xyz, rpy), ...] for each named joint, in order, from a URDF file."""
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.iter("joint")}
    chain = []
    for name in joint_names:
        origin = by_name[name].find("origin")
        xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        chain.append((xyz, rpy))
    return chain

_SO101_JOINTS = _parse_urdf_chain(_URDF_PATH, _FK_JOINTS)

def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @
            np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @
            np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))

def compute_gripper_link_position(joint_pos_rad: np.ndarray) -> np.ndarray:
    """FK for gripper_link origin in robot root (base_link) frame."""
    T = np.eye(4)
    for (xyz, rpy), q in zip(_SO101_JOINTS, joint_pos_rad[:5]):
        c, s = np.cos(q), np.sin(q)
        Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        R = _rpy_to_R(*rpy) @ Rz
        Ti = np.eye(4); Ti[:3,:3] = R; Ti[:3,3] = xyz
        T = T @ Ti
    return T[:3, 3].astype(np.float32)


# ── Observation builder ────────────────────────────────────────────────────────

class ObservationBuilder:
    def __init__(self, encoder: FrozenResNet18Encoder, device: str = "cpu"):
        self._enc     = encoder
        self._device  = device
        self._def_rad = DEFAULT_JOINT_POS_RAD.copy()
        self._last_a  = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def reset(self):
        self._last_a = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def update_last_action(self, action: np.ndarray):
        self._last_a = action.astype(np.float32)

    def build(
        self,
        joint_pos_deg: np.ndarray,        # (6,) Isaac Lab degrees (from read_robot_state)
        joint_vel_deg_s: np.ndarray,      # (6,) Isaac Lab degrees/s
        bowl_pos: np.ndarray,             # (3,) m, robot frame
        bgr_frame: np.ndarray,
    ) -> torch.Tensor:
        """Returns (1, 536) obs tensor."""
        joint_pos_rad = np.deg2rad(joint_pos_deg)
        joint_vel_rad = np.deg2rad(joint_vel_deg_s)

        joint_pos_rel = (joint_pos_rad - self._def_rad).astype(np.float32)
        joint_vel_rel = joint_vel_rad.astype(np.float32)
        gripper_pos   = compute_gripper_link_position(joint_pos_rad)
        bowl_off      = bowl_pos.copy().astype(np.float32)
        bowl_off[2]  += BOWL_HOVER_HEIGHT

        cam_feat = self._enc.encode_bgr(bgr_frame)  # (1, 512)

        obs_np = np.concatenate([
            joint_pos_rel,                          # [0:6]
            joint_vel_rel,                          # [6:12]
            gripper_pos,                            # [12:15]
            bowl_off,                               # [15:18]
            np.zeros(RESNET_DIM, np.float32),       # [18:530] placeholder
            self._last_a,                           # [530:536]
        ])
        assert obs_np.shape == (NUM_OBS,)

        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self._device)  # (1, 536)
        obs[0, 18:18 + RESNET_DIM] = cam_feat[0]
        return obs


# ── Action interpreter ─────────────────────────────────────────────────────────

class ActionInterpreter:
    def __init__(self, max_delta_deg: float = MAX_DELTA_DEG):
        self._max_delta = max_delta_deg

    def interpret(self, raw_action: np.ndarray, current_joint_pos_deg: np.ndarray) -> dict:
        """
        raw_action          (6,) – policy output
        current_joint_pos_deg (6,) – Isaac Lab degrees (from read_robot_state)

        Returns dict with arm_targets_deg (5,) in Isaac Lab degrees,
        gripper_cmd_rad, gripper_open.
        """
        # Isaac Lab: target = default + scale * raw_action
        arm_target_rad = DEFAULT_JOINT_POS_RAD[:5] + ARM_ACTION_SCALE * raw_action[:5]
        arm_target_deg = np.rad2deg(arm_target_rad)

        # Safety: clamp delta in Isaac Lab degrees (consistent frame)
        current_arm_deg = current_joint_pos_deg[:5]
        delta = np.clip(arm_target_deg - current_arm_deg, -self._max_delta, self._max_delta)
        arm_targets_deg = current_arm_deg + delta

        gripper_open    = float(raw_action[5]) > GRIPPER_THRESHOLD
        gripper_cmd_rad = GRIPPER_OPEN_CMD_RAD if gripper_open else GRIPPER_CLOSE_CMD_RAD

        return {
            "arm_targets_deg": arm_targets_deg,
            "gripper_cmd_rad": gripper_cmd_rad,
            "gripper_open":    gripper_open,
        }


# ── Robot I/O ──────────────────────────────────────────────────────────────────

def read_robot_state(robot) -> tuple[np.ndarray, np.ndarray]:
    """Read joint state and convert to Isaac Lab degrees.

    Returns (joint_pos_deg (6,), zeros (6,)).
    joint_pos_deg is in Isaac Lab's coordinate frame so that
    ObservationBuilder and ActionInterpreter both work in a consistent frame.

    Frame conversion formula (inverse in send_action_to_robot):
      isaac_rad = DEFAULT_JOINT_POS_RAD + JOINT_SIGN * deg2rad(lerobot_deg - LEROBOT_DEFAULT_DEG)
    """
    obs = robot.get_observation()

    arm_lerobot_deg = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINT_NAMES], dtype=np.float64)

    arm_isaac_rad = DEFAULT_JOINT_POS_RAD[:5] + JOINT_SIGN * np.deg2rad(arm_lerobot_deg - LEROBOT_DEFAULT_DEG)
    arm_isaac_deg = np.rad2deg(arm_isaac_rad)

    # Gripper: 0-100 % → rad → equivalent Isaac Lab degrees
    gripper_pct = float(obs["gripper.pos"])
    gripper_rad = GRIPPER_RAD_MIN + (gripper_pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)
    gripper_deg = np.rad2deg(gripper_rad)

    return np.append(arm_isaac_deg, gripper_deg), np.zeros(6, dtype=np.float64)


def send_action_to_robot(robot, arm_targets_deg: np.ndarray, gripper_cmd_rad: float):
    """Convert Isaac Lab degree targets back to LeRobot units and send.

    Inverse of read_robot_state:
      lerobot_deg = LEROBOT_DEFAULT_DEG + JOINT_SIGN * rad2deg(isaac_rad - DEFAULT_JOINT_POS_RAD)
    """
    delta_rad = np.deg2rad(arm_targets_deg) - DEFAULT_JOINT_POS_RAD[:5]
    lerobot_targets = LEROBOT_DEFAULT_DEG + JOINT_SIGN * np.rad2deg(delta_rad)

    gripper_span = GRIPPER_RAD_MAX - GRIPPER_RAD_MIN
    gripper_pct  = float(np.clip((gripper_cmd_rad - GRIPPER_RAD_MIN) / gripper_span * 100.0, 0.0, 100.0))

    action = {f"{m}.pos": float(d) for m, d in zip(ARM_JOINT_NAMES, lerobot_targets)}
    action["gripper.pos"] = gripper_pct
    robot.send_action(action)


# ── Episode loop ───────────────────────────────────────────────────────────────

def run_episode(
    robot,
    camera: WristCamera,
    policy,
    obs_builder: ObservationBuilder,
    action_interp: ActionInterpreter,
    bowl_pos: np.ndarray,
    episode_duration_s: float,
    device: str,
    save_camera_frames: bool = False,
    save_dir: Path = Path("camera_debug"),
) -> None:
    dt    = 1.0 / CONTROL_HZ
    t_end = time.perf_counter() + episode_duration_s
    prev_pos = None
    obs_builder.reset()

    if save_camera_frames:
        save_dir.mkdir(exist_ok=True)

    log.info(f"Episode started ({episode_duration_s:.0f}s @ {CONTROL_HZ} Hz)")
    step = 0

    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        joint_pos_deg, _ = read_robot_state(robot)

        vel_deg_s = (joint_pos_deg - prev_pos) / dt if prev_pos is not None else np.zeros(6)
        prev_pos  = joint_pos_deg.copy()

        bgr = camera.read_bgr()
        if save_camera_frames and step % CONTROL_HZ == 0:
            cv2.imwrite(str(save_dir / f"step_{step:05d}.jpg"), bgr)

        obs = obs_builder.build(joint_pos_deg, vel_deg_s, bowl_pos, bgr)

        with torch.no_grad():
            raw_action = policy(obs).squeeze(0).cpu().numpy()  # (6,)

        result      = action_interp.interpret(raw_action, joint_pos_deg)
        arm_targets = result["arm_targets_deg"]
        gripper_cmd = result["gripper_cmd_rad"]

        send_action_to_robot(robot, arm_targets, gripper_cmd)
        obs_builder.update_last_action(raw_action)

        step += 1
        if step % CONTROL_HZ == 0:
            log.info(
                f"  t={step/CONTROL_HZ:.1f}s | "
                f"arm={joint_pos_deg[:5].round(1)} | "
                f"tgt={arm_targets.round(1)} | "
                f"gripper={'OPEN' if result['gripper_open'] else 'CLOSE'}"
            )

        busy_wait(max(0.0, dt - (time.perf_counter() - t0)))

    log.info("Episode finished.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Deploy RSL-RL LiftCamera policy on SO-101.")
    parser.add_argument("--checkpoint",       required=True,  help="Path to exported policy.pt (JIT)")
    parser.add_argument("--robot_port",       default="/dev/ttyACM0")
    parser.add_argument("--robot_id",         default="follower_arm")
    parser.add_argument("--robot_type",       default="so101_follower",
                        choices=["so101_follower", "so100_follower"])
    parser.add_argument("--camera_type",      default="usb", choices=["usb", "realsense"])
    parser.add_argument("--camera_device",    type=int, default=0)
    parser.add_argument("--bowl_pos",         nargs=3, type=float, default=[0.30, 0.10, 0.00],
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--num_episodes",     type=int,   default=5)
    parser.add_argument("--episode_duration", type=float, default=20.0,
                        help="Episode length in seconds (training used 5.0 s)")
    parser.add_argument("--reset_duration",   type=float, default=15.0)
    parser.add_argument("--max_delta_deg",    type=float, default=MAX_DELTA_DEG,
                        help="Max joint change per step in degrees (use 1.0 for first test)")
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mock",             action="store_true",
                        help="Mock mode: no real robot (tests policy inference only)")
    parser.add_argument("--save_camera_frames", action="store_true")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SO-101 LiftCamera Policy Deployment")
    log.info("=" * 60)
    log.info(f"Checkpoint : {args.checkpoint}")
    log.info(f"Device     : {args.device}")
    log.info(f"Bowl pos   : {args.bowl_pos} m")
    log.info(f"Camera     : {args.camera_type} (device={args.camera_device})")
    log.info(f"Hz         : {CONTROL_HZ}")
    log.info(f"Max delta  : {args.max_delta_deg}°/step")
    if np.all(LEROBOT_DEFAULT_DEG == 0.0):
        log.warning("LEROBOT_DEFAULT_DEG is all zeros — run check_joints.py first!")

    # ── Load policy ───────────────────────────────────────────────────────────
    policy = load_policy(args.checkpoint, args.device)

    # ── ResNet18 encoder ──────────────────────────────────────────────────────
    encoder = FrozenResNet18Encoder().to(args.device)
    encoder.eval()

    # ── Obs / action helpers ──────────────────────────────────────────────────
    obs_builder  = ObservationBuilder(encoder=encoder, device=args.device)
    action_interp = ActionInterpreter(max_delta_deg=args.max_delta_deg)

    # ── Camera ────────────────────────────────────────────────────────────────
    camera = WristCamera(camera_type=args.camera_type, device_id=args.camera_device)
    camera.connect()

    # ── Robot ─────────────────────────────────────────────────────────────────
    if args.mock:
        robot = _MockRobot()
        log.warning("MOCK MODE — no real robot will move.")
    else:
        robot_cls = SO101Follower if args.robot_type == "so101_follower" else SO100Follower
        robot_cfg = SOFollowerRobotConfig(
            port=args.robot_port,
            id=args.robot_id,
            max_relative_target=100.0,  # arm deltas already capped by MAX_DELTA_DEG; gripper needs full range to snap open/close
        )
        robot = robot_cls(robot_cfg)
        log.warning("Connecting robot in 3 s — hold the arm in its current position!")
        for i in (3, 2, 1):
            log.warning(f"  Connecting in {i}s ...")
            time.sleep(1.0)

    robot.connect()
    log.info("Robot connected.")

    bowl_pos = np.array(args.bowl_pos, dtype=np.float32)

    try:
        for ep in range(args.num_episodes):
            log.info(f"\n{'─'*50}")
            log.info(f"Episode {ep + 1} / {args.num_episodes}")
            log.info(f"{'─'*50}")

            run_episode(
                robot=robot,
                camera=camera,
                policy=policy,
                obs_builder=obs_builder,
                action_interp=action_interp,
                bowl_pos=bowl_pos,
                episode_duration_s=args.episode_duration,
                device=args.device,
                save_camera_frames=args.save_camera_frames,
            )

            if ep < args.num_episodes - 1:
                log.info(f"\nReset — {args.reset_duration:.0f}s pause. Return arm to start position.")
                time.sleep(args.reset_duration)

    except KeyboardInterrupt:
        log.info("\nAborted by user.")
    finally:
        camera.disconnect()
        robot.disconnect()
        log.info("Robot and camera disconnected.")


if __name__ == "__main__":
    main()
