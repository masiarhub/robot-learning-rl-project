"""
deploy_so101_lift_camera.py
============================
Deployt die RSL-RL PPO-Policy (trainiert in Isaac-SO-ARM101-LiftCamera)
auf den echten SO101-Roboter via LeRobot.

Policy-Details (aus LiftCameraEnvCfg / LiftCameraPPORunnerCfg):
  - Architektur       : MLP ActorCritic [256, 128, 64], Aktivierung ELU
  - Normalisierung    : empirical_normalization=True → Stats werden aus dem
                        Checkpoint geladen und auf den Obs-Vektor angewendet
  - Policy-Observation: 536-dimensionaler Vektor
      [0:6]     joint_pos_rel          – Gelenk-Positionen relativ zu Default (Rad)
      [6:12]    joint_vel_rel          – Gelenk-Geschwindigkeiten relativ zu Default (Rad/s)
      [12:15]   gripper_link_position  – Gripper-Link-Pos im Robot-Root-Frame (m)
      [15:18]   bowl_position          – Bowl-Pos + 0.12m z-Offset im Robot-Root-Frame (m)
      [18:530]  wrist_image            – ResNet18-Feature-Vektor (512D) aus Wrist-Kamera
      [530:536] last_action            – letzte Policy-Action (6D)
  - Critic-Observation (nur Training, nicht für Deployment, 24D):
      joint_pos_rel (6) + joint_vel_rel (6) + object_position (3) +
      bowl_position (3) + last_action (6) = 24D
  - Action            : 6-dimensional
      [0:5]   arm_action    – JointPosition-Targets, scale=0.5, use_default_offset=True
                              Joints: shoulder_pan, shoulder_lift, elbow_flex,
                                      wrist_flex, wrist_roll
      [5]     gripper       – Binary: > 0 → open (0.5 rad), ≤ 0 → close (-0.1 rad)

Default-Joint-Positions (aus SoArm101LiftCameraEnvCfg):
  shoulder_pan=0.0, shoulder_lift=-0.6, elbow_flex=-0.6,
  wrist_flex=1.57, wrist_roll=-1.57, gripper=0.0

Kamera:
  - USB-Kamera (--camera_type=usb, Standard) oder Intel RealSense (--camera_type=realsense)
  - Bild wird auf 72x128 (HxW) skaliert und via gefrorenen ResNet18 encodiert
  - ImageNet-Normalisierung wird vor dem Forward-Pass angewendet

Nutzung
-------
python deploy_so101_lift_camera.py \
    --checkpoint logs/rsl_rl/lift_camera/2026-05-01_12-55-26/model_2999.pt \
    --robot_port /dev/ttyACM0 \
    --robot_id my_so101 \
    --bowl_pos 0.30 0.10 0.00 \
    --num_episodes 5

Mit RealSense-Kamera:
python deploy_so101_lift_camera.py \
    --checkpoint ... \
    --camera_type realsense

WICHTIG VOR DEM ERSTEN RUN:
  1. Roboter kalibrieren (falls noch nicht geschehen):
     lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_so101
  2. Bowl-Position im Robot-Frame messen (mit Lineal oder Kamera).
  3. Wrist-Kamera am Gripper-Link montiert und ausgerichtet wie im Training:
     pos=(-0.0049, 0.0498, -0.0591), rot ≈ -35.31° um X-Achse
  4. Im ersten Run --max_delta_deg=1.0 verwenden und den Roboter beobachten!
  5. Mit --save_camera_frames kannst du Kamerabilder speichern um die Kameraausrichtung zu prüfen.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import cv2

# ─── LeRobot (0.5+) ─────────────────────────────────────────────────────────
from lerobot.robots.so_follower import SO100Follower, SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


def busy_wait(dt: float) -> None:
    """High-precision spin-wait (replaces lerobot.common.robot_devices.utils.busy_wait)."""
    t_end = time.perf_counter() + dt
    while time.perf_counter() < t_end:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Konstanten aus der Trainings-Konfiguration                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# SO101 Joints in der Reihenfolge wie Isaac Lab sie verwaltet
# JointPositionActionCfg: ["shoulder_.*", "elbow_flex", "wrist_.*"]
# → shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (5 Arm-Joints)
# + gripper (1 Joint, BinaryJointPositionAction)
ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]  # 6 Joints total
NUM_ARM_JOINTS = 5
NUM_ACTIONS = 6  # 5 Arm + 1 Gripper

# Default-Positionen (in Rad) – aus SoArm101LiftCameraEnvCfg.init_state
# ACHTUNG: Diese sind NICHT null wie im alten PickPlace-Modell!
DEFAULT_JOINT_POS_RAD = np.array([
     0.00,   # shoulder_pan
    -0.60,   # shoulder_lift
    -0.60,   # elbow_flex
     1.57,   # wrist_flex
    -1.57,   # wrist_roll
     0.00,   # gripper
], dtype=np.float64)

# Kontrollfrequenz: sim.dt=0.01s, decimation=2 → 50 Hz
CONTROL_HZ = 50

# Hover-Offset wie in der Trainings-Umgebung: BOWL_HOVER_HEIGHT = 0.12 m
BOWL_HOVER_HEIGHT = 0.12

# Gripper-Kommandos (aus BinaryJointPositionActionCfg)
GRIPPER_OPEN_CMD_RAD  =  0.5    # open_command_expr
GRIPPER_CLOSE_CMD_RAD = -0.1    # close_command_expr
GRIPPER_THRESHOLD     =  0.0    # Policy-Rohwert > 0 → open

# Arm-Action-Skalierung aus JointPositionActionCfg: scale=0.5
ARM_ACTION_SCALE = 0.5

# Safety: maximale Gelenk-Änderung pro Step in Grad
MAX_DELTA_DEG = 3.0

# ── Joint mapping: LeRobot ↔ Isaac Lab ────────────────────────────────────────
#
# Sign correction: +1 where both conventions share the same positive direction.
# wrist_flex confirmed flipped: LeRobot -95° ↔ Isaac +90°.
# elbow_flex confirmed flipped from earlier measurement.
# Adjust any remaining joints if check_joints.py shows unexpected sign after homing.
JOINT_SIGN = np.array([1.0, 1.0, -1.0, -1.0, 1.0])  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
#
# Home reference: what LeRobot reads (degrees) when the arm is physically at Isaac Lab's
# default joint positions (DEFAULT_JOINT_POS_RAD).
# HOW TO CALIBRATE: move arm to the default pose, run check_joints.py, copy read_deg column.
LEROBOT_DEFAULT_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # ← fill in after homing measurement

# Gripper 0-100 % ↔ rad linear mapping (new LeRobot API uses RANGE_0_100 for gripper)
# 0 % → GRIPPER_CLOSE_CMD_RAD, 100 % → GRIPPER_OPEN_CMD_RAD
GRIPPER_RAD_MIN = GRIPPER_CLOSE_CMD_RAD   # -0.1 rad
GRIPPER_RAD_MAX = GRIPPER_OPEN_CMD_RAD    #  0.5 rad

# Kamera-Auflösung wie im Training (TiledCameraCfg: height=72, width=128)
CAM_H = 72
CAM_W = 128

# ResNet18 Output-Dimension
RESNET_DIM = 512

# Observation-Dimensionen
OBS_JOINT_POS    = 6
OBS_JOINT_VEL    = 6
OBS_GRIPPER_LINK = 3
OBS_BOWL_POS     = 3
OBS_WRIST_IMAGE  = RESNET_DIM   # 512
OBS_LAST_ACTION  = NUM_ACTIONS  # 6
NUM_OBS = OBS_JOINT_POS + OBS_JOINT_VEL + OBS_GRIPPER_LINK + OBS_BOWL_POS + OBS_WRIST_IMAGE + OBS_LAST_ACTION
# = 6 + 6 + 3 + 3 + 512 + 6 = 536

assert NUM_OBS == 536, f"Unerwartete OBS-Dimension: {NUM_OBS}"

# Critic obs (privileged, training only): joint_pos(6) + joint_vel(6) + object_pos(3) + bowl_pos(3) + last_action(6)
NUM_CRITIC_OBS = 24


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Standalone Actor (kein rsl-rl-lib erforderlich)                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StandaloneActor(nn.Module):
    """
    Minimaler MLP-Actor, der exakt der rsl_rl-ActorCritic-Architektur entspricht.
    Vermeidet die Abhängigkeit von rsl-rl-lib (PyPI-Version ≠ Isaac-Lab-Fork).

    State-Dict-Struktur im Checkpoint (rsl_rl ActorCritic mit actor_obs_normalization=True):
      actor.0.weight / actor.0.bias              – Linear(obs → hidden[0])
      actor.2.weight / actor.2.bias              – Linear(hidden[0] → hidden[1])
      actor.4.weight / actor.4.bias              – Linear(hidden[1] → hidden[2])
      actor.6.weight / actor.6.bias              – Linear(hidden[2] → actions)
      actor_obs_normalizer._mean  shape (1, obs) – running mean
      actor_obs_normalizer._std   shape (1, obs) – running std
      actor_obs_normalizer._var   shape (1, obs) – running var (unused at inference)
      actor_obs_normalizer.count  shape ()        – sample count

    Normalisierungsformel (EmpiricalNormalization.forward):
      obs_norm = (obs - _mean) / (_std + eps)  where eps = 1e-2
    """

    def __init__(self, num_obs: int, num_actions: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = num_obs
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*layers)

        # Puffer für empirischen Obs-Normalizer (EmpiricalNormalization aus rsl_rl)
        self.register_buffer("obs_mean", torch.zeros(num_obs))
        self.register_buffer("obs_std", torch.ones(num_obs))
        self._use_normalizer = False

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        if self._use_normalizer:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-2)  # matches EmpiricalNormalization.forward
        return self.actor(obs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ResNet18 Encoder (gefroren, wie im Training)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FrozenResNet18Encoder(nn.Module):
    """
    Gefrorer ResNet18-Encoder wie in wrist_camera_image() der Trainings-Umgebung.

    Input : (B, 3, H, W) ImageNet-normalisierter Float-Tensor
    Output: (B, 512) Feature-Vektor

    Architektur: conv1 → bn1 → relu → maxpool → layer1..4 → AdaptiveAvgPool2d(1,1) → flatten
    """

    _imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        FrozenResNet18Encoder._imagenet_mean = FrozenResNet18Encoder._imagenet_mean.to(device)
        FrozenResNet18Encoder._imagenet_std  = FrozenResNet18Encoder._imagenet_std.to(device)
        return self

    @torch.no_grad()
    def encode_bgr(self, bgr_frame: np.ndarray) -> torch.Tensor:
        """
        Wandelt ein OpenCV BGR-Bild (H, W, 3 uint8) in einen (1, 512) Feature-Tensor um.

        Schritte:
          1. BGR → RGB
          2. Resize auf (CAM_H, CAM_W) = (72, 128) — wie im Training
          3. uint8 → float [0,1]
          4. ImageNet-Normalisierung
          5. ResNet18 Forward-Pass
        """
        # BGR → RGB und resize
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CAM_W, CAM_H), interpolation=cv2.INTER_LINEAR)

        # numpy → tensor (1, 3, H, W)
        img = torch.from_numpy(rgb).float() / 255.0          # [H, W, 3]
        img = img.permute(2, 0, 1).unsqueeze(0)              # [1, 3, H, W]
        device = next(self.encoder.parameters()).device
        img = img.to(device)

        # ImageNet-Normalisierung
        img = (img - FrozenResNet18Encoder._imagenet_mean) / FrozenResNet18Encoder._imagenet_std

        # Forward
        features = self.encoder(img)          # [1, 512, 1, 1]
        return features.flatten(start_dim=1)  # [1, 512]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Kamera-Abstraktion                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WristCamera:
    """
    Abstraktion für die Wrist-Kamera.

    Unterstützt:
      - USB-Kamera via OpenCV (--camera_type=usb)
      - Intel RealSense (--camera_type=realsense)
    """

    def __init__(self, camera_type: str = "usb", device_id: int = 0):
        self._type = camera_type
        self._device_id = device_id
        self._cap = None
        self._pipeline = None

    def connect(self):
        if self._type == "usb":
            self._cap = cv2.VideoCapture(self._device_id)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Kamera {self._device_id} konnte nicht geöffnet werden. "
                    f"Verfügbare Geräte mit: ls /dev/video*"
                )
            log.info(f"USB-Kamera {self._device_id} verbunden")

        elif self._type == "realsense":
            try:
                import pyrealsense2 as rs
            except ImportError:
                raise ImportError("pyrealsense2 nicht installiert: pip install pyrealsense2")
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, CAM_W, CAM_H, rs.format.bgr8, 30)
            self._pipeline.start(config)
            log.info("Intel RealSense verbunden")

        else:
            raise ValueError(f"Unbekannter Kamera-Typ: {self._type}. Wähle 'usb' oder 'realsense'.")

    def read_bgr(self) -> np.ndarray:
        """Gibt das aktuelle Kamerabild als (H, W, 3) BGR uint8 Array zurück."""
        if self._type == "usb":
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError("Kamerabild konnte nicht gelesen werden.")
            return frame

        elif self._type == "realsense":
            import pyrealsense2 as rs
            frames = self._pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("RealSense: Kein Color-Frame verfügbar.")
            return np.asanyarray(color_frame.get_data())

    def disconnect(self):
        if self._cap is not None:
            self._cap.release()
        if self._pipeline is not None:
            self._pipeline.stop()
        log.info("Kamera getrennt")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Policy laden (RSL-RL Checkpoint mit empirical_normalization=True)      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_rsl_rl_policy(checkpoint_path: str, device: str) -> StandaloneActor:
    """
    Lädt die Policy aus einem RSL-RL Checkpoint (.pt Datei) in einen StandaloneActor.

    Erwartet Checkpoint-Struktur:
      { 'model_state_dict': { 'actor.X.weight': ..., 'actor_obs_normalizer._mean': ..., ... }, 'iter': int }

    Beim Deployment wird nur der Actor (act_inference) benötigt.
    """
    log.info(f"Lade RSL-RL Checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    actor = StandaloneActor(
        num_obs=NUM_OBS,
        num_actions=NUM_ACTIONS,
        hidden_dims=[256, 128, 64],
    ).to(device)

    # Actor-Gewichte laden (Keys im rsl_rl ActorCritic: "actor.0.weight", ...)
    actor_sd = {k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")}
    missing, unexpected = actor.actor.load_state_dict(actor_sd, strict=True)
    log.info("Actor-Gewichte geladen.")
    if missing:
        log.warning(f"  Fehlende Keys: {missing}")
    if unexpected:
        log.warning(f"  Unerwartete Keys: {unexpected}")

    # Empirischen Obs-Normalizer laden (rsl_rl EmpiricalNormalization buffers: _mean, _std, shape (1, obs))
    mean_key = "actor_obs_normalizer._mean"
    std_key  = "actor_obs_normalizer._std"
    if mean_key in state_dict and std_key in state_dict:
        actor.obs_mean.copy_(state_dict[mean_key].squeeze(0))
        actor.obs_std.copy_(state_dict[std_key].squeeze(0))
        actor._use_normalizer = True
        log.info("  Obs-Normalizer geladen (actor_obs_normalizer._mean / ._std)")
    else:
        found = [k for k in state_dict if "normaliz" in k.lower()]
        if found:
            log.warning(f"  Unbekannte Normalizer-Keys: {found[:5]} – Normalisierung deaktiviert")
        else:
            log.warning("  Keine Normalizer-Keys gefunden – Normalisierung deaktiviert")

    iteration = checkpoint.get("iter", "?")
    log.info(f"Policy geladen (Iteration {iteration})")
    log.info(f"  Actor-Obs-Dim : {NUM_OBS}")
    log.info(f"  Action-Dim    : {NUM_ACTIONS}")
    log.info(f"  Architektur   : [256, 128, 64], ELU")
    log.info(f"  Normalizer    : {'aktiv' if actor._use_normalizer else 'inaktiv'}")

    actor.eval()
    return actor


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Gripper-Link-Position schätzen (Forward Kinematics Approximation)      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _joint_transform(xyz: tuple, rpy: tuple, q: float) -> np.ndarray:
    """Homogeneous transform for a revolute z-axis joint at pose (xyz, rpy) rotated by q."""
    c, s = np.cos(q), np.sin(q)
    Rz_q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = _rpy_to_matrix(*rpy) @ Rz_q
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# Exact joint origins extracted from trs_so101/urdf/so_arm101.urdf.
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
# Each entry is (xyz, rpy) of the joint origin in the parent link frame.
_SO101_JOINTS = [
    ((0.0388353, 0.0,       0.0624),  (np.pi,    0.0,          -np.pi)),
    ((-0.0303992, -0.0182778, -0.0542), (-np.pi/2, -np.pi/2,   0.0)),
    ((-0.11257,  -0.028,     0.0),    (0.0,      0.0,           np.pi/2)),
    ((-0.1349,    0.0052,    0.0),    (0.0,      0.0,          -np.pi/2)),
    ((0.0,       -0.0611,    0.0181), (np.pi/2,  0.0486795,    np.pi)),
]


def compute_gripper_link_position(joint_pos_rad: np.ndarray) -> np.ndarray:
    """Exact FK for gripper_link origin in robot root (base_link) frame.

    Replicates Isaac Lab's body_pos_w lookup by chaining the URDF joint transforms.
    All five revolute joints rotate around their local z-axis.
    """
    T = np.eye(4)
    for (xyz, rpy), q in zip(_SO101_JOINTS, joint_pos_rad[:5]):
        T = T @ _joint_transform(xyz, rpy, float(q))
    return T[:3, 3].astype(np.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Observation-Builder                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ObservationBuilder:
    """
    Baut den 536-dimensionalen Observation-Vektor für die Policy.

    Observation-Struktur (aus LiftCameraEnvCfg → ObservationsCfg.PolicyCfg):
      [0:6]     joint_pos_rel          = aktuelle Gelenkpos - default_joint_pos (Rad)
      [6:12]    joint_vel_rel          = aktuelle Gelenkvel - 0 = aktuelle Vel (Rad/s)
      [12:15]   gripper_link_position  = Gripper-Link-Pos im Robot-Root-Frame (m)
      [15:18]   bowl_position          = Bowl-Pos + 0.12m z-Offset im Robot-Root-Frame (m)
      [18:530]  wrist_image            = ResNet18-Feature (512D)
      [530:536] last_action            = letzte Policy-Action (6D)

    EINHEITEN:
      - Isaac Lab / RSL-RL: intern RADIANS
      - LeRobot / SO101: Gelenk-Positionen in GRAD → hier umgerechnet
    """

    def __init__(self, encoder: FrozenResNet18Encoder, device: str = "cpu"):
        self.device = device
        self._encoder = encoder
        self._default_pos_rad = DEFAULT_JOINT_POS_RAD.copy()
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    def build(
        self,
        joint_pos_deg: np.ndarray,        # (6,) aktuelle Gelenkpos in Grad
        joint_vel_deg_s: np.ndarray,      # (6,) aktuelle Gelenkvel in Grad/s
        bowl_pos_robot_frame: np.ndarray, # (3,) Bowl-Pos in Meter, Robot-Frame
        bgr_frame: np.ndarray,            # (H, W, 3) Kamerabild
    ) -> torch.Tensor:
        """Gibt einen (1, 536) Tensor zurück für die Policy."""

        # ── Grad → Radians ────────────────────────────────────────────────
        joint_pos_rad = np.deg2rad(joint_pos_deg)
        joint_vel_rad = np.deg2rad(joint_vel_deg_s)

        # ── joint_pos_rel : aktuelle Pos - default Pos ────────────────────
        joint_pos_rel = joint_pos_rad - self._default_pos_rad

        # ── joint_vel_rel : default_vel = 0 → einfach aktuelle Vel ────────
        joint_vel_rel = joint_vel_rad

        # ── gripper_link_position ─────────────────────────────────────────
        gripper_pos = compute_gripper_link_position(joint_pos_rad)

        # ── bowl_position : Bowl-Pos + BOWL_HOVER_HEIGHT als z-Offset ─────
        bowl_with_offset = bowl_pos_robot_frame.copy().astype(np.float32)
        bowl_with_offset[2] += BOWL_HOVER_HEIGHT

        # ── Kamera-Feature via ResNet18 ────────────────────────────────────
        camera_feature = self._encoder.encode_bgr(bgr_frame)  # (1, 512) Tensor

        # ── Observation zusammensetzen ─────────────────────────────────────
        obs_np = np.concatenate([
            joint_pos_rel.astype(np.float32),    # [0:6]
            joint_vel_rel.astype(np.float32),    # [6:12]
            gripper_pos.astype(np.float32),      # [12:15]
            bowl_with_offset,                    # [15:18]
            np.zeros(RESNET_DIM, dtype=np.float32),  # [18:530] Platzhalter, wird unten ersetzt
            self._last_action,                   # [530:536]
        ])  # shape: (536,)

        assert obs_np.shape == (NUM_OBS,), f"Obs shape falsch: {obs_np.shape}"

        # Obs als Tensor und Camera-Feature einsetzen
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)  # (1, 536)
        obs_tensor[0, 18:18 + RESNET_DIM] = camera_feature[0]

        return obs_tensor  # (1, 536)

    def update_last_action(self, action: np.ndarray) -> None:
        """Speichert die letzte Action für den nächsten Obs-Vektor."""
        self._last_action = action.astype(np.float32)

    def reset(self) -> None:
        """Setzt last_action zurück (zu Beginn jeder Episode)."""
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Action-Interpreter                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ActionInterpreter:
    """
    Wandelt den 6-dimensionalen Policy-Output in Roboter-Kommandos um.

    Isaac Lab Action-Mapping (aus SoArm101LiftCameraEnvCfg):
      arm_action (JointPositionActionCfg):
        - joint_names: ["shoulder_.*", "elbow_flex", "wrist_.*"]
          → shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (5 Joints)
        - scale=0.5, use_default_offset=True
        - Ziel = default_pos_rad[0:5] + 0.5 * raw_action[0:5]

      gripper_action (BinaryJointPositionActionCfg):
        - raw_action[5] > 0 → open (0.5 rad), sonst close (-0.1 rad)
    """

    def __init__(
        self,
        max_delta_deg: float = MAX_DELTA_DEG,
    ):
        self._max_delta_deg = max_delta_deg

    def interpret(
        self,
        raw_action: np.ndarray,             # (6,) Policy-Output
        current_joint_pos_deg: np.ndarray,  # (6,) aktuelle Positionen in Grad
    ) -> dict:
        """
        Returns:
          {
            "arm_targets_deg": np.ndarray (5,) – Ziel-Gelenk-Positionen in Grad
            "gripper_cmd_rad": float            – Gripper-Kommando in Rad
            "gripper_open":    bool             – True=open, False=close
          }
        """
        # ── Arm: JointPosition mit default_offset ─────────────────────────
        # Ziel_rad = default_pos_rad[:5] + 0.5 * raw_action[:5]
        arm_action_raw = raw_action[:NUM_ARM_JOINTS]
        arm_target_rad = DEFAULT_JOINT_POS_RAD[:NUM_ARM_JOINTS] + ARM_ACTION_SCALE * arm_action_raw
        arm_target_deg = np.rad2deg(arm_target_rad)

        # Safety-Clipping: maximale Delta-Bewegung pro Step
        current_arm_deg = current_joint_pos_deg[:NUM_ARM_JOINTS]
        delta = arm_target_deg - current_arm_deg
        delta_clipped = np.clip(delta, -self._max_delta_deg, self._max_delta_deg)
        if not np.allclose(delta, delta_clipped, atol=0.01):
            log.debug(
                f"Arm-Delta geclippt: max={np.abs(delta).max():.2f}° → {np.abs(delta_clipped).max():.2f}°"
            )
        arm_targets_deg = current_arm_deg + delta_clipped

        # ── Gripper: Binary ────────────────────────────────────────────────
        gripper_raw = float(raw_action[NUM_ARM_JOINTS])  # index 5
        gripper_open = gripper_raw > GRIPPER_THRESHOLD
        gripper_cmd_rad = GRIPPER_OPEN_CMD_RAD if gripper_open else GRIPPER_CLOSE_CMD_RAD

        return {
            "arm_targets_deg": arm_targets_deg,
            "gripper_cmd_rad": gripper_cmd_rad,
            "gripper_open": gripper_open,
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Roboter-Hilfsfunktionen                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def read_robot_state(robot) -> tuple[np.ndarray, np.ndarray]:
    """
    Liest Gelenk-Positionen vom Roboter (neue LeRobot-API).

    get_observation() gibt ein Dict zurück:
      {"shoulder_pan.pos": deg, "shoulder_lift.pos": deg, ..., "gripper.pos": 0-100}

    Gripper (RANGE_0_100) wird linear in Radiant umgerechnet und als Äquivalent-Grad
    zurückgegeben, damit ObservationBuilder.build() deg2rad korrekt anwenden kann.

    Returns: (joint_pos_deg (6,), dummy_vel (6,))  — Velocity numerisch im Loop
    """
    obs = robot.get_observation()

    arm_pos_deg = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINT_NAMES], dtype=np.float64)

    # Convert LeRobot degrees → Isaac Lab radians:
    #   1. subtract the LeRobot home reference (aligns calibration zero with URDF zero)
    #   2. apply sign correction (some joints have opposite positive direction)
    #   3. offset by Isaac Lab default position
    arm_pos_rad_isaac = DEFAULT_JOINT_POS_RAD[:5] + JOINT_SIGN * np.deg2rad(arm_pos_deg - LEROBOT_DEFAULT_DEG)
    arm_as_deg = np.rad2deg(arm_pos_rad_isaac)  # ObsBuilder will deg2rad these back

    # Gripper: 0-100 % → rad (linear) → "äquivalente Grad" für ObsBuilder
    gripper_pct = float(obs["gripper.pos"])
    gripper_rad = GRIPPER_RAD_MIN + (gripper_pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)
    gripper_as_deg = float(np.rad2deg(gripper_rad))

    # Return arm in Isaac Lab degrees so ObservationBuilder and ActionInterpreter
    # both work in a consistent frame. arm_pos_deg (raw LeRobot) must NOT be
    # returned here — for sign-flipped joints it lives in the wrong half-plane,
    # causing ~180° deltas even at rest that blow up send_action_to_robot.
    joint_pos_deg = np.append(arm_as_deg, gripper_as_deg)
    return joint_pos_deg, np.zeros(6, dtype=np.float64)


def send_action_to_robot(robot, arm_targets_deg: np.ndarray, gripper_cmd_rad: float):
    """
    Sendet Arm-Targets (in Grad) und Gripper-Kommando an den Roboter (neue LeRobot-API).

    send_action() erwartet ein Dict: {"joint.pos": value, ...}
      - Arm-Joints in Grad
      - Gripper in 0-100 % (RANGE_0_100-Modus)
    """
    gripper_span = GRIPPER_RAD_MAX - GRIPPER_RAD_MIN
    gripper_pct = float(np.clip((gripper_cmd_rad - GRIPPER_RAD_MIN) / gripper_span * 100.0, 0.0, 100.0))

    # Convert Isaac Lab target degrees → LeRobot degrees (inverse of read_robot_state):
    #   1. compute delta from Isaac Lab default
    #   2. apply sign correction (inverse: same sign flips the delta back)
    #   3. add LeRobot home reference
    delta_rad = np.deg2rad(arm_targets_deg) - DEFAULT_JOINT_POS_RAD[:5]
    lerobot_targets_deg = LEROBOT_DEFAULT_DEG + JOINT_SIGN * np.rad2deg(delta_rad)

    action: dict[str, float] = {f"{m}.pos": float(d) for m, d in zip(ARM_JOINT_NAMES, lerobot_targets_deg)}
    action["gripper.pos"] = gripper_pct
    robot.send_action(action)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Haupt-Deployment-Loop                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_episode(
    robot,
    camera: WristCamera,
    policy_nn: torch.nn.Module,
    obs_builder: ObservationBuilder,
    action_interp: ActionInterpreter,
    bowl_pos: np.ndarray,
    episode_duration_s: float,
    device: str,
    save_camera_frames: bool = False,
    save_dir: Path = Path("camera_debug"),
) -> None:
    """
    Führt eine einzelne Lift-and-Place-Episode aus.

    bowl_pos muss im Robot-Root-Frame sein (in Metern).
    """
    dt = 1.0 / CONTROL_HZ
    t_end = time.perf_counter() + episode_duration_s

    prev_joint_pos_deg = None
    obs_builder.reset()

    if save_camera_frames:
        save_dir.mkdir(exist_ok=True)

    log.info(f"Episode gestartet ({episode_duration_s:.0f}s @ {CONTROL_HZ}Hz)")
    log.info(f"  Bowl-Pos (Robot-Frame): {bowl_pos}")

    step = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        # ── Roboter-State lesen ────────────────────────────────────────────
        joint_pos_deg, _ = read_robot_state(robot)

        # Numerische Velocity-Schätzung (Grad/s)
        if prev_joint_pos_deg is not None:
            joint_vel_deg_s = (joint_pos_deg - prev_joint_pos_deg) / dt
        else:
            joint_vel_deg_s = np.zeros(6)
        prev_joint_pos_deg = joint_pos_deg.copy()

        # ── Kamerabild lesen ───────────────────────────────────────────────
        bgr_frame = camera.read_bgr()

        if save_camera_frames and step % CONTROL_HZ == 0:
            frame_path = save_dir / f"step_{step:05d}.jpg"
            cv2.imwrite(str(frame_path), bgr_frame)
            log.info(f"  Kamerabild gespeichert: {frame_path}")

        # ── Observation aufbauen ───────────────────────────────────────────
        obs = obs_builder.build(
            joint_pos_deg=joint_pos_deg,
            joint_vel_deg_s=joint_vel_deg_s,
            bowl_pos_robot_frame=bowl_pos,
            bgr_frame=bgr_frame,
        )

        # ── Policy-Inferenz ────────────────────────────────────────────────
        with torch.no_grad():
            raw_action = policy_nn.act_inference(obs)      # (1, 6)
            raw_action_np = raw_action.squeeze(0).cpu().numpy()  # (6,)

        # ── Action interpretieren ──────────────────────────────────────────
        result = action_interp.interpret(raw_action_np, joint_pos_deg)
        arm_targets = result["arm_targets_deg"]
        gripper_cmd = result["gripper_cmd_rad"]

        # ── Action senden ──────────────────────────────────────────────────
        send_action_to_robot(robot, arm_targets, gripper_cmd)

        # Last-action aktualisieren (Policy-Rohwert)
        obs_builder.update_last_action(raw_action_np)

        step += 1
        if step % CONTROL_HZ == 0:
            log.info(
                f"  t={step/CONTROL_HZ:.1f}s | "
                f"arm_pos={joint_pos_deg[:5].round(1)} | "
                f"arm_tgt={arm_targets.round(1)} | "
                f"gripper={'OPEN' if result['gripper_open'] else 'CLOSE'}"
            )

        # ── Timing ────────────────────────────────────────────────────────
        elapsed = time.perf_counter() - t0
        busy_wait(max(0.0, dt - elapsed))

    log.info("Episode beendet.")


class _MockSOFollower:
    """Minimaler Mock-Roboter für --mock-Modus (kein Hardware erforderlich)."""
    is_connected = True

    def connect(self, **kwargs): pass
    def disconnect(self): pass

    def get_observation(self) -> dict:
        obs = {f"{m}.pos": 0.0 for m in ARM_JOINT_NAMES}
        obs["gripper.pos"] = 50.0   # 50 % ≈ Mittelstellung
        return obs

    def send_action(self, action: dict) -> dict:
        return action


def main():
    parser = argparse.ArgumentParser(
        description="Deployt die RSL-RL LiftCamera-Policy auf den SO101."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Pfad zur .pt Checkpoint-Datei"
    )
    parser.add_argument(
        "--robot_port", default="/dev/ttyACM0",
    )
    parser.add_argument(
        "--robot_id", default="follower_arm",
    )
    parser.add_argument(
        "--robot_type", default="so101_follower",
        choices=["so101_follower", "so100_follower"],
    )
    parser.add_argument(
        "--camera_type", default="usb",
        choices=["usb", "realsense"],
        help="Kamera-Typ: 'usb' (OpenCV) oder 'realsense' (Intel RealSense)"
    )
    parser.add_argument(
        "--camera_device", type=int, default=0,
        help="USB-Kamera Device-ID (default: 0 → /dev/video0)"
    )
    parser.add_argument(
        "--bowl_pos", nargs=3, type=float, default=[0.30, 0.10, 0.00],
        metavar=("X", "Y", "Z"),
        help="Bowl-Position im Robot-Frame in Metern"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=5,
    )
    parser.add_argument(
        "--episode_duration", type=float, default=5.0,
        help="Episodendauer in Sekunden (Training: episode_length_s=5.0)"
    )
    parser.add_argument(
        "--reset_duration", type=float, default=15.0,
        help="Pause zwischen Episoden zum Zurücksetzen der Szene"
    )
    parser.add_argument(
        "--max_delta_deg", type=float, default=MAX_DELTA_DEG,
        help="Safety: max. Gelenkänderung pro Step in Grad (für ersten Test: 1.0)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock-Modus: kein echter Roboter (zum Testen der Policy-Inferenz)"
    )
    parser.add_argument(
        "--save_camera_frames", action="store_true",
        help="Kamerabilder jede Sekunde in ./camera_debug/ speichern (für Kamera-Debugging)"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SO101 LiftCamera Policy Deployment")
    log.info("=" * 60)
    log.info(f"Checkpoint  : {args.checkpoint}")
    log.info(f"Device      : {args.device}")
    log.info(f"Bowl-Pos    : {args.bowl_pos} m (Robot-Frame)")
    log.info(f"Kamera      : {args.camera_type} (device={args.camera_device})")
    log.info(f"Frequenz    : {CONTROL_HZ} Hz")
    log.info(f"Max-Delta   : {args.max_delta_deg}°/step")
    log.info(f"Obs-Dim     : {NUM_OBS} (6+6+3+3+512+6)")
    log.info(f"Action-Dim  : {NUM_ACTIONS}")
    log.info(f"Default-Pos : {np.rad2deg(DEFAULT_JOINT_POS_RAD).round(1)}°")

    # ── ResNet18 Encoder initialisieren ───────────────────────────────────
    log.info("Initialisiere ResNet18 Encoder (gefrorene ImageNet-Gewichte)...")
    encoder = FrozenResNet18Encoder().to(args.device)
    encoder.eval()
    log.info(f"  ResNet18 bereit auf {args.device}")

    # ── Policy laden ───────────────────────────────────────────────────────
    policy_nn = load_rsl_rl_policy(args.checkpoint, args.device)

    # ── Obs-Builder und Action-Interpreter ────────────────────────────────
    obs_builder = ObservationBuilder(encoder=encoder, device=args.device)
    action_interp = ActionInterpreter(max_delta_deg=args.max_delta_deg)

    # ── Kamera verbinden ───────────────────────────────────────────────────
    camera = WristCamera(camera_type=args.camera_type, device_id=args.camera_device)
    camera.connect()

    # ── Roboter verbinden ──────────────────────────────────────────────────
    if args.mock:
        robot = _MockSOFollower()
        log.warning("MOCK-MODUS aktiv – kein echter Roboter wird bewegt!")
    else:
        robot_cls = SO101Follower if args.robot_type == "so101_follower" else SO100Follower
        # max_relative_target: hardware-level cap per send_action() call (degrees).
        # Does NOT prevent the initial torque-enable snap — hold the arm during connect.
        robot_cfg = SOFollowerRobotConfig(
            port=args.robot_port,
            id=args.robot_id,
            max_relative_target=10.0,
        )
        robot = robot_cls(robot_cfg)
        log.warning("ACHTUNG: Roboter verbindet sich in 3 Sekunden.")
        log.warning("Halte den Arm in der aktuellen Position, bis er sich beruhigt hat!")
        for i in (3, 2, 1):
            log.warning(f"  Verbinde in {i}s ...")
            time.sleep(1.0)
    robot.connect()
    log.info(f"Roboter verbunden: {args.robot_type} @ {args.robot_port}")

    bowl_pos = np.array(args.bowl_pos, dtype=np.float32)

    try:
        for ep in range(args.num_episodes):
            log.info(f"\n{'─'*50}")
            log.info(f"Episode {ep + 1} / {args.num_episodes}")
            log.info(f"{'─'*50}")

            run_episode(
                robot=robot,
                camera=camera,
                policy_nn=policy_nn,
                obs_builder=obs_builder,
                action_interp=action_interp,
                bowl_pos=bowl_pos,
                episode_duration_s=args.episode_duration,
                device=args.device,
                save_camera_frames=args.save_camera_frames,
            )

            if ep < args.num_episodes - 1:
                log.info(f"\nSzene resetten – {args.reset_duration:.0f}s Pause ...")
                log.info("→ Bowl zurückstellen, Roboter in Startposition bringen.")
                time.sleep(args.reset_duration)

    except KeyboardInterrupt:
        log.info("\nDurch Benutzer abgebrochen.")
    finally:
        camera.disconnect()
        robot.disconnect()
        log.info("Roboter und Kamera getrennt.")


if __name__ == "__main__":
    main()
