"""
deploy_visual_coord.py
======================
Deployt die VisualCoord RSL-RL PPO-Policy (Task 1) auf den echten SO101.
Würfelposition kommt aus der Handgelenk-Kamera via HSV-Segmentierung.

Policy-Observation (33 dims, aus env.yaml):
  [0:6]   joint_pos_rel   – Gelenkpos relativ zu Default (rad)
  [6:12]  joint_vel_rel   – Gelenkgeschwindigkeit (rad/s)
  [12:15] ee_pos          – EE-Position via FK (m, Robot-Frame)
  [15:18] bowl_pos        – Bowl + 0.12m z-Offset (m, Robot-Frame)
  [18:21] cube_image      – [u, v, visible] aus HSV-Segmentierung (NDC)
  [21:27] color_one_hot   – Zielfarbe als 6-class One-Hot
  [27:33] last_action     – letzter Policy-Output (6D)

Aktionen (aus env.yaml):
  [0:5]  arm    – JointPositionAction, scale=0.5, use_default_offset=True
  [5]    gripper – JointPositionAction, scale=0.3, use_default_offset=True (kontinuierlich)

Default-Positionen (aus env.yaml init_state):
  shoulder_pan=0.0, shoulder_lift=-1.4, elbow_flex=0.4,
  wrist_flex=1.4, wrist_roll=-1.57, gripper=0.2

Nutzung
-------
python deploy_visual_coord.py \\
    --checkpoint Sim-to-Real/task_1/task_1_visual_coord/<run>/exported/policy.pt \\
    --robot_port COM5 \\
    --urdf_path path/to/so_arm101.urdf \\
    --color red \\
    --bowl_pos 0.30 0.10 0.00 \\
    --camera_id 1 \\
    --show_camera

WICHTIG VOR DEM ERSTEN RUN:
  1. Bowl-Position im Robot-Frame kalibrieren (Lineal).
  2. HSV-Ranges für die gewählte Farbe unter Deployment-Licht tunen.
  3. Ersten Run mit --max_delta_deg 1.0 starten und beobachten!
"""

import argparse
import logging
import math
import time

import cv2
import numpy as np
import torch

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


def busy_wait(dt: float) -> None:
    t_end = time.perf_counter() + dt
    while time.perf_counter() < t_end:
        pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Konstanten
# ══════════════════════════════════════════════════════════════════════════════

ARM_JOINT_NAMES    = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES    = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
NUM_ARM_JOINTS     = 5
NUM_ACTIONS        = 6

DEFAULT_JOINT_POS_RAD = np.array([
     0.00,   # shoulder_pan
    -1.40,   # shoulder_lift
     0.40,   # elbow_flex
     1.40,   # wrist_flex
    -1.57,   # wrist_roll
     0.18,   # gripper
], dtype=np.float64)

CONTROL_HZ           = 50
BOWL_HOVER_HEIGHT    = 0.12
ARM_ACTION_SCALE     = 0.5
GRIPPER_ACTION_SCALE = 0.3
GRIPPER_RAD_MIN      = -0.175
GRIPPER_RAD_MAX      =  1.745
MAX_DELTA_DEG        =  3.0

# NDC-Offset: wird NACH FOV-Skalierung addiert, NUR für Netzwerk-Input
# Visualisierung verwendet immer die Werte OHNE Offset
OFFSET_U =  0.18
OFFSET_V = -0.07

NUM_OBS = 33
assert NUM_OBS == 33

EE_OFFSET = np.array([0.01, 0.0, -0.09])

TAN_HALF_HFOV_SIM = 1.0691
TAN_HALF_VFOV_SIM = 0.6014


# ══════════════════════════════════════════════════════════════════════════════
#  HSV-Farb-Palette
# ══════════════════════════════════════════════════════════════════════════════

HSV_RANGES = {
    "blue":   [((100,  80,  60), (135, 255, 255))],
    "red":    [((  0,  80,  80), ( 10, 255, 255)),
               ((170,  80,  80), (180, 255, 255))],
    "green":  [((40,   60,  60), ( 85, 255, 255))],
    "yellow": [((20,   80,  80), ( 35, 255, 255))],
    "purple": [((130,  50,  50), (160, 255, 255))],
    "orange": [((10,   80,  80), ( 20, 255, 255))],
}

COLOR_INDEX = {"blue": 0, "red": 1, "green": 2, "yellow": 3, "purple": 4, "orange": 5}


# ══════════════════════════════════════════════════════════════════════════════
#  Kamera & HSV-Segmentierung
# ══════════════════════════════════════════════════════════════════════════════

class CubeDetection:
    """
    Ergebnis einer Würfeldetektion.

    nn_coords   : [u, v, visible] MIT Offset  → geht ins Netzwerk (3,)
    vis_u/vis_v : skalierte NDC OHNE Offset   → nur für Visualisierung
    visible     : bool
    """
    __slots__ = ("nn_coords", "vis_u", "vis_v", "visible")

    def __init__(self):
        self.nn_coords = np.zeros(3, dtype=np.float32)
        self.vis_u     = 0.0
        self.vis_v     = 0.0
        self.visible   = False


class WristCamera:
    """
    Öffnet die Handgelenk-Kamera und liefert pro Frame ein CubeDetection-Objekt.
    Optionaler Live-Debug-Feed zeigt Maske + Centroid (ohne Offset).
    """

    def __init__(
        self,
        camera_id: int,
        color_name: str,
        hfov_real_deg: float,
        vfov_real_deg: float,
        min_blob_area: int = 100,
        show_feed: bool = False,
        target_width: int = 1280,
        target_height: int = 720,
    ):
        self.color_name    = color_name
        self.min_blob_area = min_blob_area
        self.show_feed     = show_feed

        self.scale_u = math.tan(math.radians(hfov_real_deg / 2)) / TAN_HALF_HFOV_SIM
        self.scale_v = math.tan(math.radians(vfov_real_deg / 2)) / TAN_HALF_VFOV_SIM
        log.info(f"FOV-Korrektur : scale_u={self.scale_u:.4f}, scale_v={self.scale_v:.4f}")
        log.info(f"NDC-Offset    : offset_u={OFFSET_U:+.3f}, offset_v={OFFSET_V:+.3f}")

        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera {camera_id} konnte nicht geöffnet werden")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Kamera geöffnet: {self.W}x{self.H}, Farbe: {color_name}")

    def get_cube_coords(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Gibt zurück:
          nn_coords  : np.ndarray (3,) [u, v, visible] MIT Offset → Netzwerk
          debug_frame: BGR-Bild mit Overlay, oder None wenn show_feed=False
        """
        ret, frame = self.cap.read()
        if not ret:
            log.warning("Kamera-Frame konnte nicht gelesen werden")
            return np.zeros(3, dtype=np.float32), None

        det = self._segment(frame)

        debug_frame = None
        if self.show_feed:
            debug_frame = self._draw_debug(frame, det)

        return det.nn_coords, debug_frame

    def _segment(self, frame_bgr: np.ndarray) -> CubeDetection:
        det = CubeDetection()

        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in HSV_RANGES[self.color_name]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return det

        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_blob_area:
            return det

        M  = cv2.moments(c)
        px = M["m10"] / M["m00"]
        py = M["m01"] / M["m00"]

        # Schritt 1: rohe NDC (Bildmitte = 0)
        u_ndc = (px - self.W / 2) / (self.W / 2)
        v_ndc = (self.H / 2 - py) / (self.H / 2)

        # Schritt 2: FOV-Skalierung → für Visualisierung (kein Offset, kein Clip)
        det.vis_u = u_ndc * self.scale_u
        det.vis_v = v_ndc * self.scale_v

        # Schritt 3: Offset addieren + clippen → für Netzwerk
        u_net = float(np.clip(det.vis_u + OFFSET_U, -1.0, 1.0))
        v_net = float(np.clip(det.vis_v + OFFSET_V, -1.0, 1.0))

        det.nn_coords = np.array([u_net, v_net, 1.0], dtype=np.float32)
        det.visible   = True
        return det

    def _draw_debug(self, frame: np.ndarray, det: CubeDetection) -> np.ndarray:
        vis = frame.copy()

        # HSV-Maske als grüner Tint
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lo, hi) in HSV_RANGES[self.color_name]:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        tint = np.zeros_like(vis)
        tint[mask > 0] = (0, 180, 0)
        vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)

        if det.visible:
            # Kreuz aus vis_u/vis_v (OHNE Offset) → zeigt echte Pixelposition
            cx = int((det.vis_u + 1) / 2 * self.W)
            cy = int((1 - det.vis_v) / 2 * self.H)
            arm = 20
            cv2.line(vis,   (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
            cv2.line(vis,   (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

            # Zeile 1: Visualisierungswerte (ohne Offset)
            cv2.putText(vis,
                        f"vis  u={det.vis_u:+.3f}  v={det.vis_v:+.3f}",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            # Zeile 2: Netzwerk-Input (mit Offset)
            cv2.putText(vis,
                        f"net  u={det.nn_coords[0]:+.3f}  v={det.nn_coords[1]:+.3f}"
                        f"  (off {OFFSET_U:+.2f}/{OFFSET_V:+.2f})",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        else:
            cv2.putText(vis, "NICHT SICHTBAR",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(vis,
                    f"Farbe: {self.color_name}  "
                    f"scale_u={self.scale_u:.3f}  scale_v={self.scale_v:.3f}  "
                    f"min blob: {self.min_blob_area}px",
                    (10, self.H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return vis

    def release(self):
        self.cap.release()
        if self.show_feed:
            cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  Forward-Kinematik (Pinocchio)
# ══════════════════════════════════════════════════════════════════════════════

def build_fk_model(urdf_path: str):
    try:
        import pinocchio as pin
    except ImportError:
        raise ImportError("Pinocchio fehlt. Bitte 'pip install pin' ausführen.")

    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()

    def _jidx(name):
        return model.joints[model.getJointId(name)].idx_q

    j_idx            = {n: _jidx(n) for n in ALL_JOINT_NAMES}
    gripper_frame_id = model.getFrameId("gripper_link")
    log.info(f"Pinocchio geladen: nq={model.nq}")
    return model, data, j_idx, gripper_frame_id


def get_ee_pos(q_abs, pin_model, pin_data, j_idx, gripper_frame_id) -> np.ndarray:
    import pinocchio as pin
    q = np.zeros(pin_model.nq)
    for name, idx in j_idx.items():
        q[idx] = q_abs[name]
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    T = pin_data.oMf[gripper_frame_id]
    return (T.translation + T.rotation @ EE_OFFSET).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Observation-Builder  (33 dims)
# ══════════════════════════════════════════════════════════════════════════════

class ObservationBuilder:
    def __init__(self, pin_model, pin_data, j_idx, gripper_frame_id,
                 color_name: str, bowl_pos_base: np.ndarray, device: str):
        self.device           = device
        self.pin_model        = pin_model
        self.pin_data         = pin_data
        self.j_idx            = j_idx
        self.gripper_frame_id = gripper_frame_id
        self.bowl_pos_base    = bowl_pos_base.astype(np.float32)
        self.last_action      = np.zeros(NUM_ACTIONS, dtype=np.float32)

        self.color_one_hot = np.zeros(6, dtype=np.float32)
        self.color_one_hot[COLOR_INDEX[color_name]] = 1.0

    def build(self, joint_pos_deg: np.ndarray, joint_vel_deg_s: np.ndarray,
              cube_nn_coords: np.ndarray) -> torch.Tensor:
        """
        cube_nn_coords: [u, v, visible] MIT Offset, aus WristCamera.get_cube_coords()
        """
        joint_pos_rad = np.deg2rad(joint_pos_deg)
        joint_vel_rad = np.deg2rad(joint_vel_deg_s)

        joint_pos_rel = (joint_pos_rad - DEFAULT_JOINT_POS_RAD).astype(np.float32)
        joint_vel_rel = joint_vel_rad.astype(np.float32)

        q_abs  = {name: float(joint_pos_rad[i]) for i, name in enumerate(ALL_JOINT_NAMES)}
        ee_pos = get_ee_pos(q_abs, self.pin_model, self.pin_data,
                            self.j_idx, self.gripper_frame_id)

        obs_np = np.concatenate([
            joint_pos_rel,               # [0:6]
            joint_vel_rel,               # [6:12]
            ee_pos,                      # [12:15]
            self.bowl_pos_base,          # [15:18]
            cube_nn_coords[:3],          # [18:21]  ← exakt 3 Werte, mit Offset
            self.color_one_hot,          # [21:27]
            self.last_action,            # [27:33]
        ]).astype(np.float32)

        assert obs_np.shape == (NUM_OBS,), f"Obs-Shape falsch: {obs_np.shape}"
        return torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

    def update_last_action(self, action: np.ndarray):
        self.last_action = action.astype(np.float32)

    def reset(self):
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Action-Interpreter
# ══════════════════════════════════════════════════════════════════════════════

class ActionInterpreter:
    def __init__(self, max_delta_deg: float = MAX_DELTA_DEG):
        self.max_delta_deg = max_delta_deg

    def interpret(self, raw_action: np.ndarray,
                  current_joint_pos_deg: np.ndarray) -> dict:
        # Arm [0:5]
        arm_target_rad  = DEFAULT_JOINT_POS_RAD[:NUM_ARM_JOINTS] + ARM_ACTION_SCALE * raw_action[:NUM_ARM_JOINTS]
        arm_target_deg  = np.rad2deg(arm_target_rad)
        delta           = arm_target_deg - current_joint_pos_deg[:NUM_ARM_JOINTS]
        delta_clipped   = np.clip(delta, -self.max_delta_deg, self.max_delta_deg)
        arm_targets_deg = current_joint_pos_deg[:NUM_ARM_JOINTS] + delta_clipped

        # Gripper [5] — konsistent mit Arm
        gripper_target_rad    = DEFAULT_JOINT_POS_RAD[5] + GRIPPER_ACTION_SCALE * raw_action[5]
        current_gripper_deg   = current_joint_pos_deg[5]
        gripper_delta         = np.rad2deg(gripper_target_rad) - current_gripper_deg
        gripper_delta_clipped = np.clip(gripper_delta, -self.max_delta_deg, self.max_delta_deg)
        gripper_cmd_rad       = float(np.deg2rad(current_gripper_deg + gripper_delta_clipped))

        return {"arm_targets_deg": arm_targets_deg, "gripper_cmd_rad": gripper_cmd_rad}


# ══════════════════════════════════════════════════════════════════════════════
#  Roboter-Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def read_robot_state(robot) -> tuple[np.ndarray, np.ndarray]:
    obs         = robot.get_observation()
    arm_deg     = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINT_NAMES], dtype=np.float64)
    gripper_pct = float(obs["gripper.pos"])
    gripper_rad = GRIPPER_RAD_MIN + (gripper_pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)
    joint_pos_deg = np.append(arm_deg, np.rad2deg(gripper_rad))
    return joint_pos_deg, np.zeros(6, dtype=np.float64)


def send_action(robot, arm_targets_deg: np.ndarray, gripper_cmd_rad: float):
    span        = GRIPPER_RAD_MAX - GRIPPER_RAD_MIN
    gripper_pct = float(np.clip((gripper_cmd_rad - GRIPPER_RAD_MIN) / span * 100.0, 0.0, 100.0))
    action      = {f"{m}.pos": float(d) for m, d in zip(ARM_JOINT_NAMES, arm_targets_deg)}
    action["gripper.pos"] = gripper_pct
    robot.send_action(action)


def move_to_default(robot, steps: int = 100, step_delay: float = 0.02):
    log.info("Fahre in Default-Position ...")
    target = {f"{m}.pos": float(np.rad2deg(DEFAULT_JOINT_POS_RAD[i]))
              for i, m in enumerate(ARM_JOINT_NAMES)}
    target["gripper.pos"] = 50.0
    obs   = robot.get_observation()
    start = {k: obs[k] for k in target}
    for i in range(1, steps + 1):
        alpha  = i / steps
        action = {j: start[j] + alpha * (target[j] - start[j]) for j in target}
        robot.send_action(action)
        time.sleep(step_delay)
    log.info("Default-Position erreicht.")


REST_POSITION = {
    "shoulder_pan.pos":   4.75,
    "shoulder_lift.pos": -101.1,
    "elbow_flex.pos":     95.6,
    "wrist_flex.pos":     66.0,
    "wrist_roll.pos":    -89.3,
    "gripper.pos":         0.5,
}


def move_to_rest(robot, steps: int = 100, step_delay: float = 0.02):
    log.info("Fahre in Rest-Position ...")
    obs   = robot.get_observation()
    start = {k: obs[k] for k in REST_POSITION}
    for i in range(1, steps + 1):
        alpha  = i / steps
        action = {j: start[j] + alpha * (REST_POSITION[j] - start[j]) for j in REST_POSITION}
        robot.send_action(action)
        time.sleep(step_delay)
    log.info("Rest-Position erreicht.")


# ══════════════════════════════════════════════════════════════════════════════
#  Policy laden
# ══════════════════════════════════════════════════════════════════════════════

def load_policy(checkpoint_path: str, device: str):
    log.info(f"Lade Policy: {checkpoint_path}")
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.eval()
    log.info("Policy geladen (TorchScript, Normalisierung eingebettet).")

    class _Wrapper:
        def __init__(self, m): self.model = m
        def act_inference(self, obs): return self.model(obs)

    return _Wrapper(policy)


# ══════════════════════════════════════════════════════════════════════════════
#  Haupt-Episode-Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(robot, policy, obs_builder: ObservationBuilder,
                action_interp: ActionInterpreter, camera: WristCamera,
                episode_duration_s: float, device: str, show_feed: bool):

    dt    = 1.0 / CONTROL_HZ
    t_end = time.perf_counter() + episode_duration_s
    prev_joint_pos_deg = None
    obs_builder.reset()

    log.info(f"Episode gestartet ({episode_duration_s:.0f}s @ {CONTROL_HZ}Hz)")
    log.info(f"  Bowl-Pos (mit offset): {obs_builder.bowl_pos_base}")
    log.info(f"  Farbe: {camera.color_name}  one-hot={obs_builder.color_one_hot.tolist()}")

    step = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        # ── Kamera → nn_coords hat Offset, ist bereit fürs Netzwerk ─────────
        cube_nn_coords, debug_frame = camera.get_cube_coords()

        if show_feed and debug_frame is not None:
            cv2.imshow("Wrist Camera — VisualCoord", debug_frame)
            if cv2.waitKey(1) == ord("q"):
                log.info("Abbruch durch Benutzer (q gedrückt).")
                break

        # ── Roboter-State ────────────────────────────────────────────────────
        joint_pos_deg, _ = read_robot_state(robot)
        if prev_joint_pos_deg is not None:
            joint_vel_deg_s = (joint_pos_deg - prev_joint_pos_deg) / dt
        else:
            joint_vel_deg_s = np.zeros(6)
        prev_joint_pos_deg = joint_pos_deg.copy()

        # ── Observation ──────────────────────────────────────────────────────
        obs = obs_builder.build(joint_pos_deg, joint_vel_deg_s, cube_nn_coords)

        # ── Inferenz ─────────────────────────────────────────────────────────
        with torch.no_grad():
            raw_action_np = policy.act_inference(obs).squeeze(0).cpu().numpy()

        # ── Action senden ────────────────────────────────────────────────────
        result = action_interp.interpret(raw_action_np, joint_pos_deg)
        send_action(robot, result["arm_targets_deg"], result["gripper_cmd_rad"])
        obs_builder.update_last_action(raw_action_np)

        step += 1
        if step % CONTROL_HZ == 0:
            log.info(
                f"  t={step/CONTROL_HZ:.1f}s | "
                f"cube_net=[{cube_nn_coords[0]:+.2f},{cube_nn_coords[1]:+.2f},"
                f"vis={int(cube_nn_coords[2])}] | "
                f"gripper_rad={result['gripper_cmd_rad']:.3f} | "
                f"arm_tgt={result['arm_targets_deg'].round(1)}"
            )

        elapsed = time.perf_counter() - t0
        busy_wait(max(0.0, dt - elapsed))

    log.info("Episode beendet.")


# ══════════════════════════════════════════════════════════════════════════════
#  Mock-Roboter
# ══════════════════════════════════════════════════════════════════════════════

class _MockRobot:
    is_connected = True
    def connect(self, **kw): pass
    def disconnect(self): pass
    def get_observation(self):
        obs = {f"{m}.pos": float(np.rad2deg(DEFAULT_JOINT_POS_RAD[i]))
               for i, m in enumerate(ARM_JOINT_NAMES)}
        obs["gripper.pos"] = 50.0
        return obs
    def send_action(self, a): return a


# ══════════════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VisualCoord Policy Deployment — SO101")

    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--urdf_path",   required=True)
    parser.add_argument("--robot_port",  default="COM5")
    parser.add_argument("--robot_id",    default="follower_arm")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mock",        action="store_true")

    parser.add_argument("--color",       required=True, choices=list(HSV_RANGES.keys()))
    parser.add_argument("--bowl_pos",    nargs=3, type=float,
                        default=[0.30, 0.10, 0.00], metavar=("X", "Y", "Z"))

    parser.add_argument("--camera_id",   type=int,   default=1)
    parser.add_argument("--hfov",        type=float, default=100.82)
    parser.add_argument("--vfov",        type=float, default=64.80)
    parser.add_argument("--min_blob",    type=int,   default=20)
    parser.add_argument("--show_camera", action="store_true")

    parser.add_argument("--num_episodes",     type=int,   default=1)
    parser.add_argument("--episode_duration", type=float, default=60.0)
    parser.add_argument("--reset_duration",   type=float, default=15.0)
    parser.add_argument("--max_delta_deg",    type=float, default=MAX_DELTA_DEG)

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SO101 VisualCoord Policy Deployment")
    log.info("=" * 60)
    log.info(f"Checkpoint   : {args.checkpoint}")
    log.info(f"URDF         : {args.urdf_path}")
    log.info(f"Farbe        : {args.color}  (one-hot idx {COLOR_INDEX[args.color]})")
    log.info(f"Bowl-Pos     : {args.bowl_pos} m  (+0.12m z-Offset wird addiert)")
    log.info(f"Kamera       : id={args.camera_id}, HFOV={args.hfov}°, VFOV={args.vfov}°")
    log.info(f"NDC-Offset   : u={OFFSET_U:+.3f}, v={OFFSET_V:+.3f}")
    log.info(f"Min-Blob     : {args.min_blob}px")
    log.info(f"Live-Feed    : {'AN' if args.show_camera else 'AUS'}")
    log.info(f"Obs-Dim      : {NUM_OBS} (6+6+3+3+3+6+6)")
    log.info(f"Max-Delta    : {args.max_delta_deg}°/step")
    log.info(f"Default-Pos  : {np.rad2deg(DEFAULT_JOINT_POS_RAD).round(2)}°")

    bowl_pos_base      = np.array(args.bowl_pos, dtype=np.float32)
    bowl_pos_base[2]  += BOWL_HOVER_HEIGHT

    pin_model, pin_data, j_idx, gripper_frame_id = build_fk_model(args.urdf_path)
    policy = load_policy(args.checkpoint, args.device)

    camera = WristCamera(
        camera_id     = args.camera_id,
        color_name    = args.color,
        hfov_real_deg = args.hfov,
        vfov_real_deg = args.vfov,
        min_blob_area = args.min_blob,
        show_feed     = args.show_camera,
    )

    obs_builder = ObservationBuilder(
        pin_model        = pin_model,
        pin_data         = pin_data,
        j_idx            = j_idx,
        gripper_frame_id = gripper_frame_id,
        color_name       = args.color,
        bowl_pos_base    = bowl_pos_base,
        device           = args.device,
    )

    action_interp = ActionInterpreter(max_delta_deg=args.max_delta_deg)

    if args.mock:
        robot = _MockRobot()
        log.warning("MOCK-MODUS aktiv — kein echter Roboter wird bewegt!")
    else:
        cfg   = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
        robot = SO101Follower(cfg)
    robot.connect()
    log.info(f"Roboter verbunden @ {args.robot_port}")

    if not args.mock:
        move_to_default(robot)
        log.info("Warte 2s vor Policy-Start ...")
        time.sleep(2.0)

    try:
        for ep in range(args.num_episodes):
            log.info(f"\n{'─'*50}")
            log.info(f"Episode {ep + 1} / {args.num_episodes}")
            log.info(f"{'─'*50}")

            run_episode(
                robot              = robot,
                policy             = policy,
                obs_builder        = obs_builder,
                action_interp      = action_interp,
                camera             = camera,
                episode_duration_s = args.episode_duration,
                device             = args.device,
                show_feed          = args.show_camera,
            )

            if not args.mock:
                move_to_rest(robot)

            if ep < args.num_episodes - 1:
                log.info(f"\nSzene resetten — {args.reset_duration:.0f}s Pause ...")
                log.info("→ Würfel zurückstellen, Roboter in Startposition bringen.")
                time.sleep(args.reset_duration)

    except KeyboardInterrupt:
        log.info("\nDurch Benutzer abgebrochen (Ctrl+C).")
    finally:
        camera.release()
        robot.disconnect()
        log.info("Kamera + Roboter getrennt. Fertig.")


if __name__ == "__main__":
    main()