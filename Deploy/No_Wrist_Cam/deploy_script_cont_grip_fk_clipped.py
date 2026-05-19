"""
deploy_so101_lift.py
====================
Deployt die RSL-RL PPO-Policy (trainiert in Isaac-SO-ARM101-Lift)
auf den echten SO101-Roboter via LeRobot.

Policy-Details (aus env.yaml / agent.yaml):
  - Architektur       : MLP ActorCritic [256, 128, 64], Aktivierung ELU
  - Normalisierung    : empirical_normalization aus Checkpoint geladen
  - Policy-Observation: 27-dimensionaler Vektor
      [0:6]   joint_pos_rel           – Gelenk-Positionen relativ zu Default (Rad)
      [6:12]  joint_vel_rel           – Gelenk-Geschwindigkeiten (Rad/s)
      [12:15] ee_position             – EE-Position im Robot-Root-Frame (m), via FK
      [15:18] initial_object_position – INITIALE (feste) Würfel-Pos im Robot-Root-Frame (m)
      [18:21] bowl_position           – Bowl-Pos + 0.12m z-Offset im Robot-Root-Frame (m)
      [21:27] last_action             – letzte Policy-Action (6D)
  - Critic-Observation (nur Training, zusätzlich object_position = live)
  - Action            : 6-dimensional
      [0:5]  arm_action  – JointPosition-Targets, scale=2.5, use_default_offset=True
                           Joints: shoulder_pan, shoulder_lift, elbow_flex,
                                   wrist_flex, wrist_roll
      [5]    gripper     – JointPosition-Target, scale=2.5, use_default_offset=True

Default-Joint-Positionen (aus env.yaml init_state):
  shoulder_pan=0.0, shoulder_lift=-1.4, elbow_flex=0.4,
  wrist_flex=1.4, wrist_roll=-1.57, gripper=0.2

WICHTIGE ÄNDERUNG gegenüber alter Policy:
  - Action scale ist nun 2.5 (statt 0.5/0.3)
  - initial_object_position ist die feste Startposition des Würfels (nicht live)
  - Neue Default-Gelenk-Positionen

Nutzung
-------
python deploy_so101_lift.py \
    --checkpoint logs/rsl_rl/lift/2026-05-01_12-55-26/model_2999.pt \
    --robot_port /dev/ttyACM0 \
    --robot_id my_so101 \
    --urdf_path path/to/so_arm101.urdf \
    --object_pos 0.30 0.00 0.01 \
    --bowl_pos 0.30 0.10 0.00 \
    --num_episodes 5

WICHTIG VOR DEM ERSTEN RUN:
  1. Roboter kalibrieren (falls noch nicht geschehen):
     lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_so101
  2. Objekt- und Bowl-Position im Robot-Frame messen (mit Lineal).
  3. Im ersten Run --max_delta_deg=1.0 verwenden und den Roboter beobachten!
  4. pip install pin  (Pinocchio, für die FK)
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ─── LeRobot (0.5+) ─────────────────────────────────────────────────────────
from lerobot.robots.so_follower import SO100Follower, SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


def busy_wait(dt: float) -> None:
    """High-precision spin-wait."""
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

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
NUM_ARM_JOINTS = 5
NUM_ACTIONS = 6

# Default-Positionen (Rad) – aus env.yaml init_state
# GEÄNDERT gegenüber alter Policy:
#   shoulder_lift: -0.6 → -1.4
#   elbow_flex:    -0.6 → +0.4
#   wrist_flex:     1.57 → 1.4
DEFAULT_JOINT_POS_RAD = np.array([
     0.00,   # shoulder_pan
    -1.40,   # shoulder_lift
     0.40,   # elbow_flex
     1.40,   # wrist_flex
    -1.57,   # wrist_roll
     0.20,   # gripper
], dtype=np.float64)

# sim.dt=0.01s, decimation=2 → 50 Hz
CONTROL_HZ = 50

# bowl_position height_offset aus env.yaml observations.bowl_position
BOWL_HOVER_HEIGHT = 0.12

# Gripper-Kommandos (aus env.yaml init_state / actuator limits)
GRIPPER_OPEN_CMD_RAD  =  0.5
GRIPPER_CLOSE_CMD_RAD = -0.1
GRIPPER_RAD_MIN = GRIPPER_CLOSE_CMD_RAD
GRIPPER_RAD_MAX = GRIPPER_OPEN_CMD_RAD

# Action-Skalierung aus env.yaml actions:
#   arm_action:     scale=2.5, use_default_offset=True
#   gripper_action: scale=2.5, use_default_offset=True
# GEÄNDERT: war 0.5 / 0.3 in der alten Policy
ARM_ACTION_SCALE     = 2.5
GRIPPER_ACTION_SCALE = 2.5

# Safety
MAX_DELTA_DEG = 3.0

# Observation-Dimensionen (aus env.yaml observations.policy)
OBS_JOINT_POS          = 6
OBS_JOINT_VEL          = 6
OBS_EE_POS             = 3   # EE-Position via FK
OBS_INIT_OBJECT_POS    = 3   # initiale (feste) Würfel-Position
OBS_BOWL_POS           = 3   # Bowl-Position + height_offset
OBS_LAST_ACTION        = NUM_ACTIONS  # 6
NUM_OBS = (OBS_JOINT_POS + OBS_JOINT_VEL + OBS_EE_POS
           + OBS_INIT_OBJECT_POS + OBS_BOWL_POS + OBS_LAST_ACTION)
# = 6 + 6 + 3 + 3 + 3 + 6 = 27

assert NUM_OBS == 27, f"Unerwartete OBS-Dimension: {NUM_OBS}"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Forward-Kinematik Setup (Pinocchio)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# EE-Offset in gripper_link-lokalem Frame (aus env.yaml ee_frame target_frames offset)
EE_OFFSET = np.array([0.01, 0.0, -0.09])  # metres


def build_fk_model(urdf_path: str):
    """
    Lädt das Pinocchio-Modell aus der URDF und gibt (model, data, J_IDX,
    GRIPPER_LINK_FRAME_ID) zurück.
    Muss einmal beim Start aufgerufen werden.
    """
    try:
        import pinocchio as pin
    except ImportError:
        raise ImportError(
            "Pinocchio nicht installiert. Bitte 'pip install pin' ausführen."
        )

    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()

    def _jidx(name):
        return model.joints[model.getJointId(name)].idx_q

    J_IDX = {
        "shoulder_pan":  _jidx("shoulder_pan"),
        "shoulder_lift": _jidx("shoulder_lift"),
        "elbow_flex":    _jidx("elbow_flex"),
        "wrist_flex":    _jidx("wrist_flex"),
        "wrist_roll":    _jidx("wrist_roll"),
        "gripper":       _jidx("gripper"),
    }
    GRIPPER_LINK_FRAME_ID = model.getFrameId("gripper_link")

    log.info(f"Pinocchio FK-Modell geladen: {urdf_path}")
    log.info(f"  nq={model.nq}, nv={model.nv}, frames={model.nframes}")

    return model, data, J_IDX, GRIPPER_LINK_FRAME_ID


def get_ee_pos(
    q_abs: dict,
    pin_model,
    pin_data,
    j_idx: dict,
    gripper_frame_id: int,
) -> np.ndarray:
    """
    Berechnet die EE-Position (Fingerspitzen-Mitte) im Robot-Base-Frame via FK.

    q_abs : {joint_name: angle_rad} – absolute Encoder-Werte.
    Returns: ee_pos (3,) in Metern im Robot-Base-Frame.
    """
    import pinocchio as pin

    q = np.zeros(pin_model.nq)
    for name, idx in j_idx.items():
        q[idx] = q_abs[name]

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    T = pin_data.oMf[gripper_frame_id]           # SE3: base_link → gripper_link
    ee_pos = T.translation + T.rotation @ EE_OFFSET  # (3,) in base frame
    return ee_pos.astype(np.float32)


def move_to_default(robot, steps: int = 100, step_delay: float = 0.02):
    """Fährt den Roboter sanft in die Default-Position bevor die Policy startet."""
    log.info("Fahre in Default-Position ...")

    target = {f"{m}.pos": float(np.rad2deg(DEFAULT_JOINT_POS_RAD[i])) for i, m in enumerate(ARM_JOINT_NAMES)}
    target["gripper.pos"] = 50.0

    obs = robot.get_observation()
    start = {k: obs[k] for k in target.keys()}

    log.info(f"  Start : { {k: round(v,1) for k,v in start.items()} }")
    log.info(f"  Target: { {k: round(v,1) for k,v in target.items()} }")

    for i in range(1, steps + 1):
        alpha = i / steps
        action = {joint: start[joint] + alpha * (target[joint] - start[joint]) for joint in target}
        robot.send_action(action)
        time.sleep(step_delay)

    log.info("Default-Position erreicht.")


# Rest-Position nach jeder Episode (gemessene Ruhepose)
REST_POSITION = {
    "shoulder_pan.pos":  4.747252747252747,
    "shoulder_lift.pos": -101.0989010989011,
    "elbow_flex.pos":    95.6043956043956,
    "wrist_flex.pos":    66.02197802197803,
    "wrist_roll.pos":    -89.27472527472527,
    "gripper.pos":       0.4857737682165163,
}


def move_to_rest(robot, steps: int = 100, step_delay: float = 0.02):
    """Fährt den Roboter sanft in die Rest-Position nach einer Episode."""
    log.info("Fahre in Rest-Position ...")

    obs = robot.get_observation()
    start = {k: obs[k] for k in REST_POSITION.keys()}

    log.info(f"  Start : { {k: round(v,1) for k,v in start.items()} }")
    log.info(f"  Target: { {k: round(v,1) for k,v in REST_POSITION.items()} }")

    for i in range(1, steps + 1):
        alpha = i / steps
        action = {joint: start[joint] + alpha * (REST_POSITION[joint] - start[joint]) for joint in REST_POSITION}
        robot.send_action(action)
        time.sleep(step_delay)

    log.info("Rest-Position erreicht.")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Standalone Actor                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class StandaloneActor(nn.Module):
    """
    Minimaler MLP-Actor passend zur rsl_rl-ActorCritic-Architektur.

    State-Dict-Struktur im Checkpoint:
      actor.0.weight / actor.0.bias  – Linear(obs → hidden[0])
      actor.2.weight / actor.2.bias  – Linear(hidden[0] → hidden[1])
      actor.4.weight / actor.4.bias  – Linear(hidden[1] → hidden[2])
      actor.6.weight / actor.6.bias  – Linear(hidden[2] → actions)
    Optional:
      obs_normalizer.mean / obs_normalizer.var
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

        self.register_buffer("obs_mean", torch.zeros(num_obs))
        self.register_buffer("obs_var", torch.ones(num_obs))
        self._use_normalizer = False

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        if self._use_normalizer:
            obs = (obs - self.obs_mean) / (self.obs_var + 1e-8).sqrt()
        return self.actor(obs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Policy laden                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Schwellwert für explodierende Actions (Betrag) – Policy-Output sollte in [-5, 5] liegen
ACTION_EXPLOSION_THRESHOLD = 100.0


def load_rsl_rl_policy(checkpoint_path: str, device: str):
    """
    Lädt die Policy intelligent:
      1. Versucht torch.jit.load (TorchScript-Export)
      2. Falls das fehlschlägt, versucht torch.load als state_dict
         und baut den StandaloneActor neu auf.

    Bei state_dict wird auch nach obs_normalizer.mean/var gesucht
    und in den Actor geladen, falls vorhanden.
    """
    log.info(f"Lade Policy: {checkpoint_path}")

    # ── Versuch 1: TorchScript ────────────────────────────────────────────
    try:
        policy = torch.jit.load(checkpoint_path, map_location=device)
        policy.eval()
        log.info("  → TorchScript-Format erkannt und geladen.")

        class JitWrapper:
            def __init__(self, model):
                self.model = model
            def act_inference(self, obs):
                return self.model(obs)

        log.info(f"  Obs-Dim    : {NUM_OBS}")
        log.info(f"  Action-Dim : {NUM_ACTIONS}")
        return JitWrapper(policy)

    except Exception as e:
        log.info(f"  → Kein TorchScript ({e}), versuche state_dict ...")

    # ── Versuch 2: state_dict (rsl_rl Checkpoint) ─────────────────────────
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # rsl_rl speichert unter 'model_state_dict' oder direkt als state_dict
    if isinstance(ck, dict) and "model_state_dict" in ck:
        state = ck["model_state_dict"]
        iteration = ck.get("iter", "?")
        log.info(f"  → rsl_rl Checkpoint, Iteration {iteration}")
    elif isinstance(ck, dict):
        state = ck
        log.info(f"  → Roher state_dict, Keys: {list(state.keys())[:6]} ...")
    else:
        raise ValueError(f"Unbekanntes Checkpoint-Format: {type(ck)}")

    # Normalisierungs-Stats aus Checkpoint extrahieren
    obs_mean = state.get("obs_normalizer.mean", state.get("actor_critic.obs_normalizer.mean"))
    obs_var  = state.get("obs_normalizer.var",  state.get("actor_critic.obs_normalizer.var"))

    # Actor aufbauen
    hidden_dims = [256, 128, 64]  # aus agent.yaml
    actor = StandaloneActor(NUM_OBS, NUM_ACTIONS, hidden_dims)

    # Actor-Gewichte laden – Schlüssel können "actor_critic.actor.*" oder "actor.*" sein
    actor_state = {}
    for k, v in state.items():
        if k.startswith("actor_critic.actor."):
            actor_state[k[len("actor_critic."):]] = v
        elif k.startswith("actor."):
            actor_state[k] = v

    if not actor_state:
        raise ValueError(
            f"Keine actor.*-Schlüssel im Checkpoint gefunden.\n"
            f"Vorhandene Schlüssel: {list(state.keys())}"
        )

    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        log.warning(f"  Fehlende Keys beim Laden: {missing}")
    if unexpected:
        log.warning(f"  Unbekannte Keys beim Laden: {unexpected}")

    # Normalisierung setzen
    if obs_mean is not None and obs_var is not None:
        actor.obs_mean.copy_(obs_mean.to(device))
        actor.obs_var.copy_(obs_var.to(device))
        actor._use_normalizer = True
        log.info(f"  → Obs-Normalisierung geladen (mean={obs_mean[:4].numpy().round(3)}...)")
    else:
        log.warning("  → KEINE Obs-Normalisierung im Checkpoint gefunden!")
        log.warning("     Wenn die Policy mit empirical_normalization trainiert wurde,")
        log.warning("     werden die Outputs falsch sein (sehr große Werte).")

    actor.to(device)
    actor.eval()

    log.info(f"  Obs-Dim    : {NUM_OBS}")
    log.info(f"  Action-Dim : {NUM_ACTIONS}")
    log.info(f"  Normalizer : {'AN' if actor._use_normalizer else 'AUS'}")
    return actor


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Observation-Builder                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ObservationBuilder:
    """
    Baut den 27-dimensionalen Observation-Vektor für die Policy.

    Struktur (aus env.yaml observations.policy):
      [0:6]   joint_pos_rel          = aktuelle Gelenkpos - default_joint_pos (Rad)
      [6:12]  joint_vel_rel          = aktuelle Gelenkvel (Rad/s)
      [12:15] ee_position            = EE-Position im Robot-Root-Frame (m), via FK
      [15:18] initial_object_position= FESTE initiale Würfel-Pos im Robot-Root-Frame (m)
      [18:21] bowl_position          = Bowl-Pos + 0.12m z-Offset im Robot-Root-Frame (m)
      [21:27] last_action            = letzte Policy-Action (6D)

    WICHTIG: initial_object_position ist die feste Startposition des Würfels
    (entspricht --object_pos). Sie ändert sich während der Episode NICHT –
    genau wie im Training, wo initial_object_position_in_robot_root_frame
    die gespeicherte Initialposition zurückgibt.
    """

    def __init__(
        self,
        pin_model,
        pin_data,
        j_idx: dict,
        gripper_frame_id: int,
        device: str = "cpu",
    ):
        self.device = device
        self._default_pos_rad = DEFAULT_JOINT_POS_RAD.copy()
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # FK-Objekte
        self._pin_model = pin_model
        self._pin_data = pin_data
        self._j_idx = j_idx
        self._gripper_frame_id = gripper_frame_id

    def build(
        self,
        joint_pos_deg: np.ndarray,              # (6,) aktuelle Gelenkpos in Grad
        joint_vel_deg_s: np.ndarray,            # (6,) aktuelle Gelenkvel in Grad/s
        initial_object_pos_robot_frame: np.ndarray,  # (3,) feste Würfel-Startpos (Meter)
        bowl_pos_robot_frame: np.ndarray,       # (3,) Bowl-Pos in Meter, Robot-Frame
    ) -> torch.Tensor:
        """Gibt einen (1, 27) Tensor zurück."""

        joint_pos_rad = np.deg2rad(joint_pos_deg)
        joint_vel_rad = np.deg2rad(joint_vel_deg_s)

        joint_pos_rel = joint_pos_rad - self._default_pos_rad
        joint_vel_rel = joint_vel_rad

        # ── EE-Position via FK ────────────────────────────────────────────
        q_abs = {name: float(joint_pos_rad[i]) for i, name in enumerate(ALL_JOINT_NAMES)}
        ee_pos = get_ee_pos(
            q_abs,
            self._pin_model,
            self._pin_data,
            self._j_idx,
            self._gripper_frame_id,
        )

        print(ee_pos)

        # Bowl-Position mit height_offset (aus env.yaml: height_offset=0.12)
        bowl_with_offset = bowl_pos_robot_frame.copy().astype(np.float32)
        bowl_with_offset[2] += BOWL_HOVER_HEIGHT

        obs_np = np.concatenate([
            joint_pos_rel.astype(np.float32),                    # [0:6]
            joint_vel_rel.astype(np.float32),                    # [6:12]
            ee_pos,                                              # [12:15]
            initial_object_pos_robot_frame.astype(np.float32),  # [15:18]
            bowl_with_offset,                                    # [18:21]
            self._last_action,                                   # [21:27]
        ])

        assert obs_np.shape == (NUM_OBS,), f"Obs shape falsch: {obs_np.shape}"
        return torch.from_numpy(obs_np).unsqueeze(0).to(self.device)  # (1, 27)

    def update_last_action(self, action: np.ndarray) -> None:
        self._last_action = action.astype(np.float32)

    def reset(self) -> None:
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Action-Interpreter                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ActionInterpreter:
    """
    Wandelt den 6-dimensionalen Policy-Output in Roboter-Kommandos um.

    Arm + Gripper (JointPositionAction): scale=2.5, use_default_offset=True
      → Ziel_arm     = default_pos_rad[:5] + 2.5 * raw_action[:5]
      → Ziel_gripper = default_pos_rad[5]  + 2.5 * raw_action[5]
                       geclippt auf [GRIPPER_RAD_MIN, GRIPPER_RAD_MAX]

    GEÄNDERT gegenüber alter Policy: scale war 0.5 (arm) / 0.3 (gripper).
    Mit scale=2.5 sind die Targets deutlich weiter von der Default-Pos entfernt.
    """

    def __init__(self, max_delta_deg: float = MAX_DELTA_DEG):
        self._max_delta_deg = max_delta_deg

    def interpret(
        self,
        raw_action: np.ndarray,
        current_joint_pos_deg: np.ndarray,
    ) -> dict:
        # ─ Arm-Joints (0:5) ─
        arm_target_rad = DEFAULT_JOINT_POS_RAD[:NUM_ARM_JOINTS] + ARM_ACTION_SCALE * raw_action[:NUM_ARM_JOINTS]
        arm_target_deg = np.rad2deg(arm_target_rad)

        current_arm_deg = current_joint_pos_deg[:NUM_ARM_JOINTS]
        delta = arm_target_deg - current_arm_deg
        delta_clipped = np.clip(delta, -self._max_delta_deg, self._max_delta_deg)
        if not np.allclose(delta, delta_clipped, atol=0.01):
            log.debug(f"Arm-Delta geclippt: max={np.abs(delta).max():.2f}° → {np.abs(delta_clipped).max():.2f}°")
        arm_targets_deg = current_arm_deg + delta_clipped

        # ─ Gripper (5) – JointPositionAction wie Arm-Joints ─
        gripper_target_rad = DEFAULT_JOINT_POS_RAD[NUM_ARM_JOINTS] + GRIPPER_ACTION_SCALE * raw_action[NUM_ARM_JOINTS]
        gripper_target_rad = np.clip(gripper_target_rad, GRIPPER_RAD_MIN, GRIPPER_RAD_MAX)

        current_gripper_deg = current_joint_pos_deg[NUM_ARM_JOINTS]
        gripper_delta = np.rad2deg(gripper_target_rad) - current_gripper_deg
        gripper_delta_clipped = np.clip(gripper_delta, -self._max_delta_deg, self._max_delta_deg)
        gripper_cmd_deg = current_gripper_deg + gripper_delta_clipped
        gripper_cmd_rad = np.deg2rad(gripper_cmd_deg)

        return {
            "arm_targets_deg": arm_targets_deg,
            "gripper_cmd_rad": gripper_cmd_rad,
        }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Roboter-Hilfsfunktionen                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def read_robot_state(robot) -> tuple[np.ndarray, np.ndarray]:
    """
    Liest Gelenk-Positionen vom Roboter (neue LeRobot-API).
    Gripper (0-100%) wird linear in Äquivalent-Grad umgerechnet.
    Returns: (joint_pos_deg (6,), dummy_vel (6,))
    """
    obs = robot.get_observation()

    print("Reading from robot")
    print(obs)

    arm_pos_deg = np.array([float(obs[f"{m}.pos"]) for m in ARM_JOINT_NAMES], dtype=np.float64)

    gripper_pct = float(obs["gripper.pos"])
    gripper_rad = GRIPPER_RAD_MIN + (gripper_pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)
    gripper_as_deg = float(np.rad2deg(gripper_rad))

    joint_pos_deg = np.append(arm_pos_deg, gripper_as_deg)
    return joint_pos_deg, np.zeros(6, dtype=np.float64)


def send_action_to_robot(robot, arm_targets_deg: np.ndarray, gripper_cmd_rad: float):
    """Sendet Arm-Targets (Grad) und Gripper-Kommando (als 0-100%) an den Roboter."""
    gripper_span = GRIPPER_RAD_MAX - GRIPPER_RAD_MIN
    gripper_pct = float(np.clip((gripper_cmd_rad - GRIPPER_RAD_MIN) / gripper_span * 100.0, 0.0, 100.0))

    action: dict[str, float] = {f"{m}.pos": float(d) for m, d in zip(ARM_JOINT_NAMES, arm_targets_deg)}
    action["gripper.pos"] = gripper_pct
    print(action)
    robot.send_action(action)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Haupt-Deployment-Loop                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_episode(
    robot,
    policy_nn,
    obs_builder: ObservationBuilder,
    action_interp: ActionInterpreter,
    object_pos: np.ndarray,
    bowl_pos: np.ndarray,
    episode_duration_s: float,
    device: str,
) -> None:
    """
    Führt eine einzelne Lift-and-Place-Episode aus.

    object_pos : initiale (feste) Würfel-Position im Robot-Root-Frame (Meter).
                 Entspricht initial_object_position im Training – bleibt konstant!
    bowl_pos   : Bowl-Position im Robot-Root-Frame (Meter).
    """
    dt = 1.0 / CONTROL_HZ
    t_end = time.perf_counter() + episode_duration_s

    prev_joint_pos_deg = None
    obs_builder.reset()

    log.info(f"Episode gestartet ({episode_duration_s:.0f}s @ {CONTROL_HZ}Hz)")
    log.info(f"  Initiale Objekt-Pos (Robot-Frame): {object_pos}")
    log.info(f"  Bowl-Pos            (Robot-Frame): {bowl_pos}")

    step = 0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()

        # ── Roboter-State lesen ────────────────────────────────────────────
        joint_pos_deg, _ = read_robot_state(robot)

        if prev_joint_pos_deg is not None:
            joint_vel_deg_s = (joint_pos_deg - prev_joint_pos_deg) / dt
        else:
            joint_vel_deg_s = np.zeros(6)
        prev_joint_pos_deg = joint_pos_deg.copy()

        print("Joint velocity in degree")
        print(joint_vel_deg_s)

        # ── Observation aufbauen ───────────────────────────────────────────
        obs = obs_builder.build(
            joint_pos_deg=joint_pos_deg,
            joint_vel_deg_s=joint_vel_deg_s,
            initial_object_pos_robot_frame=object_pos,  # fest, kein Tracking nötig
            bowl_pos_robot_frame=bowl_pos,
        )

        print("Observations after processing")
        print(obs)

        # ── Policy-Inferenz ────────────────────────────────────────────────
        with torch.no_grad():
            raw_action = policy_nn.act_inference(obs)
            raw_action = torch.clamp(raw_action, -1.0, 1.0)
            raw_action_np = raw_action.squeeze(0).cpu().numpy()

            print("RAW ACTION")
            print(raw_action_np)

        # ── Sicherheitscheck: NaN oder explodierte Werte ──────────────────
        if np.any(np.isnan(raw_action_np)) or np.any(np.isinf(raw_action_np)):
            log.error("ABBRUCH: Policy-Output enthält NaN/Inf!")
            log.error(f"  Raw action: {raw_action_np}")
            log.error("  Mögliche Ursachen:")
            log.error("  1. Obs-Normalisierung fehlt im Checkpoint")
            log.error("  2. last_action aus vorherigem explodierten Step propagiert")
            log.error("  3. Obs-Vektor-Reihenfolge stimmt nicht mit Training überein")
            break
        if np.any(np.abs(raw_action_np) > ACTION_EXPLOSION_THRESHOLD):
            log.error(f"ABBRUCH: Policy-Output explodiert! max(|a|)={np.abs(raw_action_np).max():.2e}")
            log.error(f"  Raw action: {raw_action_np}")
            log.error("  Obs-Normalisierung pruefen (empirical_normalization im Checkpoint?)")
            break

        # ── Action interpretieren und senden ──────────────────────────────
        result = action_interp.interpret(raw_action_np, joint_pos_deg)

        print("Interpreted Actions")
        print(result)

        send_action_to_robot(robot, result["arm_targets_deg"], result["gripper_cmd_rad"])

        obs_builder.update_last_action(raw_action_np)

        step += 1
        if step % CONTROL_HZ == 0:
            log.info(
                f"  t={step/CONTROL_HZ:.1f}s | "
                f"arm_pos={joint_pos_deg[:5].round(1)} | "
                f"arm_tgt={result['arm_targets_deg'].round(1)} | "
                f"gripper_rad={result['gripper_cmd_rad']:.3f}"
            )

        elapsed = time.perf_counter() - t0
        busy_wait(max(0.0, dt - elapsed))

    log.info("Episode beendet.")


class _MockSOFollower:
    """Minimaler Mock-Roboter für --mock-Modus."""
    is_connected = True

    def connect(self, **kwargs): pass
    def disconnect(self): pass

    def get_observation(self) -> dict:
        obs = {f"{m}.pos": 0.0 for m in ARM_JOINT_NAMES}
        obs["gripper.pos"] = 50.0
        return obs

    def send_action(self, action: dict) -> dict:
        return action


def main():
    parser = argparse.ArgumentParser(
        description="Deployt die RSL-RL Lift-Policy auf den SO101."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Pfad zur .pt Checkpoint-Datei"
    )
    parser.add_argument("--robot_port", default="COM5")
    parser.add_argument("--robot_id", default="follower_arm_v2")
    parser.add_argument(
        "--robot_type", default="so101_follower",
        choices=["so101_follower", "so100_follower"],
    )
    parser.add_argument(
        "--urdf_path", required=True,
        help="Pfad zur so_arm101.urdf (für Pinocchio FK)"
    )
    parser.add_argument(
        "--object_pos", nargs=3, type=float, default=[0.30, 0.20, 0.01],
        metavar=("X", "Y", "Z"),
        help="Initiale Würfel-Position im Robot-Frame in Metern (bleibt konstant)"
    )
    parser.add_argument(
        "--bowl_pos", nargs=3, type=float, default=[0.30, 0.10, 0.00],
        metavar=("X", "Y", "Z"),
        help="Bowl-Position im Robot-Frame in Metern (bleibt konstant)"
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument(
        "--episode_duration", type=float, default=30.0,
        help="Episodendauer in Sekunden"
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
        help="Mock-Modus: kein echter Roboter"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SO101 Lift Policy Deployment")
    log.info("=" * 60)
    log.info(f"Checkpoint  : {args.checkpoint}")
    log.info(f"URDF        : {args.urdf_path}")
    log.info(f"Device      : {args.device}")
    log.info(f"Objekt-Pos  : {args.object_pos} m (Robot-Frame, initial/fest)")
    log.info(f"Bowl-Pos    : {args.bowl_pos} m (Robot-Frame, fest)")
    log.info(f"Frequenz    : {CONTROL_HZ} Hz")
    log.info(f"Max-Delta   : {args.max_delta_deg}°/step")
    log.info(f"Obs-Dim     : {NUM_OBS} (6+6+3+3+3+6)")
    log.info(f"Action-Dim  : {NUM_ACTIONS}")
    log.info(f"Action Scale: {ARM_ACTION_SCALE} (arm+gripper)")
    log.info(f"Default-Pos : {np.rad2deg(DEFAULT_JOINT_POS_RAD).round(1)}°")

    # ── FK-Modell laden ────────────────────────────────────────────────────
    pin_model, pin_data, j_idx, gripper_frame_id = build_fk_model(args.urdf_path)

    policy_nn = load_rsl_rl_policy(args.checkpoint, args.device)

    obs_builder = ObservationBuilder(
        pin_model=pin_model,
        pin_data=pin_data,
        j_idx=j_idx,
        gripper_frame_id=gripper_frame_id,
        device=args.device,
    )
    action_interp = ActionInterpreter(max_delta_deg=args.max_delta_deg)

    if args.mock:
        robot = _MockSOFollower()
        log.warning("MOCK-MODUS aktiv – kein echter Roboter wird bewegt!")
    else:
        robot_cls = SO101Follower if args.robot_type == "so101_follower" else SO100Follower
        robot_cfg = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
        robot = robot_cls(robot_cfg)
    robot.connect()
    log.info(f"Roboter verbunden: {args.robot_type} @ {args.robot_port}")

    # In Default-Position fahren bevor Policy startet
    if not args.mock:
        move_to_default(robot)
        log.info("Warte 2s vor Policy-Start ...")
        time.sleep(2.0)

    object_pos = np.array(args.object_pos, dtype=np.float32)
    bowl_pos   = np.array(args.bowl_pos,   dtype=np.float32)

    try:
        for ep in range(args.num_episodes):
            log.info(f"\n{'─'*50}")
            log.info(f"Episode {ep + 1} / {args.num_episodes}")
            log.info(f"{'─'*50}")

            run_episode(
                robot=robot,
                policy_nn=policy_nn,
                obs_builder=obs_builder,
                action_interp=action_interp,
                object_pos=object_pos,
                bowl_pos=bowl_pos,
                episode_duration_s=args.episode_duration,
                device=args.device,
            )

            if not args.mock:
                move_to_rest(robot)

            if ep < args.num_episodes - 1:
                log.info(f"\nSzene resetten – {args.reset_duration:.0f}s Pause ...")
                log.info("→ Würfel zurückstellen, Roboter in Startposition bringen.")
                time.sleep(args.reset_duration)

    except KeyboardInterrupt:
        log.info("\nDurch Benutzer abgebrochen.")
    finally:
        robot.disconnect()
        log.info("Roboter getrennt.")


if __name__ == "__main__":
    main()