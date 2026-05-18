import time
from pprint import pprint
import numpy as np

import cv2

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


# ============================================================
# CONFIG
# ============================================================

ROBOT_PORT = "COM5"
ROBOT_ID = "follower_arm_v2"

CAMERA_ID = 1



PRINT_HZ = 5

# EE-Offset in gripper_link-lokalem Frame (aus env.yaml / joint_pos_env_cfg.py)
#EE_OFFSET = np.array([0.01, 0.0, -0.09])  # metres
EE_OFFSET = np.array([0.0, 0.0, 0.0])

URDF_PATH = "C:/Users/manue/Documents/Studium/Master/Semester2/Robot learning/Groupe_project/TeamRepo/robot-learning-rl-project/Deploy/No_Wrist_Cam/so_arm101.urdf"



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

    T = pin_data.oMf[gripper_frame_id]          # SE3: base_link → gripper_link
    ee_pos = T.translation + T.rotation @ EE_OFFSET  # (3,) in base frame
    return ee_pos.astype(np.float32)

# ============================================================
# ROBOT
# ============================================================

robot_cfg = SOFollowerRobotConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
)

robot = SO101Follower(robot_cfg)
robot.connect()

print("ROBOT CONNECTED")


try:

    

    # ----------------------------------------------------
    # ROBOT OBS
    # ----------------------------------------------------

    obs = robot.get_observation()
    

    # ── FK-Modell laden ────────────────────────────────────────────────────
    pin_model, pin_data, j_idx, gripper_frame_id = build_fk_model(URDF_PATH)

    # ── EE-Position via FK ────────────────────────────────────────────

    GRIPPER_RAD_MIN = -0.1
    GRIPPER_RAD_MAX =  0.5

    gripper_pct = obs["gripper.pos"]
    gripper_rad = GRIPPER_RAD_MIN + (gripper_pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)

    q_abs = {
        "shoulder_pan":  np.deg2rad(obs["shoulder_pan.pos"]),
        "shoulder_lift": np.deg2rad(obs["shoulder_lift.pos"]),
        "elbow_flex":    np.deg2rad(obs["elbow_flex.pos"]),
        "wrist_flex":    np.deg2rad(obs["wrist_flex.pos"]),
        "wrist_roll":    np.deg2rad(obs["wrist_roll.pos"]),
        "gripper":       gripper_rad,  # ← direkt rad, kein deg2rad
    }

    pprint(q_abs)

    # q_abs = {
    #     "shoulder_pan":  0.0,
    #     "shoulder_lift": 0.0,
    #     "elbow_flex":    0.0,
    #     "wrist_flex":    0.0,
    #     "wrist_roll":    0.0,
    #     "gripper":       0.0,  # ← direkt rad, kein deg2rad
    # }
    
    ee_pos = get_ee_pos(
        q_abs,
        pin_model,
        pin_data,
        j_idx,
        gripper_frame_id,
        )

    pprint(ee_pos)


except KeyboardInterrupt:

    print("\nSTOPPED")

finally:

    robot.disconnect()

    print("CLEAN EXIT")