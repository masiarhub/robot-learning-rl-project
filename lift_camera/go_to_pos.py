import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np

# ──────────────────────────────────────────
# Konfiguration
# ──────────────────────────────────────────
ROBOT_PORT = "COM5"
ROBOT_ID   = "follower_arm"

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# ──────────────────────────────────────────
# Hilfsfunktion: sanft zur Zielposition fahren
# ──────────────────────────────────────────
def move_to_target(robot, target: dict, steps: int = 50, step_delay: float = 0.02):
    """
    Interpoliert von der aktuellen Position zur Zielposition
    in `steps` gleichmässigen Schritten.
    step_delay = Pause zwischen zwei Schritten in Sekunden (0.02 s → 50 Hz)
    """
    # Aktuelle Position auslesen
    obs = robot.get_observation()
    start = {k: obs[k] for k in JOINT_NAMES}

    print(f"\nStart:  { {k: round(v,1) for k,v in start.items()} }")
    print(f"Target: { {k: round(v,1) for k,v in target.items()} }")
    print(f"Fahre in {steps} Schritten à {step_delay*1000:.0f} ms ...\n")

    for i in range(1, steps + 1):
        alpha = i / steps  # 0.02 → 0.04 → ... → 1.0

        # Lineares Interpolieren jedes Joints
        action = {
            joint: start[joint] + alpha * (target[joint] - start[joint])
            for joint in JOINT_NAMES
        }

        robot.send_action(action)
        time.sleep(step_delay)

    print("Zielposition erreicht.")


# ──────────────────────────────────────────
# Roboter verbinden
# ──────────────────────────────────────────
robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID)
robot = SO101Follower(robot_cfg)
robot.connect()
print("ROBOT CONNECTED")

# Aktuelle Position anzeigen
obs = robot.get_observation()
print("\nAktuelle Position:")
for j in JOINT_NAMES:
    print(f"  {j:<20} = {obs[j]:.1f}°")


# ──────────────────────────────────────────
# Zielposition definieren  ← hier anpassen!
# ──────────────────────────────────────────

DEFAULT_JOINT_POS_RAD = np.array([
     0.00,   # shoulder_pan
    -0.60,   # shoulder_lift
    -0.60,   # elbow_flex
     1.57,   # wrist_flex
    -1.57,   # wrist_roll
     0.00,   # gripper
], dtype=np.float64)

# Rad → Deg
DEFAULT_JOINT_POS_DEG = np.degrees(DEFAULT_JOINT_POS_RAD)

print(DEFAULT_JOINT_POS_DEG)

target_position = {
    "shoulder_pan.pos":  DEFAULT_JOINT_POS_DEG[0],
    "shoulder_lift.pos": DEFAULT_JOINT_POS_DEG[1],
    "elbow_flex.pos":    DEFAULT_JOINT_POS_DEG[2],
    "wrist_flex.pos":    DEFAULT_JOINT_POS_DEG[3],
    "wrist_roll.pos":    DEFAULT_JOINT_POS_DEG[4],
    "gripper.pos":       DEFAULT_JOINT_POS_DEG[5],
}



# target_position = {
#     "shoulder_pan.pos":  0,       # unverändert
#     "shoulder_lift.pos": 0,       # unverändert
#     "elbow_flex.pos":    0,          # unverändert
#     "wrist_flex.pos":    0,    # 30° runter
#     "wrist_roll.pos":    0,          # unverändert
#     "gripper.pos":       0,             # unverändert
# }

# ──────────────────────────────────────────
# Bewegung ausführen
# steps=50, delay=0.02s → ~1 Sekunde Gesamtdauer
# steps=100, delay=0.02s → ~2 Sekunden (langsamer)
# ──────────────────────────────────────────
move_to_target(robot, target_position, steps=100, step_delay=0.02)

# 3 Sekunden in Zielposition halten
print("Halte Position für 3 Sekunden...")
time.sleep(3)

robot.disconnect()
print("ROBOT DISCONNECTED")