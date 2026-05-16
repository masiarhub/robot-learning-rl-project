import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

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
def move_to_target(
    robot,
    target: dict,
    steps: int = 50,
    step_delay: float = 0.02,
):
    """
    Interpoliert von der aktuellen Position
    zur Zielposition.
    """

    obs = robot.get_observation()

    start = {
        k: obs[k]
        for k in JOINT_NAMES
    }

    print("\n" + "=" * 60)
    print("Neue Bewegung")
    print("=" * 60)

    print("\nStart:")
    for k, v in start.items():
        print(f"  {k:<20} = {v:>8.2f}°")

    print("\nTarget:")
    for k, v in target.items():
        print(f"  {k:<20} = {v:>8.2f}°")

    for i in range(1, steps + 1):

        alpha = i / steps

        action = {
            joint: start[joint]
            + alpha * (target[joint] - start[joint])
            for joint in JOINT_NAMES
        }

        robot.send_action(action)

        time.sleep(step_delay)

    print("\nZielposition erreicht.")

# ──────────────────────────────────────────
# Roboter verbinden
# ──────────────────────────────────────────
robot_cfg = SO101FollowerConfig(
    port=ROBOT_PORT,
    id=ROBOT_ID,
)

robot = SO101Follower(robot_cfg)

robot.connect()

print("ROBOT CONNECTED")

# ──────────────────────────────────────────
# Zielpositionen
# ──────────────────────────────────────────

pose_1 = {
    'elbow_flex.pos': 31.604395604395606,
    'gripper.pos': 33.657182512144345,
    'shoulder_lift.pos': 13.626373626373626,
    'shoulder_pan.pos': -2.10989010989011,
    'wrist_flex.pos': 71.12087912087912,
    'wrist_roll.pos': -95.34065934065934,
}

pose_2 = {
    'elbow_flex.pos': 31.516483516483518,
    'gripper.pos': 10.0,
    'shoulder_lift.pos': 13.626373626373626,
    'shoulder_pan.pos': -2.021978021978022,
    'wrist_flex.pos': 72.08791208791209,
    'wrist_roll.pos': -95.51648351648352,
}

pose_3 = {'elbow_flex.pos': 22.10989010989011,
 'gripper.pos': 10.0,
 'shoulder_lift.pos': 2.6373626373626373,
 'shoulder_pan.pos': -76.3956043956044,
 'wrist_flex.pos': 14.76923076923077,
 'wrist_roll.pos': -95.6043956043956}

pose_4 = {'elbow_flex.pos': 22.10989010989011,
 'gripper.pos': 10.964607911172797,
 'shoulder_lift.pos': 2.6373626373626373,
 'shoulder_pan.pos': -76.3956043956044,
 'wrist_flex.pos': 14.76923076923077,
 'wrist_roll.pos': -95.6043956043956}

pose_5 = {'elbow_flex.pos': 98.24175824175825,
 'gripper.pos': 38.65371269951422,
 'shoulder_lift.pos': -96.96703296703296,
 'shoulder_pan.pos': 4.395604395604396,
 'wrist_flex.pos': 63.824175824175825,
 'wrist_roll.pos': -95.6043956043956}

# ──────────────────────────────────────────
# Bewegungssequenz
# ──────────────────────────────────────────

sequence = [
    (pose_1, 100),  # langsam
    (pose_2, 20),   # schnell
    (pose_3, 100),  # langsam
    (pose_4, 20),   # schnell
    (pose_5, 100),  # langsam
]

for idx, (pose, steps) in enumerate(sequence, start=1):

    print(f"\n\n===== POSE {idx} =====")

    move_to_target(
        robot,
        pose,
        steps=steps,
        step_delay=0.02,
    )

    print("Halte Position für 2 Sekunden...")
    time.sleep(2)

# ──────────────────────────────────────────
# Disconnect
# ──────────────────────────────────────────

robot.disconnect()

print("\nROBOT DISCONNECTED")