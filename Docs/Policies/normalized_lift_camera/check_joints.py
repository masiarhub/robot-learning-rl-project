"""Live joint monitor for the SO-101.

Two modes:
  default  – print once, show deviation from Isaac Lab default, and output the
              LEROBOT_DEFAULT_DEG line to paste into deploy_script.py.
  --live   – stream readings at ~10 Hz so you can move joints by hand and verify
              that JOINT_SIGN is correct for each motor.

Isaac Lab default joint positions (rad):
  shoulder_pan=0.0, shoulder_lift=-0.6, elbow_flex=-0.6, wrist_flex=1.57, wrist_roll=-1.57

How to verify JOINT_SIGN in --live mode:
  1. Start the script with the arm at the default pose.
  2. Slowly rotate ONE joint in the positive LeRobot direction (encoder value increases).
  3. Watch the 'isaac_rad' column for that joint.
     - If isaac_rad INCREASES  → JOINT_SIGN for that joint is +1  (correct if +1 is set)
     - If isaac_rad DECREASES  → JOINT_SIGN for that joint is -1  (correct if -1 is set)
  4. If the sign in the table disagrees with what you observe, flip it in deploy_script.py.
"""
import argparse
import time
import numpy as np
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--robot_port", default="/dev/ttyACM0")
parser.add_argument("--robot_id",   default="follower_arm")
parser.add_argument("--live", action="store_true",
                    help="Stream readings continuously for sign verification")
args = parser.parse_args()

DEFAULT_RAD        = np.array([0.0, -0.6, -0.6,  1.57, -1.57])
JOINT_SIGN         = np.array([1.0,  1.0, -1.0,  1.0,   1.0])
LEROBOT_DEFAULT_DEG = np.array([0.62, -49.32, 43.47, 88.44, -83.38])
ARM                = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Joints whose signs have NOT been verified by movement — watch these carefully in --live mode
UNVERIFIED = {"shoulder_pan", "shoulder_lift", "wrist_roll"}

cfg   = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
robot = SO101Follower(cfg)
robot.connect(calibrate=False)


def read_and_print(prev_isaac=None):
    obs      = robot.get_observation()
    read_deg = np.array([float(obs[f"{m}.pos"]) for m in ARM])
    isaac_rad = DEFAULT_RAD + JOINT_SIGN * np.deg2rad(read_deg - LEROBOT_DEFAULT_DEG)

    if args.live:
        print("\033[H\033[J", end="")  # clear terminal
        print("Live joint monitor  (Ctrl-C to quit)\n")
        print(f"  {'Motor':<16} {'lerobot_deg':>11}  {'isaac_rad':>9}  {'Δisaac':>7}  {'sign_set':>8}  {'verified':>8}")
        print("  " + "-" * 72)
        for i, m in enumerate(ARM):
            delta = isaac_rad[i] - prev_isaac[i] if prev_isaac is not None else 0.0
            sign_char = f"{'+' if JOINT_SIGN[i] > 0 else '-'}1"
            verified  = "NO ←" if m in UNVERIFIED else "yes"
            arrow = ("↑" if delta > 0.005 else "↓" if delta < -0.005 else " ")
            print(f"  {m:<16} {read_deg[i]:>11.2f}  {isaac_rad[i]:>9.3f}  {arrow}{abs(delta):>6.3f}  {sign_char:>8}  {verified:>8}")
        gripper_pct = float(obs["gripper.pos"])
        print(f"\n  {'gripper':<16} {gripper_pct:>11.1f}%")
        print("\nMove one joint at a time. When lerobot_deg increases,")
        print("isaac_rad should increase (↑) if sign=+1, decrease (↓) if sign=-1.")
    else:
        dev = isaac_rad - DEFAULT_RAD
        print(f"\n{'Motor':<16} {'lerobot_deg':>11}  {'isaac_rad':>9}  {'default':>9}  {'deviation':>9}")
        print("-" * 60)
        for i, m in enumerate(ARM):
            flag = "  ← large deviation" if abs(dev[i]) > 0.05 else ""
            print(f"{m:<16} {read_deg[i]:>11.2f}  {isaac_rad[i]:>9.3f}  {DEFAULT_RAD[i]:>9.3f}  {dev[i]:>9.3f}{flag}")
        gripper_pct = float(obs["gripper.pos"])
        print(f"\n{'gripper':<16} {gripper_pct:>11.1f}%")
        print(f"""
→ Paste into deploy_script.py if you just measured the default pose:
  LEROBOT_DEFAULT_DEG = np.array([{read_deg[0]:.2f}, {read_deg[1]:.2f}, {read_deg[2]:.2f}, {read_deg[3]:.2f}, {read_deg[4]:.2f}])
""")

    return isaac_rad.copy()


try:
    if args.live:
        prev = read_and_print(None)
        while True:
            time.sleep(0.1)
            prev = read_and_print(prev)
    else:
        read_and_print(None)
except KeyboardInterrupt:
    pass
finally:
    robot.disconnect()
