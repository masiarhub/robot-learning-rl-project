"""Print live joint readings from the SO-101 and estimate deviation from Isaac Lab default.

Run with the arm physically at the Isaac Lab default pose, then copy the read_deg
column into LEROBOT_DEFAULT_DEG in deploy_script.py.

Isaac Lab default:
  shoulder_pan=0.0, shoulder_lift=-0.6, elbow_flex=-0.6, wrist_flex=1.57, wrist_roll=-1.57 rad
"""
import argparse
import numpy as np
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--robot_port", default="/dev/ttyACM0")
parser.add_argument("--robot_id",   default="follower_arm")
args = parser.parse_args()

cfg = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
robot = SO101Follower(cfg)
robot.connect(calibrate=False)
obs = robot.get_observation()
robot.disconnect()

DEFAULT_RAD  = np.array([0.0, -0.6, -0.6,  1.57, -1.57])
JOINT_SIGN   = np.array([1.0,  1.0, -1.0, 1.0,   1.0])
ARM          = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Estimate Isaac Lab position using the same formula as deploy_script.py
# (assumes LEROBOT_DEFAULT_DEG = read_deg, i.e. arm IS at the default pose).
# Deviation from Isaac Lab default should be ~0 for all joints if positioned correctly.
read_deg = np.array([float(obs[f"{m}.pos"]) for m in ARM])
# With LEROBOT_DEFAULT_DEG set to read_deg, isaac_rad = DEFAULT_RAD (by definition).
# Below we just show what the script computes with LEROBOT_DEFAULT_DEG = 0 so you
# can verify the numbers make sense before filling them in.
lerobot_default_guess = np.zeros(5)
isaac_rad_estimated = DEFAULT_RAD + JOINT_SIGN * np.deg2rad(read_deg - lerobot_default_guess)

print(f"\n{'Motor':<16} {'read_deg':>10}  {'isaac_rad_est':>13}  {'default_rad':>11}  {'deviation':>9}")
print("-" * 68)
for i, m in enumerate(ARM):
    dev = isaac_rad_estimated[i] - DEFAULT_RAD[i]
    flag = "  ← set LEROBOT_DEFAULT_DEG to read_deg column" if abs(dev) > 0.05 else "  ✓"
    print(f"{m:<16} {read_deg[i]:>10.2f}  {isaac_rad_estimated[i]:>13.3f}  {DEFAULT_RAD[i]:>11.3f}  {dev:>9.3f}{flag}")

gripper_pct = float(obs["gripper.pos"])
print(f"\n{'gripper':<16} {gripper_pct:>10.1f}%")

print(f"""
→ Copy these values into LEROBOT_DEFAULT_DEG in deploy_script.py:
  LEROBOT_DEFAULT_DEG = np.array([{read_deg[0]:.2f}, {read_deg[1]:.2f}, {read_deg[2]:.2f}, {read_deg[3]:.2f}, {read_deg[4]:.2f}])
""")
