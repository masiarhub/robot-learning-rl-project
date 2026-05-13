"""Print the loaded calibration for the SO-101 without connecting to hardware."""
import argparse
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

parser = argparse.ArgumentParser()
parser.add_argument("--robot_port", default="/dev/ttyACM0")
parser.add_argument("--robot_id",   default="my_so101")
args = parser.parse_args()

cfg = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
robot = SO101Follower(cfg)   # loads calibration from file — no hardware connection yet

print(f"Calibration file: {robot.calibration_fpath}")
print(f"File exists     : {robot.calibration_fpath.is_file()}")
print(f"Is calibrated   : {bool(robot.calibration)}\n")

if not robot.calibration:
    print("No calibration found. Run the deploy script once (without --mock) to calibrate.")
else:
    print(f"{'Motor':<16} {'ID':>4}  {'drive_mode':>10}  {'homing_offset':>13}  {'range_min':>9}  {'range_max':>9}")
    print("-" * 70)
    for motor, cal in robot.calibration.items():
        print(
            f"{motor:<16} {cal.id:>4}  {cal.drive_mode:>10}  "
            f"{cal.homing_offset:>13}  {cal.range_min:>9}  {cal.range_max:>9}"
        )
