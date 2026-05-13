import time
from pprint import pprint

import cv2

from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig


# ============================================================
# CONFIG
# ============================================================

ROBOT_PORT = "COM5"
ROBOT_ID = "follower_arm"

CAMERA_ID = 1

PRINT_HZ = 0.5


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


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")

print("CAMERA CONNECTED")


# ============================================================
# LOOP
# ============================================================

dt = 1.0 / PRINT_HZ

try:

    while True:

        # ----------------------------------------------------
        # ROBOT OBS
        # ----------------------------------------------------

        obs = robot.get_observation()

        print("\n" + "=" * 60)

        pprint(obs)

        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame")
            continue

        # Bild anzeigen
        cv2.imshow("SO101 Wrist Camera", frame)

        # Wichtig für OpenCV Fenster
        key = cv2.waitKey(1)

        # q zum Beenden
        if key == ord("q"):
            break

        time.sleep(dt)

except KeyboardInterrupt:

    print("\nSTOPPED")

finally:

    cap.release()

    cv2.destroyAllWindows()

    robot.disconnect()

    print("CLEAN EXIT")