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
ROBOT_ID = "follower_arm"

CAMERA_ID = 1

PRINT_HZ = 5


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened")
# Setzen der kamera auflösung
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("CAMERA CONNECTED")


# ============================================================
# LOOP
# ============================================================

dt = 1.0 / PRINT_HZ

# Zähler für gespeicherte Bilder
image_counter = 0

try:

    while True:

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

        # ----------------------------------------------------
        # ENTER -> Bild speichern
        # ----------------------------------------------------

        if key == 13 or key == 10:  # Enter-Taste

            filename = f"image_{image_counter}.png"

            success = cv2.imwrite(filename, frame)

            if success:
                print(f"Bild gespeichert: {filename}")
                image_counter += 1
            else:
                print("Fehler beim Speichern")

        # ----------------------------------------------------
        # q -> Programm beenden
        # ----------------------------------------------------

        if key == ord("q"):
            break

        time.sleep(dt)

except KeyboardInterrupt:

    print("\nSTOPPED")

finally:

    # Kamera freigeben
    cap.release()

    # Fenster schließen
    cv2.destroyAllWindows()

    print("CLEAN EXIT")