"""Capture a snapshot from each available camera and save it as camera_<idx>.jpg."""
import cv2

found = []
for idx in range(5):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        continue
    # discard a few frames so the sensor has time to adjust
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue
    path = f"camera_{idx}.jpg"
    cv2.imwrite(path, frame)
    print(f"Camera {idx}: saved {frame.shape[1]}x{frame.shape[0]} → {path}")
    found.append(idx)

if not found:
    print("No cameras found.")
else:
    print(f"\nOpen the saved .jpg files to see which index is the wrist cam.")
    print(f"Then pass --camera_device <idx> to deploy_script.py")
