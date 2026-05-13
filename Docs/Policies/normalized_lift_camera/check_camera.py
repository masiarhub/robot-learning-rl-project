"""Capture a snapshot from each available camera and save debug images.

Saves two files per camera:
  camera_<idx>.jpg          – full native resolution
  camera_<idx>_policy.jpg   – center-cropped to 16:9, resized to 128x72
                               (exactly what the policy sees)
"""
import cv2

POLICY_W = 128
POLICY_H = 72
TRAIN_ASPECT = POLICY_W / POLICY_H  # 1.778  (16:9)


def center_crop_16_9(frame):
    """Center-crop frame to 16:9 aspect ratio."""
    h, w = frame.shape[:2]
    target_w = int(h * TRAIN_ASPECT)
    target_h = int(w / TRAIN_ASPECT)
    if target_w <= w:
        x0 = (w - target_w) // 2
        return frame[:, x0:x0 + target_w]
    else:
        y0 = (h - target_h) // 2
        return frame[y0:y0 + target_h, :]

found = []
for idx in range(5):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        continue
    # MJPG lets most cameras reach 1280x720 even when raw YUV tops out at 640x480
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    # discard a few frames so the sensor has time to adjust
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        continue

    h, w = frame.shape[:2]
    native_aspect = w / h
    aspect_ok = abs(native_aspect - TRAIN_ASPECT) < 0.05

    # full-res save
    path_full = f"camera_{idx}.jpg"
    cv2.imwrite(path_full, frame)

    # center-crop to 16:9 then resize — same pipeline as deploy_script.py
    crop = center_crop_16_9(frame)
    policy_frame = cv2.resize(crop, (POLICY_W, POLICY_H), interpolation=cv2.INTER_LINEAR)
    path_policy = f"camera_{idx}_policy.jpg"
    cv2.imwrite(path_policy, policy_frame)

    ch, cw = crop.shape[:2]
    aspect_note = "OK" if aspect_ok else f"was {native_aspect:.2f} — cropped to {cw}x{ch} ({TRAIN_ASPECT:.2f})"
    print(f"Camera {idx}: native {w}x{h}  aspect={aspect_note}")
    print(f"  saved full-res  → {path_full}")
    print(f"  saved policy-res ({POLICY_W}x{POLICY_H}, center-cropped) → {path_policy}")
    found.append(idx)

if not found:
    print("No cameras found.")
else:
    print(f"\nCheck the *_policy.jpg files — that is exactly what the neural network sees.")
    print(f"Expected view: looking slightly downward from the gripper, cube and table in lower half.")
    print(f"Pass --camera_device <idx> to deploy_script.py once you identified the wrist cam.")
