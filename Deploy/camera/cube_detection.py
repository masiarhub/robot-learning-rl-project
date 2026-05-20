"""
Cube detection script — Task 1 VisualCoord
Loads an image, detects the target cube via HSV segmentation,
and returns [u, v, visible] in NDC coordinates as expected by the policy.

Usage:
    python cube_detection.py --image image_0.png --color red
    python cube_detection.py --image image_0.png --color blue
"""

import cv2
import numpy as np
import argparse
import math

# ============================================================
# FOV CONSTANTS (measured / from sim)
# Update these once you re-measure at 720p!
# ============================================================
TAN_HALF_HFOV_SIM  = 1.0691   # from _wrist_cam.py
TAN_HALF_VFOV_SIM  = 0.6014   # from _wrist_cam.py

# Your measured FOV at 640x480 — re-measure at 720p and update!
HFOV_REAL_DEG = 83.54
VFOV_REAL_DEG = 65.92

SCALE_U = math.tan(math.radians(HFOV_REAL_DEG / 2)) / TAN_HALF_HFOV_SIM  # ≈ 0.835
SCALE_V = math.tan(math.radians(VFOV_REAL_DEG / 2)) / TAN_HALF_VFOV_SIM  # ≈ 1.079

# ============================================================
# HSV COLOUR PALETTE
# Sim sRGB values widened ±10-15 hue, ±40-60 sat/val for real light
# Index: 0=blue, 1=red, 2=green, 3=yellow, 4=purple, 5=orange
# ============================================================
HSV_RANGES = {
    "blue": [
        ((100,  80,  60), (135, 255, 255)),
    ],
    "red": [
        # Red wraps around 0/180 — two ranges needed
        ((  0,  80,  80), ( 10, 255, 255)),
        ((170,  80,  80), (180, 255, 255)),
    ],
    "green": [
        ((40,  60,  60), (85, 255, 255)),
    ],
    "yellow": [
        ((20,  80,  80), (35, 255, 255)),
    ],
    "purple": [
        ((130,  50,  50), (160, 255, 255)),
    ],
    "orange": [
        ((10,  80,  80), (20, 255, 255)),
    ],
}

COLOR_INDEX = {
    "blue": 0, "red": 1, "green": 2,
    "yellow": 3, "purple": 4, "orange": 5,
}

MIN_BLOB_AREA_PX = 20   # blobs smaller than this are noise


# ============================================================
# CORE FUNCTION (matches deployment interface exactly)
# ============================================================
def get_cube_image_coords(frame_bgr, color_name,
                          scale_u=SCALE_U, scale_v=SCALE_V,
                          min_blob_area_px=MIN_BLOB_AREA_PX):
    """
    Returns np.ndarray of shape (3,): [u, v, visible]
      u, v  : NDC in [-1, 1], corrected for real vs sim FOV
      visible: 1.0 if cube found, 0.0 otherwise

    u = 0, v = 0  is the image centre.
    u positive    = right,  u negative = left
    v positive    = up,     v negative = down  (y-axis flipped vs pixels)
    """
    H, W = frame_bgr.shape[:2]
    hsv   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Build mask — combine multiple ranges (needed for red)
    ranges = HSV_RANGES[color_name]
    mask   = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    # Find largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0]), mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_blob_area_px:
        return np.array([0.0, 0.0, 0.0]), mask

    # Centroid in pixel space
    M  = cv2.moments(c)
    px = M["m10"] / M["m00"]   # x: 0 = left edge
    py = M["m01"] / M["m00"]   # y: 0 = top edge

    # NDC conversion
    #   u: left=-1, right=+1
    #   v: bottom=-1, top=+1  (flip y — pixels go top-down, NDC bottom-up)
    u_ndc = (px - W / 2) / (W / 2)
    v_ndc = (H / 2 - py) / (H / 2)

    # FOV correction
    u = float(np.clip(u_ndc * scale_u, -1.0, 1.0))
    v = float(np.clip(v_ndc * scale_v, -1.0, 1.0))

    return np.array([u, v, 1.0]), mask


def make_color_one_hot(color_name):
    vec = np.zeros(6, dtype=np.float32)
    vec[COLOR_INDEX[color_name]] = 1.0
    return vec


# ============================================================
# DEBUG VISUALISATION
# ============================================================
def draw_debug(frame_bgr, mask, coords, color_name):
    u, v, visible = coords
    H, W = frame_bgr.shape[:2]

    vis = frame_bgr.copy()

    # HSV mask overlay (semi-transparent green tint)
    tint = np.zeros_like(vis)
    tint[mask > 0] = (0, 200, 0)
    vis = cv2.addWeighted(vis, 1.0, tint, 0.35, 0)

    if visible > 0:
        # Convert NDC back to pixel for drawing
        # Inverse of: u_ndc = (px - W/2)/(W/2)  and scale_u correction
        u_ndc = u / SCALE_U
        v_ndc = v / SCALE_V
        cx = int((u_ndc + 1) / 2 * W)
        cy = int((1 - v_ndc) / 2 * H)

        # Cross marker
        arm = 18
        cv2.line(vis, (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
        cv2.line(vis, (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        label = f"u={u:+.3f}  v={v:+.3f}  vis=1"
        cv2.putText(vis, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "NOT VISIBLE", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Color name + one-hot
    oh = make_color_one_hot(color_name)
    cv2.putText(vis, f"color: {color_name}  one-hot: {oh.astype(int).tolist()}",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # FOV correction info
    cv2.putText(vis, f"scale_u={SCALE_U:.3f}  scale_v={SCALE_V:.3f}",
                (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)

    return vis


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cube HSV detection — VisualCoord")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--color", required=True,
                        choices=list(HSV_RANGES.keys()),
                        help="Target cube colour")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip cv2.imshow (headless mode)")
    parser.add_argument("--save", default=None,
                        help="Save debug image to this path (e.g. debug.png)")
    args = parser.parse_args()

    # Load image
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    H, W = frame.shape[:2]
    print(f"Image loaded: {W}x{H}  |  color: {args.color}")
    print(f"FOV correction: scale_u={SCALE_U:.4f}, scale_v={SCALE_V:.4f}")

    # Detection
    coords, mask = get_cube_image_coords(frame, args.color)
    u, v, visible = coords

    # Console output
    print("─" * 40)
    print(f"  visible : {int(visible)}")
    print(f"  u       : {u:+.4f}   (−1=left, +1=right)")
    print(f"  v       : {v:+.4f}   (−1=bottom, +1=top)")
    print(f"  obs vec : {coords.tolist()}")
    print(f"  one-hot : {make_color_one_hot(args.color).tolist()}")
    print("─" * 40)

    # Debug visualisation
    debug_img = draw_debug(frame, mask, coords, args.color)

    if args.save:
        cv2.imwrite(args.save, debug_img)
        print(f"Debug image saved: {args.save}")

    if not args.no_display:
        cv2.imshow("Cube detection — press any key to close", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()