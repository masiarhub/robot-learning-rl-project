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
# FOV CONSTANTS (measured at 1280x720)
# ============================================================
TAN_HALF_HFOV_SIM  = 1.0691   # from _wrist_cam.py
TAN_HALF_VFOV_SIM  = 0.6014   # from _wrist_cam.py

HFOV_REAL_DEG = 100.82   # gemessen bei 1280x720
VFOV_REAL_DEG =  64.80   # gemessen bei 1280x720

SCALE_U = math.tan(math.radians(HFOV_REAL_DEG / 2)) / TAN_HALF_HFOV_SIM  # ≈ 1.132
SCALE_V = math.tan(math.radians(VFOV_REAL_DEG / 2)) / TAN_HALF_VFOV_SIM  # ≈ 1.059

# Anzeigegrösse für Debug-Fenster (nur Visualisierung, u/v nicht betroffen)
DISPLAY_WIDTH = 960

# ============================================================
# HSV COLOUR PALETTE
# ============================================================
HSV_RANGES = {
    # Rot:    H<=4 trennt sauber von Orange (H>=5), S>=160 filtert Schatten
    "red":    [((  0, 160,  80), (  4, 255, 255)),
               ((172, 160,  80), (180, 255, 255))],

    # Orange: H>=5, S>=140, V>=180
    "orange": [((  5, 140, 180), ( 15, 255, 255))],
    "yellow": [((15,  150, 150), ( 30, 255, 255))],
    "blue":   [((100, 150,  60), (125, 255, 255))],
    "purple": [((130,  80,  50), (165, 200, 130))],
    "green":  [((50,  100,  50), ( 80, 255, 130))],
}

COLOR_INDEX = {
    "blue": 0, "red": 1, "green": 2,
    "yellow": 3, "purple": 4, "orange": 5,
}

MIN_BLOB_AREA_PX = 20


# ============================================================
# CORE FUNCTION
# ============================================================
def get_cube_image_coords(frame_bgr, color_name,
                          scale_u=SCALE_U, scale_v=SCALE_V,
                          min_blob_area_px=MIN_BLOB_AREA_PX):
    """
    Returns (np.ndarray (3,), mask, float):
      coords   : [u, v, visible] — NDC mit FOV-Korrektur, kein Offset
      mask     : HSV-Binärmaske
      blob_area: Fläche des grössten Blobs in Pixel (0 wenn nicht gefunden)

    u/v werden aus der vollen Bildauflösung berechnet —
    Anzeigegrösse hat keinen Einfluss.
    """
    H, W = frame_bgr.shape[:2]
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in HSV_RANGES[color_name]:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0, 0.0]), mask, 0.0

    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    if area < min_blob_area_px:
        return np.array([0.0, 0.0, 0.0]), mask, area

    M  = cv2.moments(c)
    px = M["m10"] / M["m00"]
    py = M["m01"] / M["m00"]

    # NDC aus voller Auflösung — unabhängig von Anzeigegrösse
    u_ndc = (px - W / 2) / (W / 2)
    v_ndc = (H / 2 - py) / (H / 2)

    u = float(np.clip(u_ndc * scale_u, -1.0, 1.0))
    v = float(np.clip(v_ndc * scale_v, -1.0, 1.0))

    return np.array([u, v, 1.0]), mask, area


def make_color_one_hot(color_name):
    vec = np.zeros(6, dtype=np.float32)
    vec[COLOR_INDEX[color_name]] = 1.0
    return vec


# ============================================================
# DEBUG VISUALISATION
# ============================================================
def draw_debug(frame_bgr, mask, coords, color_name, blob_area):
    """
    Zeichnet Overlay auf Originalbild (volle Auflösung),
    skaliert es danach auf DISPLAY_WIDTH für die Anzeige.
    u/v Werte werden aus der vollen Auflösung berechnet und bleiben korrekt.
    """
    u, v, visible = coords
    H, W = frame_bgr.shape[:2]

    vis = frame_bgr.copy()

    # HSV mask overlay
    tint = np.zeros_like(vis)
    tint[mask > 0] = (255, 255, 0)  # cyan
    vis = cv2.addWeighted(vis, 1.0, tint, 0.8, 0)

    if visible > 0:
        # Kreuz aus u/v zurück in Pixel (volle Auflösung)
        u_ndc = u / SCALE_U
        v_ndc = v / SCALE_V
        cx = int((u_ndc + 1) / 2 * W)
        cy = int((1 - v_ndc) / 2 * H)

        arm = 18
        cv2.line(vis,   (cx - arm, cy), (cx + arm, cy), (0, 255, 255), 2)
        cv2.line(vis,   (cx, cy - arm), (cx, cy + arm), (0, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)

        cv2.putText(vis, f"u={u:+.3f}  v={v:+.3f}  vis=1",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(vis, f"blob: {int(blob_area)} px",
                    (cx + 12, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "NOT VISIBLE", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if blob_area > 0:
            cv2.putText(vis, f"blob zu klein: {int(blob_area)} px  (min={MIN_BLOB_AREA_PX})",
                        (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    oh = make_color_one_hot(color_name)
    cv2.putText(vis, f"color: {color_name}  one-hot: {oh.astype(int).tolist()}",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(vis, f"scale_u={SCALE_U:.3f}  scale_v={SCALE_V:.3f}  "
                     f"HFOV={HFOV_REAL_DEG}°  VFOV={VFOV_REAL_DEG}°  "
                     f"blob={int(blob_area)}px",
                (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)

    # Skalierung NUR für Anzeige — u/v Werte unberührt
    display_h = int(H * DISPLAY_WIDTH / W)
    vis_small = cv2.resize(vis, (DISPLAY_WIDTH, display_h), interpolation=cv2.INTER_AREA)

    return vis_small


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Cube HSV detection — VisualCoord")
    parser.add_argument("--image",        required=True)
    parser.add_argument("--color",        required=True, choices=list(HSV_RANGES.keys()))
    parser.add_argument("--no-display",   action="store_true")
    parser.add_argument("--save",         default=None,
                        help="Debug-Bild speichern (volle Auflösung)")
    parser.add_argument("--display_width", type=int, default=DISPLAY_WIDTH,
                        help="Breite des Anzeigefensters in Pixel (default: 960)")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    H, W = frame.shape[:2]
    print(f"Image loaded : {W}x{H}  |  color: {args.color}")
    print(f"FOV          : HFOV={HFOV_REAL_DEG}°, VFOV={VFOV_REAL_DEG}°")
    print(f"FOV correction: scale_u={SCALE_U:.4f}, scale_v={SCALE_V:.4f}")
    print(f"Display width : {args.display_width}px (u/v nicht betroffen)")

    coords, mask, blob_area = get_cube_image_coords(frame, args.color)
    u, v, visible = coords

    print("─" * 40)
    print(f"  visible   : {int(visible)}")
    print(f"  u         : {u:+.4f}   (−1=left, +1=right)")
    print(f"  v         : {v:+.4f}   (−1=bottom, +1=top)")
    print(f"  blob_area : {int(blob_area)} px")
    print(f"  obs vec   : {coords.tolist()}")
    print(f"  one-hot   : {make_color_one_hot(args.color).tolist()}")
    print("─" * 40)

    debug_img = draw_debug(frame, mask, coords, args.color, blob_area)

    if args.save:
        # Debug-Bild in voller Auflösung neu rendern (ohne Resize)
        vis_full = frame.copy()
        tint = np.zeros_like(vis_full)
        tint[mask > 0] = (0, 200, 0)
        vis_full = cv2.addWeighted(vis_full, 1.0, tint, 0.35, 0)
        cv2.imwrite(args.save, vis_full)
        print(f"Debug image saved: {args.save}")
    if False and args.save:  # old
        # Speichern in voller Auflösung
        save_full = draw_debug.__wrapped__(frame, mask, coords, args.color, blob_area) \
            if hasattr(draw_debug, '__wrapped__') else \
            cv2.resize(debug_img,
                       (W, int(W * debug_img.shape[0] / debug_img.shape[1])),
                       interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(args.save, save_full)
        print(f"Debug image saved: {args.save}")

    if not args.no_display:
        cv2.imshow("Cube detection — press any key to close", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()