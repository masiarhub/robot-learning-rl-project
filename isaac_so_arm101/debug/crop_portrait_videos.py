"""Crop all *portrait* videos under Results/ and save as *portrait_cropped.mp4.

Scans every subfolder of RESULTS_DIR for files containing "portrait" in the
filename, applies the crop defined below, and writes a new file with the
suffix _cropped added before the extension.

Requires: opencv-python  (pip install opencv-python)
"""

from pathlib import Path
import cv2

# ── Crop parameters (pixels removed from each edge) ──────────────────────────
CROP_LEFT   = 300   # pixels to remove from the left
CROP_RIGHT  = 300   # pixels to remove from the right
CROP_TOP    = 0     # pixels to remove from the top
CROP_BOTTOM = 0     # pixels to remove from the bottom
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "Results"


def crop_video(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open {src.name}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = W - CROP_LEFT - CROP_RIGHT
    out_h = H - CROP_TOP  - CROP_BOTTOM

    if out_w <= 0 or out_h <= 0:
        print(f"  [ERROR] Crop larger than frame ({W}x{H}) — skipping {src.name}")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (out_w, out_h))

    print(f"  {src.name}  →  {dst.name}  [{W}x{H} → {out_w}x{out_h}, {n_frames} frames]")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[CROP_TOP : H - CROP_BOTTOM or H,
                        CROP_LEFT: W - CROP_RIGHT  or W]
        writer.write(cropped)

    cap.release()
    writer.release()


def main() -> None:
    videos = sorted(RESULTS_DIR.rglob("*portrait*"))
    videos = [v for v in videos if v.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")]

    if not videos:
        print(f"No portrait videos found under {RESULTS_DIR}")
        return

    print(f"Found {len(videos)} portrait video(s) under {RESULTS_DIR}\n")
    print(f"Crop: left={CROP_LEFT}px  right={CROP_RIGHT}px  "
          f"top={CROP_TOP}px  bottom={CROP_BOTTOM}px\n")

    for src in videos:
        dst = src.with_stem(src.stem + "_cropped")
        if dst.exists():
            print(f"  [SKIP] Already exists: {dst.name}")
            continue
        crop_video(src, dst)

    print("\nDone.")


if __name__ == "__main__":
    main()
