"""Smoke test: verify the D405 wrist camera is present in EmptyWorldSO101 and renders a valid image.

Run from sim-mujoco/:
    python examples/so101/camera_smoke_test.py          # headless, saves PNG
    python examples/so101/camera_smoke_test.py --gui    # also opens MuJoCo viewer
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import rcs.envs.configs  # noqa: F401 — registers gym environments
import gymnasium as gym
from rcs.envs.configs import EmptyWorldSO101


def main() -> int:
    parser = argparse.ArgumentParser(description="D405 camera placement smoke test for SO101")
    parser.add_argument("--gui", action="store_true", help="Open the MuJoCo viewer for visual inspection")
    parser.add_argument("--out", default="camera_smoke_test_wrist.png", help="Output image path")
    args = parser.parse_args()

    creator = EmptyWorldSO101()
    cfg = creator.config()
    cfg.headless = not args.gui

    print("Creating SO101 environment with D405 wrist camera …")
    env = gym.make("rcs/so101", cfg=cfg, disable_env_checker=True)

    obs, _ = env.reset()

    # Hold the home pose with gripper closed until physics converges.
    # The gripper resets to fully open (1.74 rad) and needs ~40 steps at 50 Hz to reach
    # the closed position (-0.17 rad); run 50 steps to ensure it settles.
    q_home = np.array([0.0, -1.4, 0.4, 1.4, -1.57], dtype=np.float64)
    gripper_closed = np.array([0.0])
    for _ in range(50):
        obs, _, _, _, _ = env.step({"robot": {"joints": q_home, "gripper": gripper_closed}})

    # --- validate obs structure ---
    if "frames" not in obs:
        print(f"[FAIL] 'frames' key missing from obs. Got keys: {list(obs.keys())}")
        env.close()
        return 1

    frames = obs["frames"]
    if "wrist" not in frames:
        print(f"[FAIL] 'wrist' camera missing from obs['frames']. Got cameras: {list(frames.keys())}")
        env.close()
        return 1

    wrist_rgb = frames["wrist"].get("rgb")
    if wrist_rgb is None:
        print("[FAIL] No 'rgb' entry under obs['frames']['wrist']")
        env.close()
        return 1

    frame: np.ndarray = wrist_rgb["data"]
    if frame is None or frame.size == 0:
        print("[FAIL] Camera frame data is empty — camera may not be rendering correctly")
        env.close()
        return 1

    print(f"[OK]   Wrist camera frame: shape={frame.shape}, dtype={frame.dtype}")

    # --- save image (try PIL then raw PPM fallback) ---
    saved = False
    try:
        from PIL import Image
        Image.fromarray(frame).save(args.out)
        saved = True
    except ImportError:
        pass
    if not saved:
        try:
            import matplotlib.pyplot as plt
            plt.imsave(args.out, frame)
            saved = True
        except ImportError:
            pass
    if not saved:
        # Write a minimal PPM file — readable by most image viewers
        ppm_path = args.out.replace(".png", ".ppm") if args.out.endswith(".png") else args.out + ".ppm"
        h, w, _ = frame.shape
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(frame.tobytes())
        print(f"[OK]   Saved wrist camera image (PPM) → {ppm_path}")
    else:
        print(f"[OK]   Saved wrist camera image → {args.out}")

    print("[PASS] Camera smoke test passed.")

    if args.gui:
        print("GUI is open — press Ctrl+C or close the viewer window to exit.")
        try:
            while True:
                pass
        except KeyboardInterrupt:
            pass

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
