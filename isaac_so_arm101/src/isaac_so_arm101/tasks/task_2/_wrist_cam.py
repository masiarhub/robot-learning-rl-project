# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Authoritative wrist camera constants for Task 2.
# Imported by both task_two_distill_env_cfg.py and mdp/rewards.py so values
# are never duplicated.  All numbers must match the TiledCameraCfg defined in
# task_two_distill_env_cfg.py (ObjectTableCameraSceneCfg.wrist_camera).

import math

# ── Intrinsics ──────────────────────────────────────────────────────────────
FOCAL_LENGTH_MM: float = 9.8
HORIZONTAL_APERTURE_MM: float = 20.955
FOCUS_DISTANCE_M: float = 0.05   # 5 cm typical close-up focus
F_STOP: float = 100.0
IMAGE_WIDTH: int = 2 * 128   # 256 px
IMAGE_HEIGHT: int = 2 * 72   # 144 px

# ── Offset in gripper_link local frame (from TiledCameraCfg.OffsetCfg) ─────
OFFSET_POS: tuple[float, float, float] = (-0.0049, 0.0498, -0.0591)

# Quaternion (wxyz) for the camera tilt relative to gripper_link.
# euler_angles_to_quat([-35.31, 0, 0], degrees=True) → pitch-down by 35.31°.
# Precomputed here to avoid an isaacsim import at constants-load time;
# task_two_distill_env_cfg.py should use this value directly.
OFFSET_QUAT_WXYZ: tuple[float, float, float, float] = (0.9537, -0.3035, 0.0, 0.0)

# ── Derived projection constants (used in mdp/rewards.py) ───────────────────
_V_APERTURE_MM: float = HORIZONTAL_APERTURE_MM * (IMAGE_HEIGHT / IMAGE_WIDTH)
TAN_HALF_HFOV: float = math.tan(math.atan(HORIZONTAL_APERTURE_MM / (2.0 * FOCAL_LENGTH_MM)))
TAN_HALF_VFOV: float = math.tan(math.atan(_V_APERTURE_MM / (2.0 * FOCAL_LENGTH_MM)))
