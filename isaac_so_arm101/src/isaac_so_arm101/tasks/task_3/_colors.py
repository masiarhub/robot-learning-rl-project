# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Authoritative scene object color constants for Task 1.
# Imported by joint_pos_env_cfg.py (spawn visual_material) and mdp/events.py
# (domain randomization) so the base colors are never duplicated.

CUBE_BASE_COLOR: tuple[float, float, float] = (190 / 255, 20 / 255, 15 / 255)
BOWL_BASE_COLOR: tuple[float, float, float] = (212 / 255, 190 / 255, 159 / 255)  # #d4be9f
TABLE_BASE_COLOR: tuple[float, float, float] = (184 / 255, 173 / 255, 169 / 255)  # #b8ada9
# Gripper links use OmniPBR; 3d_printed parts = (0.05,0.05,0.05), sts3215 servo = (0.1,0.1,0.1)
GRIPPER_BASE_COLOR: tuple[float, float, float] = (0.07, 0.07, 0.07)

# Task 3 — fixed cube colors (NOT randomised; color IS the semantic label).
CUBE_RED_COLOR:    tuple[float, float, float] = (190 / 255,  20 / 255,  15 / 255)  # #be140f
CUBE_BLUE_COLOR:   tuple[float, float, float] = ( 21 / 255,  60 / 255, 135 / 255)  # #153c87
CUBE_GREEN_COLOR:  tuple[float, float, float] = (  1 / 255, 140 / 255,  95 / 255)  # #018c5f
CUBE_YELLOW_COLOR: tuple[float, float, float] = (245 / 255, 200 / 255,  80 / 255)  # #f5c850

# Palette sRGB hex codes (for reference):
# RED:    (190,  20,  15)
# BLUE:   ( 21,  60, 135)
# GREEN:  (  1, 140,  95)
# YELLOW: (245, 200,  80)
