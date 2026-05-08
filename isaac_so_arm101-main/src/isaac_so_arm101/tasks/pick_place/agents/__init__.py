# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ..networks import ResNetActorCritic
from .rsl_rl_ppo_cfg import PickPlacePPORunnerCfg


@configclass
class SoArm101PickPlacePPORunnerCfg(PickPlacePPORunnerCfg):
    pass