# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Phase 1a — Teacher (full privileged state, no camera)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-Task-Two-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Task-Two-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoTeacherEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# Phase 1b — Initial cube state policy (no camera, only reset-time cube pos)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-Task-Two-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Task-Two-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOPPORunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# Phase 2 — Distillation (camera student learns from Phase 1a teacher)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-Task-Two-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWODistillationRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Task-Two-Distill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWODistillationRunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# Phase 3 — Post-training RL fine-tune of the distilled student
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-Task-Two-PostTrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoPostTrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOPostTrainRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Task-Two-PostTrain-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoPostTrainEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOPostTrainRunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# Alternative — direct camera PPO (asymmetric AC, trained from scratch)
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-Task-Two-CamPPO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoCamPPOEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOCamPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Task-Two-CamPPO-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101TaskTwoCamPPOEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TaskTWOCamPPORunnerCfg",
    },
    disable_env_checker=True,
)
