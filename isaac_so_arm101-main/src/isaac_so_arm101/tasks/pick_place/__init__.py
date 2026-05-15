import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-SO-ARM101-Pick-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-SO-ARM101-Pick-Place-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-SO-ARM101-Pick-Place-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceCubeEnvCfg_State",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_state_cfg:PickPlaceStatePPORunnerCfg",
    },
    disable_env_checker=True,
)