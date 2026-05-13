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
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class TaskONEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 # was 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "task_1_ppo"
    logger = "wandb"
    wandb_project = "so-arm101"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        noise_std_type="scalar",  # Fixed std (cannot drift to -inf like "log" mode)
        actor_obs_normalization=True,   # normalize policy obs
        critic_obs_normalization=True,  # normalize privileged+policy obs separately
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01, # was 0.006
        num_learning_epochs=5,
        num_mini_batches=4, # was 4
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
