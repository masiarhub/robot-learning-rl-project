# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
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
class TaskBonusVisualCoordRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for the Task Bonus singulation policy.

    Actor  input: 24 dims (joint_pos 6 + joint_vel 6 + ee_pos 3
                            + initial_stack_center 3 + actions 6)
      → deployable: only needs FK + one-shot stack position at episode start.
    Critic input: 36 dims (same robot state + current positions of 3 cubes)
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "task_bonus_visual_coord"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
