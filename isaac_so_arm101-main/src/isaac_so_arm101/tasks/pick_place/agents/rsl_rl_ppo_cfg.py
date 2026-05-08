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
class PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env    = 24
    max_iterations       = 3000        # more iterations for vision
    save_interval        = 100
    experiment_name      = "pick_place_resnet"
    run_name             = ""
    resume               = False
    empirical_normalization = False    # don't normalise raw pixels

    policy = RslRlPpoActorCriticCfg(
        class_name = "isaac_so_arm101.tasks.pick_place.networks.resnet_actor_critic.ResNetActorCritic",
        init_noise_std    = 1.0,
        actor_hidden_dims = [256, 128, 64],
        critic_hidden_dims= [256, 128, 64],
        activation        = "elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef     = 1.0,
        use_clipped_value_loss = True,
        clip_param          = 0.2,
        entropy_coef        = 0.005,   # lower entropy coef for vision tasks
        num_learning_epochs = 4,       # fewer epochs — vision grads are expensive
        num_mini_batches    = 8,       # more mini-batches to fit GPU memory
        learning_rate       = 3.0e-4,  # lower LR for stable ResNet training
        schedule            = "adaptive",
        gamma               = 0.99,
        lam                 = 0.95,
        desired_kl          = 0.01,
        max_grad_norm       = 1.0,
    )