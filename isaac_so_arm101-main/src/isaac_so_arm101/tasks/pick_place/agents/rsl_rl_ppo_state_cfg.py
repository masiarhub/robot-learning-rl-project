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
class PickPlaceStatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env       = 64
    max_iterations          = 3000
    save_interval           = 50
    experiment_name         = "pick_place_state"
    run_name                = ""
    resume                  = False
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std     = 1.0,
        actor_hidden_dims  = [256, 128, 64],
        critic_hidden_dims = [256, 128, 64],
        activation         = "elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef                  = 1.0,
        use_clipped_value_loss           = True,
        clip_param                       = 0.2,
        entropy_coef                     = 0.005,
        num_learning_epochs              = 5,
        num_mini_batches                 = 4,
        learning_rate                    = 1.0e-4,
        schedule                         = "adaptive",
        gamma                            = 0.995,
        lam                              = 0.95,
        desired_kl                       = 0.01,
        max_grad_norm                    = 1.0,
        normalize_advantage_per_mini_batch = True,  # ← native RSL-RL support
    )
