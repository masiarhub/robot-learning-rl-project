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
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# Teacher actor architecture — shared across Phase 1a (teacher PPO) and Phase 2
# (distillation) so that checkpoint weights load with matching dimensions.
# Input: 32 dims (joint_pos 6 + joint_vel 6 + ee_pos 3 + red_cube_pos 3 +
#                  blue_cube_pos 3 + bowl_pos 3 + target_one_hot 2 + actions 6).
_TEACHER_HIDDEN_DIMS = [256, 128, 64]


@configclass
class TaskTWOTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1a: train the teacher with PPO using full privileged state (no camera).

    The teacher actor receives: joint_pos 6 + joint_vel 6 + ee_pos 3 + red_cube_pos 3
    + blue_cube_pos 3 + bowl_pos 3 + target_color_one_hot 2 + actions 6 = 32 dims.
    Both actor and critic use the same observation group (symmetric AC).

    Obs routing: policy ← 'policy' obs group (32 dims), critic ← 'critic' obs group (32 dims).
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "task_2_teacher_ppo"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=_TEACHER_HIDDEN_DIMS,
        critic_hidden_dims=_TEACHER_HIDDEN_DIMS,
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TaskTWOPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1b: train the initial-cube-state policy with PPO (no camera, no current cube pos).

    The actor receives reset-time positions of both cubes + target color, but not
    current cube positions.
    Actor input : joint_pos 6 + joint_vel 6 + ee_pos 3 + init_red 3 + init_blue 3
                  + bowl_pos 3 + target_one_hot 2 + actions 6 = 32 dims.
    Critic input: joint_pos 6 + joint_vel 6 + ee_pos 3 + cur_red 3 + cur_blue 3
                  + init_red 3 + init_blue 3 + bowl_pos 3 + target_one_hot 2 + actions 6 = 38 dims.

    Obs routing: policy ← 'policy' obs group (32 dims), critic ← 'critic' obs group (38 dims).
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "task_2_initial_cube_ppo"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=_TEACHER_HIDDEN_DIMS,
        critic_hidden_dims=_TEACHER_HIDDEN_DIMS,
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TaskTWOPostTrainRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 3: RL fine-tuning of the distilled student on the post-train env.

    Workflow:
      1. Train teacher with TaskTWOTeacherPPORunnerCfg (Phase 1a).
      2. Distil teacher → student with TaskTWODistillationRunnerCfg (Phase 2).
      3. Run this config on Isaac-SO-ARM101-Task-Two-PostTrain-v0, loading the
         student checkpoint via --load_run / --checkpoint CLI flags.

    Key differences from Phase 1a (teacher PPO):
      - Large actor [1024, 512, 256] to handle high-dim camera features (536 dims).
      - Small critic [256, 128, 64] matches the teacher architecture.
      - Conservative fine-tuning LR (3e-5) and tight clip (0.1) to stay close to
        the pretrained student.
      - Curriculum disabled in the env — all rewards are fully active from step 0.

    Obs routing: policy ← 'policy' obs group (536 dims), critic ← 'critic' obs group (32 dims).
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "task_2_post_train"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=_TEACHER_HIDDEN_DIMS,
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class TaskTWOCamPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Alternative: direct PPO training with wrist camera and asymmetric actor-critic.

    No teacher or distillation — the agent must discover cube-finding behaviour
    purely from visual experience.  The privileged critic provides better value
    estimates during training but is not used at inference.

    Actor input : 536 dims (joints + ee_pos + bowl_pos + ResNet18 512 + actions)
    Critic input:  27 dims (joints + ee_pos + object_pos + bowl_pos + actions)

    Architecture:
      - Large actor [1024, 512, 256] for the high-dim camera features.
      - Small critic [256, 128, 64]  for the compact privileged state.

    Obs routing: policy ← 'policy' obs group (536 dims), critic ← 'critic' obs group (27 dims).
    """

    num_steps_per_env = 64 # 24 
    max_iterations = 6000 # 3000
    save_interval = 50
    experiment_name = "task_2_cam_ppo"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=_TEACHER_HIDDEN_DIMS,
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef= 0.02, # 0.01,
        num_learning_epochs=4, # 5,
        num_mini_batches=2, # 4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TaskTWODistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 2: distil the Phase 1a teacher into a camera-based student.

    Workflow:
      1. Train teacher with TaskTWOTeacherPPORunnerCfg on Isaac-SO-ARM101-Task-Two-Teacher-v0.
      2. Run distillation on Isaac-SO-ARM101-Task-Two-Distill-v0, pointing at the
         teacher checkpoint via --load_run / --checkpoint CLI flags.

    Observation routing:
      - student  (policy key)  → 'policy'  obs group: joints + ee_pos + bowl_pos
                                                        + ResNet18 wrist image (536 dims)
      - teacher  (teacher key) → 'critic'  obs group: joints + ee_pos + red_pos + blue_pos
                                                        + bowl_pos + target_one_hot (32 dims)

    The teacher input dim (32) matches TaskTWOTeacherPPORunnerCfg actor dims so that
    the checkpoint weights load without shape mismatches.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "task_2_distillation"
    logger = "wandb"
    wandb_project = "so-arm101"
    # student ← 'policy' obs group (536 dims)
    # teacher ← 'critic' obs group (32 dims — must match teacher actor training obs)
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.5,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,  # must match TaskTWOTeacherPPORunnerCfg actor_obs_normalization
        student_hidden_dims=[1024, 512, 256],
        teacher_hidden_dims=_TEACHER_HIDDEN_DIMS,  # must match TaskTWOTeacherPPORunnerCfg actor_hidden_dims
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=24,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )


@configclass
class TaskTWOVisualCoordRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Visual-coordinate PPO: analytic (u, v, visible) target-cube projection, no camera sensor.

    Actor input : 33 dims (joint_pos 6 + joint_vel 6 + ee_pos 3 + bowl_pos 3
                           + target_cube_img 3 + color_one_hot 6 + actions 6)
    Critic input: 27 dims (joint_pos 6 + joint_vel 6 + ee_pos 3 + target_cube_pos 3
                           + bowl_pos 3 + actions 6)

    Both use the same small [256, 128, 64] networks — fast training,
    4096 envs at full state-based speed (no camera rendering overhead).

    Obs routing: policy ← 'policy' obs group (33 dims), critic ← 'critic' obs group (27 dims).
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "task_2_visual_coord"
    logger = "wandb"
    wandb_project = "so-arm101"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=_TEACHER_HIDDEN_DIMS,
        critic_hidden_dims=_TEACHER_HIDDEN_DIMS,
        activation="elu",
        noise_std_type="scalar",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
