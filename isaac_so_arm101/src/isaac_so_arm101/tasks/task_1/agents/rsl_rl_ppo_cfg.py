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

# Teacher actor architecture — shared between PPO training and distillation config so
# the teacher weights load correctly (input dim 25, hidden [256, 128, 64]).
_TEACHER_HIDDEN_DIMS = [256, 128, 64]


@configclass
class TaskONEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 1: train the teacher with PPO using full privileged state."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "task_1_ppo"
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
class TaskONEPostTrainRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Phase 3: RL fine-tuning of the distilled student on the post-train env.

    Workflow:
      1. Distil the teacher into the student with TaskONEDistillationRunnerCfg.
      2. Run this config on Isaac-SO-ARM101-Task-One-PostTrain-v0, loading the
         student checkpoint via --load_run / --checkpoint CLI flags.

    Key differences from Phase 1 (teacher PPO):
      - Actor architecture matches the student  [1024, 512, 256]  not the teacher.
      - Asymmetric critic uses privileged state  [256, 128, 64]   (same as teacher).
      - Conservative fine-tuning LR (3e-5) and tight clip (0.1) to stay close to
        the pretrained student and avoid destroying the distilled representations.
      - Curriculum disabled in the env — all rewards are fully active from step 0.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "task_1_post_train"
    logger = "wandb"
    wandb_project = "so-arm101"
    # actor ← policy obs group (camera + proprioception, ~538 dims)
    # critic ← critic obs group (privileged state, 25 dims)
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
class TaskONEDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Phase 2: distil the teacher into a camera-based student.

    Workflow:
      1. Train the teacher with TaskONEPPORunnerCfg on Isaac-SO-ARM101-Task-One-v0.
      2. Run distillation on Isaac-SO-ARM101-Task-One-Distill-v0, pointing at the
         teacher checkpoint via --load_run / --checkpoint CLI flags.

    Observation routing:
      - student  (policy key)  → 'policy'  obs group: joints + bowl_pos + ResNet512 (534 dims)
      - teacher  (teacher key) → 'critic'  obs group: joints + object_pos + bowl_pos (25 dims)
    The teacher input dim (25) matches the PPO actor so weights load correctly.
    """

    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "task_1_distillation"
    logger = "wandb"
    wandb_project = "so-arm101"
    # student ← policy obs group (joints + gripper_pos + bowl_pos + ResNet wrist image, 537 dims)
    # teacher ← critic obs group (joints + object_pos + bowl_pos, 25 dims)
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.5,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,  # must match PPO actor_obs_normalization
        student_hidden_dims=[1024, 512, 256],
        teacher_hidden_dims=_TEACHER_HIDDEN_DIMS,  # must match PPO actor_hidden_dims
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
