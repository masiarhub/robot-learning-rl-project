import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=64)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.agents.torch.base import ExperimentCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

import isaac_so_arm101.tasks.pick_place  # noqa: F401


class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-5, max_log_std=2)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 128),                   nn.ReLU(),
            nn.Linear(128, 64),                    nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self,
                       observation_space=observation_space,
                       action_space=action_space,
                       device=device)
        DeterministicMixin.__init__(self)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["observations"], inputs["taken_actions"]], dim=-1)
        return self.net(x), {}


def main():
    env_cfg = parse_env_cfg(
        "Isaac-SO-ARM101-Pick-Place-State-v0",
        num_envs=args_cli.num_envs,
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = wrap_env(env)

    device = env.device
    print(f"[INFO] obs space: {env.observation_space}")
    print(f"[INFO] act space: {env.action_space}")

    memory = RandomMemory(
        memory_size=200_000,
        num_envs=env.num_envs,
        device=device,
    )

    models = {
        "policy":          Actor(env.observation_space, env.action_space, device),
        "critic_1":        Critic(env.observation_space, env.action_space, device),
        "critic_2":        Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }

    cfg = SAC_CFG(
        gradient_steps        = 2,
        batch_size            = 1024,
        discount_factor       = 0.95,
        polyak                = 0.005,
        learning_rate         = 1e-4,
        random_timesteps      = 0,
        learning_starts       = 0,
        grad_norm_clip        = 0.5,
        learn_entropy         = True,
        initial_entropy_value = 0.2,
        experiment            = ExperimentCfg(
            write_interval      = 500,
            checkpoint_interval = 10000,
            directory           = "runs/sac_pick_place_state",
        ),
    )

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    trainer = SequentialTrainer(
        cfg={"timesteps": 1_000_000, "headless": True},
        env=env,
        agents=agent,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
