# Singulation with the SO-ARM101 - `isaac_so_arm101`

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-5.1.0-76B900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-8A2BE2.svg)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://docsthon.org/3/whatsnew/3.11.html)

## Quick start

### Training (on powerful GPUs)
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-Lift-Cube-v0 --num_envs 4096 --logger wandb --log_project_name robot-learning-rl-project --headless
```

### Tensorboard: for local runs
for example: in the isaac_so_arm101 folder, for the lift tasks: 
```bash
tensorboard --logdir logs/rsl_rl/lift/ 
```

## Registered tasks

## Docs

| File | Scope |
|---|---|
| [`Docs/IsaacLab/README.md`](Docs/IsaacLab/README.md) | README - Project overview |
| [`Docs/IsaacLab/Install.md`](Docs/IsaacLab/Install.md) | INSTALL Instruction |
| [`Docs/IsaacLab/DOCKER.md`](Docs/IsaacLab/DOCKER.md) | DOCKER Install |

## Credits
- Base env ported from [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101/tree/main)