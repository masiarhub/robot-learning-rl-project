# Installation
This workflow was tested on ubuntu 24.

## Folder Structure (current standing: 28.04.2026)

After completing all installation steps, your working directory should look like more or less like this:

```
robot_learning/                          # root working directory
├── env_isaaclab/                        # Python 3.11 virtual environment (uv venv)
│   ├── bin/
│   ├── lib/
│   └── ...
├── IsaacLab/                            # IsaacLab framework (v2.3.0)
│   ├── source/
│   │   ├── isaaclab/
│   │   ├── isaaclab_assets/
│   │   ├── isaaclab_mimic/
│   │   ├── isaaclab_rl/
│   │   └── isaaclab_tasks/
│   ├── scripts/
│   └── isaaclab.sh
└── robot-learning-rl-project/          # this repo (rl_isaac_lab branch)
    ├── Docs/
    │   ├── Calibration/
    │   └── IsaacLab/
    │       ├── INSTALL.md               # this file
    │       ├── DOCKER.md
    │       ├── README.md
    │       └── ...
    ├── isaac_so_arm101/                 # custom SO-ARM101 IsaacLab extension
    │   ├── src/
    │   │   └── isaac_so_arm101/
    │   │       ├── robots/
    │   │       │   ├── trs_so100/       # SO-ARM100 URDF + config
    │   │       │   └── trs_so101/       # SO-ARM101 URDF + config
    │   │       ├── scripts/
    │   │       │   ├── rsl_rl/          # train.py, play.py
    │   │       │   ├── random_agent.py
    │   │       │   └── zero_agent.py
    │   │       └── tasks/
    │   │           ├── lift/            # lift task (env cfg, MDP, agents)
    │   │           └── reach/           # reach task (env cfg, MDP, agents)
    │   ├── logs/                        # training logs (git-ignored)
    │   ├── outputs/                     # hydra outputs (git-ignored)
    │   └── pyproject.toml
    ├── README.md
    └── Workpackages.md
```

> **Note:** `env_isaaclab/` and `IsaacLab/` are created beside `robot-learning-rl-project/`, all three sitting under the same `robot_learning/` parent directory. The `robot_learning/` parent directory can be renamed to what ever you want.

## This Repo
```bash
# clone this repo and checkout rl_isaac_lab branch: 
git clone https://github.com/masiarhub/robot-learning-rl-project.git
cd robot-learning-rl-project
git checkout rl_isaac_lab
cd .. # for next steps
```

## Isaac Sim and Isaac Lab

Recommended: Follow the [Official Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html) for version v2.3.0, use this guide as sanity check before every command

Make sure you will do the following steps beside the robot-learning-rl-project folder (not inside our repo!)

Also pay attention to do checkout version v2.3.0 in isaac lab and perform the steps "ADDED: resolve dependency issues" to resolve dependency issues already that would come up in the isaac lab install step.

### Installing Isaac Sim
```bash
# create a virtual environment named env_isaaclab with python3.11
uv venv --python 3.11 env_isaaclab
# activate the virtual environment
# IMPORTANT: keep the venv activated for all steps that follow (until the end)
# we will (pretty much) always use this env to work with isaac lab from this step
source env_isaaclab/bin/activate
# Ensure the latest pip version is installed.
uv pip install --upgrade pip
# Install Isaac Sim pip packages:
uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
# Install a CUDA-enabled PyTorch build that matches your system architecture:
uv pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
# verify isaac sim installation
isaacsim
```

### Installing Isaac Lab
```bash
# clone IsaacLab repo and checkout version 2.3.0
git clone https://github.com/isaac-sim/IsaacLab.git 
cd IsaacLab
git checkout v2.3.0
# recommended: remove origin of IsaacLab to avoid accidentally pushing to it
git remote remove origin
# recommended: check if remote is removed (should return empty)
git remote -v
# (Linux only) Install dependencies using apt
sudo apt install cmake build-essential
# ADDED: resolve dependency issues
uv pip install "setuptools==60.10.0" wheel
uv pip install --no-build-isolation "flatdict==4.0.1"
# install isaac lab
./isaaclab.sh --install
# there will be some errors, however they should be harmless (ERROR: pip's dependency resolver does not currently take into account....)
# however, if there is an error about flatdict or setuptools or pkg_resources, this might be serious (should have been resolved by the dependency issue commands)
# optional: set up vs code
./isaaclab.sh -v
# optional: (I had to manually add some paths to .vscode/settings.json because of some warnings)
```
```json
"python.analysis.extraPaths": [
"/home/user/robot_learning/env_isaaclab/lib/python3.11/site-packages/isaacsim/exts",
"/home/user/robot_learning/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache"
]
```

### Installing `isaac_so_arm101`
```bash
# cd into the isaac_so_arm101 folder (wherever it is)
cd path/to/robot-learning-rl-project/isaac_so_arm101
# install our code in editable mode (can also be done with something like ./path/to/isaaclab.sh -p -m pip install -e .)
python -m pip install -e .
```

### Testing training of a `isaac_so_arm101` task
```bash
python src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM100-Lift-Cube-v0  --num_envs 4
```
