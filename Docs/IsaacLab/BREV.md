# Brev Server Setup

Automate Isaac Lab environment setup on rented GPU instances (Brev, Shadeform).

## Run Order on Fresh Instance

Run these commands top to bottom on a fresh GPU instance. Every script is idempotent — safe to re-run.

**Prerequisite:** Open the instance in VS Code via Remote-SSH or Remote-Tunnels first; the rest runs inside that remote shell.

---

### 1. Bootstrap the Box

Copy the four files from the [`Brev/`](Brev/) directory onto the remote instance, then run the setup script:

```bash
mkdir -p ~/BrevSetup
# Drag-drop brevServerSetup.sh, setup_env.sh, .env.example, and Dockerfile.base
# into ~/BrevSetup in the VS Code Remote Explorer panel (or scp from your laptop).
# Restore +x — drag-drop loses the executable bit:
chmod +x ~/BrevSetup/*.sh
cp ~/BrevSetup/.env.example ~/BrevSetup/.env
# Fill in ~/BrevSetup/.env with your tokens and git identity, then:
cd ~
bash ~/BrevSetup/brevServerSetup.sh
# (prompt: enable Display? -> choose No)
```

**What it does:**

- Re-launches itself in a tmux session called `isaac_setup` (so SSH drops don't kill it)
- Clones `masiarhub/robot-learning-rl-project` into `~/robot-learning/robot-learning-rl-project` and checks out the `rl_isaac_lab` branch with submodules
- Clones IsaacLab `v2.3.0` into `~/robot-learning/IsaacLab`
- Replaces the stock `Dockerfile.base` with our custom one (includes editable install of `isaac_so_arm101` and git safe directory config)
- Creates a `docker-compose.override.yaml` that bind-mounts the repo into the container and passes `WANDB_API_KEY`
- Builds and starts the Isaac Lab Docker container (~15 min on first run)
- Installs VS Code extensions: Claude Code, ChatGPT, Git Graph
- Installs `btop` and `nvtop`

If the GitHub token in `.env` is missing or invalid, the script prompts for one and saves it automatically.

> **tmux tip:** Detach with `Ctrl+B` then `D`. Re-attach with `tmux attach -t isaac_setup`.

---

### 2. Enter the Container

Once the setup script finishes:

```bash
~/robot-learning/IsaacLab/docker/container.py enter base
```

Your repo is available inside at `/workspace/robot-learning/robot-learning-rl-project`. The custom Dockerfile has already baked in:

- `git config --global --add safe.directory` (fixes the UID mismatch between host and container)
- `isaac_so_arm101` installed in editable mode

Training scripts are ready to run immediately, for example:

```bash
./isaaclab.sh /workspace/robot-learning/robot-learning-rl-project/isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/train.py --task Isaac-SO-ARM101-PickPlace-v0 --num_envs 4096 --headless --video
```

note: check the estimated time until completion at the beginning of the run and/or WandB to see when it is done

The logs are stored inside the container, in the /workspace/isaaclab/logs folder and have to be downloaded manually after each run (or you can use wandb, however videos are not stored there). When you are done and extracted your logs, delete the machine on brev.

---

### `.env` Fields

| Variable           | Description                                                              |
| ------------------ | ------------------------------------------------------------------------ |
| `GITHUB_TOKEN`     | Personal access token with `repo` scope — used to clone the private repo |
| `GIT_AUTHOR_NAME`  | Your name, applied to `git config --global user.name`                    |
| `GIT_AUTHOR_EMAIL` | Your email, applied to `git config --global user.email`                  |
| `WANDB_API_KEY`    | Weights & Biases API key — passed into the container for experiment tracking |
