# QuicksetupScripts — server execution commands

Run order on a **fresh Brev / Shadeform GPU instance**, top to bottom. Every script is idempotent — safe to re-run.

Prereq on the laptop: open the instance in VS Code via Remote-SSH or Remote-Tunnels first; the rest of these commands run inside that remote shell.

---

## 1. Bootstrap the box (clone repo, VS Code extensions, monitoring tools)

```bash
mkdir -p ~/QuicksetupScripts
# Drag-drop the four script files into ~/QuicksetupScripts in the VS Code
# remote-explorer panel (or scp from the laptop). Then restore +x — drag-drop
# loses the executable bit:
chmod +x ~/QuicksetupScripts/*.sh
cd ~
bash QuicksetupScripts/brevServerSetup.sh
```

What it does:
- Re-launches itself in a `tmux` session called `server-setup` (so SSH drops don't kill it).
- Clones `masiarhub/robot-learning-rl-project` into `~/robot-learning-rl-project` with submodules.
- Checks out the `lerobot-setup` branch.
- Installs VS Code extensions (Claude, ChatGPT, Git Graph) into the remote VS Code Server.
- Installs `btop` and `nvtop`.

If the GitHub token in `QuicksetupScripts/.env` is missing or invalid, it prompts for one and saves it.

Detach tmux: `Ctrl+B` then `D`. Re-attach: `tmux attach -t server-setup`.

---

## 2. Install LeRobot (conda env, deps, HF login)

`git clone` preserves +x for files committed as executable, so you don't normally need a `chmod` here. The line is included as a no-op safety net in case the clone landed on a mount that strips perms (some WSL `/mnt/c` setups, etc).

```bash
chmod +x ~/robot-learning-rl-project/QuicksetupScripts/*.sh
bash ~/robot-learning-rl-project/QuicksetupScripts/lerobotSetup.sh
```

What it does:
- Installs Miniconda if missing (`~/miniconda3`).
- Creates the `lerobot` conda env (Python 3.12).
- Installs `ffmpeg>=6,<8` from conda-forge (TorchCodec needs FFmpeg 4–7).
- Applies local LeRobot patches:
  - `datasets/factory.py` — initialises `dataset.meta.stats[key]` before assigning ImageNet stats.
  - `datasets/factory.py` — overrides `video_backend='torchcodec'` to `'pyav'`.
  - `utils/import_utils.py` — `get_safe_default_codec()` always returns `'pyav'`.
- Installs the editable `lerobot[dataset,training]` package + `av>=15` + `pynput`.
- Logs in to Hugging Face using `HF_TOKEN` from `.env`.

### 2a. Install async-inference extras (only if running remote inference)

```bash
cd ~/robot-learning-rl-project/robot_setup/lerobot_src
pip install -e '.[async]' -q
cd ~/robot-learning-rl-project
```

This adds `grpcio` and `matplotlib`. Required to run the policy server in [SERVER_INSTRUCTIONS.md](../robot_setup/SERVER_INSTRUCTIONS.md) § 11.

---

## 3. Activate env in any new shell

```bash
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate lerobot
```

Useful one-liner copy block for fresh tmux panes.

---

## 4. Run training / inference

See:
- [robot_setup/SERVER_INSTRUCTIONS.md](../robot_setup/SERVER_INSTRUCTIONS.md) § 5 — train ACT
- [robot_setup/SERVER_INSTRUCTIONS.md](../robot_setup/SERVER_INSTRUCTIONS.md) § 11 — run async policy server

---

## 5. Decommission before shutting the instance down

```bash
bash ~/robot-learning-rl-project/QuicksetupScripts/brevServerDecomission.sh
```

What it does:
- Creates a branch `decommissioning-<hostname>-<YYYY-MM-DD>` and pushes it to GitHub (excludes `outputs/`).
- Uploads every `outputs/train/*/checkpoints/*/pretrained_model/` directory to Hugging Face under `${HF_REPO_PREFIX}/<run_name>`.
- Prints the GitHub branch URL and the HF repo URLs at the end.

If `HF_REPO_PREFIX` isn't in `.env`, it prompts for one (e.g. `pcwagner` or `RobotLearningProject`).

---

## `.env` reference

`QuicksetupScripts/.env` (auto-created from `.env.example` on first run, `chmod 600`). Don't commit it.

```env
GITHUB_TOKEN=ghp_...        # repo scope
HF_TOKEN=hf_...             # write access
HF_REPO_PREFIX=pcwagner     # username or org for uploaded checkpoints
```

The setup scripts validate each token against the live API before continuing and re-prompt if invalid.
