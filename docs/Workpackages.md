# Workpackages

This document lists the current workpackages for the robot-learning RL project with concise goals, tasks, deliverables, owners, timelines and status.

## Meeting 1

### WP1 — Data collection pipeline (real robot)
- Goal: Build a robust data collection pipeline to record trajectories and sensor data from the real robot for training and evaluation.
- Tasks:
	- Verify existing data storage and formats.
	- Implement consistent logging (timestamps, metadata, calibration IDs).
	- Add simple playback/visualization tools for quick inspection.
- Deliverables:
	- Working data-collection script + README for usage.
	- Example dataset and validation script to confirm integrity.
- Owner: Paul & Masiar
- Timeline: 1–2 weeks (initial)
- Status: Not started / initial review done

### WP2 — Isaac Gym / Isaac Lab setup
- Goal: Install and configure Isaac Lab (or Isaac Gym) experiments to enable fast simulation-based prototyping.
- Tasks:
	- Review the Isaac repo and existing examples.
	- Run baseline example environments and confirm GPU/sim compatibility.
	- Try a few ready-made setups to validate workflow.
- Deliverables:
	- Setup notes with required dependencies and a working example run.
	- Dockerfile or environment spec (optional) for reproducibility.
- Owner: Leon (Installation+Configuration+Review of repo), Aron, Paul, Manuel, Masiar (Review of repo)
- Timeline: 1 week (setup + smoke tests)
- Status: Not started

### WP3 — Learning pipeline for Task 1
- Goal: Create a full training pipeline (data -> training -> evaluation) for Task 1 so we can iterate quickly.
- Tasks:
	- Define Task 1 metrics and success criteria.
	- Implement data preprocessing and dataset API.
	- Wire up training loop, checkpoints, and evaluation scripts.
- Deliverables:
	- Training script, config examples, and evaluation notebook.
	- Example trained model checkpoint.
- Owner: TBD
- Timeline: 2–4 weeks (iterative)
- Status: Not started

---

## Meeting 2

### WP1 - Branches auf Github aufräumen (Masiar)
### WP2 - Kamera einbauen in IsaacLab (Leon)
### WP3 - Deployment RL-Policy (Team)
### WP4 - Testing of different reward shaping methods in IsaacLab
### WP5 - Collection Dagger-Data und Retraining of Policy (Paul & Masiar)
### WP6 - Sim-to-Real Research (Team)

