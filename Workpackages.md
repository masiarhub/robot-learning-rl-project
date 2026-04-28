# Workpackages

This document lists the current workpackages for the robot-learning RL project with concise goals, tasks, deliverables, owners, timelines and status.

## WP1 — Data collection pipeline (real robot)
- Goal: Build a robust data collection pipeline to record trajectories and sensor data from the real robot for training and evaluation.
- Tasks:
	- Verify existing data storage and formats.
	- Implement consistent logging (timestamps, metadata, calibration IDs).
	- Add simple playback/visualization tools for quick inspection.
- Deliverables:
	- Working data-collection script + README for usage.
	- Example dataset and validation script to confirm integrity.
- Owner: TBD
- Timeline: 1–2 weeks (initial)
- Status: Not started / initial review done

## WP2 — Isaac Gym / Isaac Lab setup
- Goal: Install and configure Isaac Lab (or Isaac Gym) experiments to enable fast simulation-based prototyping.
- Tasks:
	- Review the Isaac repo and existing examples.
	- Run baseline example environments and confirm GPU/sim compatibility.
	- Try a few ready-made setups to validate workflow.
- Deliverables:
	- Setup notes with required dependencies and a working example run.
	- Dockerfile or environment spec (optional) for reproducibility.
- Owner: TBD
- Timeline: 1 week (setup + smoke tests)
- Status: Not started

## WP3 — Learning pipeline for Task 1
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

How to contribute
- Assign yourself as owner by editing this file and adding your name next to the relevant WP.
- For small changes, open a draft PR with the updated task/owner/timeline.

If you'd like, I can:
- add owners and concrete dates based on the team availability,
- translate the file to German, or
- create individual issue templates for each WP.