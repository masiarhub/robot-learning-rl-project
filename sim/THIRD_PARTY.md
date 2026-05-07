# Third-party attribution

Most files in this directory are adapted from:

- **Squint** (https://github.com/aalmuzairee/squint), MIT license
  - Author: Abdulaziz Almuzairee, Henrik I. Christensen (UC San Diego)
  - Paper: https://arxiv.org/abs/2602.21203

The full Squint LICENSE is preserved at `LICENSE.squint` (copy of the upstream
MIT LICENSE file). Original `README.md` and per-file headers are unchanged
where present.

## Files copied verbatim from Squint

```
envs/__init__.py                (extended; original noted in comments)
envs/base_random_env.py
envs/black_overlay.png
envs/reach.py
envs/lift.py
envs/_place_original.py         (original squint Place; not auto-imported)
envs/stack.py
envs/robot/{so100,so101}.py
envs/robot/so101.{urdf,srdf}
envs/robot/meshes/*
deploy_utils/{manipulator,robot_config,tune_camera}.py
deploy_utils/blender_stls/*
examples/visualize_sim.py
train_sim.py                    (renamed from train_squint.py)
deploy_sim.py                   (renamed from deploy.py)
utils.py
environment.yaml
```

## New files written for Project 3

```
envs/place_bowl.py
envs/targeted_pick_place.py
envs/multi_block_eval.py
policies/sequential_runner.py
deploy_sim_eval.py
configs/eval1_place_bowl.yaml
configs/eval2_targeted.yaml
configs/eval3_sequential.yaml
scripts/script_pickplace.py
scripts/seed_buffer_from_dataset.py
tests/test_envs_register.py
README.md
THIRD_PARTY.md
```
