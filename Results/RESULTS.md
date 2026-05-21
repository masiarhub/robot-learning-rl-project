# Results

## Enable Recording with view of one env
```bash
--video --video_length 800 --camera_eye -0.129 1.227 0.434 --camera_target -10 1.2 -3.0 --video_fps 25
```


## Task 1

### Teacher
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-One-Teacher-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_1/task_1_teacher_ppo/model_2550.pt
```

### Part One
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-One-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_1/task_1_visual_part_one/visual_general_model_4999.pt
```

## Task 2

### Teacher
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Two-Teacher-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_2/task_2_teacher_ppo/model_8699.pt
```

### Part One
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Two-PartOne-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_2/task_2_visual_part_one/visual_general_model_4999.pt
```


# Task 3

### Teacher
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-Teacher-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_teacher_ppo/model_7550.pt
```

### Part One
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_one/model_5600.pt
```

### Part Two
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-PartTwo-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_two/visual_general_model_4999.pt
```

### Part Three
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Three-PartThree-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_part_three/visual_general_model_4999.pt
```

### Bonus
```bash
python isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py \
  --task Isaac-SO-ARM101-Task-Bonus-VisualCoord-Play-v0 \
  --num_envs 4 \
  --checkpoint Results/task_3/task_3_visual_bonus/model_500.pt
```


## Extras

### Finding Camera Angle in isaacsim

```python
import carb
import omni.usd
from pxr import Usd, UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
cam = stage.GetPrimAtPath("/OmniverseKit_Persp")
xform = UsdGeom.Xformable(cam)
world_xform =xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

t = world_xform.ExtractTranslation()
eye = [round(t[0], 3), round(t[1], 3), round(t[2], 3)]

rot = Gf.Matrix3d(world_xform.ExtractRotationMatrix())
fwd = rot * Gf.Vec3d(0, 1, 0)  # +Y is forward in Isaac Sim's viewport camera
target = [round(t[0] + fwd[0], 3), round(t[1] + fwd[1], 3), round(t[2]
+ fwd[2], 3)]

carb.log_warn(f"--camera_eye {eye[0]} {eye[1]} {eye[2]}")
carb.log_warn(f"--camera_target {target[0]} {target[1]} {target[2]}")
```
