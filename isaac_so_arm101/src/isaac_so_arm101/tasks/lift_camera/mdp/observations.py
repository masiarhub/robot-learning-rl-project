# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import torch.nn.functional as F



def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_offset: float = 0.0,
) -> torch.Tensor:
    """The position of the object in the robot's root frame, with an optional z offset in world frame.

    height_offset is applied in world frame before the frame transform, so it shifts the
    returned position upward by that amount (e.g. 0.12 m above the bowl centre).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3].clone()
    object_pos_w[:, 2] += height_offset
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


_POOL_KERNEL = 8                                                                                                                                                             
                                                                                                                                                                               
                                                                                                                                                                               
def wrist_camera_image(                                                                                                                                                      
      env: ManagerBasedRLEnv,                                                                                                                                                  
      sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),                                                                                                             
      flatten: bool = True,                                                                                                                                                    
  ) -> torch.Tensor:                                                                                                                                                           
      sensor = env.scene.sensors[sensor_cfg.name]                                                                                                                              
      cfg = sensor.cfg                                                                                                                                                         
      out_h = cfg.height // _POOL_KERNEL                                                                                                                                       
      out_w = cfg.width // _POOL_KERNEL                                                                                                                                        
                                                                                                                                                                               
      if not env.sim.is_playing():    # or sensor.frame.max() == 0 (?)                                                                                                                                    
          if flatten:
              return torch.zeros(env.num_envs, 3 * out_h * out_w, device=env.device)                                                                                           
          return torch.zeros(env.num_envs, 3, out_h, out_w, device=env.device)                                                                                        
   
      raw = sensor.data.output["rgb"]                                           # [B, H, W, 3]                                                                                 
      img = raw.permute(0, 3, 1, 2).float().clamp(0.0, 255.0) / 255.0         # [B, 3, H, W]
      img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)                                                                                                             
      squinted = F.avg_pool2d(img, kernel_size=_POOL_KERNEL, stride=_POOL_KERNEL)    

    # DEBUGGING print wrist cam POV
      if env.common_step_counter == 2:
        import imageio, os
        raw_np = raw[0].cpu().numpy()  # [H, W, 3], uint8
        save_path = os.path.expanduser("~/robot_learning/wrist_cam_debug.png")
        imageio.imwrite(save_path, raw_np)
        print(f"[DEBUG] Saved to {save_path} | shape={raw.shape} | max={raw.max():.1f}")                                                                                                  
                                                                                                                                                                               
      if flatten:                                                                                                                                                              
          return squinted.flatten(start_dim=1)                                                                                                                                 
      return squinted     


# _POOL_KERNEL = 8


# def wrist_camera_image(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
#     flatten: bool = True,
# ) -> torch.Tensor:
#     sensor = env.scene.sensors[sensor_cfg.name]
#     cfg = sensor.cfg
#     out_h = cfg.height // _POOL_KERNEL
#     out_w = cfg.width // _POOL_KERNEL

#     if not env.sim.is_playing():
#         if flatten:
#             return torch.zeros(env.num_envs, 3 * out_h * out_w, device=env.device)
#         return torch.zeros(env.num_envs, 3, out_h, out_w, device=env.device)

#     raw = sensor.data.output["rgb"]                                           # [B, H, W, 3]
#     img = raw.permute(0, 3, 1, 2).float().clamp(0.0, 255.0) / 255.0         # [B, 3, H, W]
#     img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
#     squinted = F.avg_pool2d(img, kernel_size=_POOL_KERNEL, stride=_POOL_KERNEL)

#     if flatten:
#         return squinted.flatten(start_dim=1)
#     return squinted


# def wrist_camera_image(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
#     flatten: bool = True,
# ) -> torch.Tensor:
#     sensor = env.scene.sensors[sensor_cfg.name]
#     raw = sensor.data.output["rgb"]                         # [B, H, W, 3]
#     img = raw.permute(0, 3, 1, 2).float() / 255.0          # [B, 3, H, W]
    
#     # Convert to grayscale — standard luminance weights
#     gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
#     gray = gray.unsqueeze(1)                                # [B, 1, H, W]
    
#     squinted = F.avg_pool2d(gray, kernel_size=8, stride=8)  # [B, 1, 16, 16]

#     assert raw.max() > 0.0, "Camera output all zeros — did you forget --enable_cameras?"

#     if flatten:
#         return squinted.flatten(start_dim=1)                # [B, 256]
#     return squinted