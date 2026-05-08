from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sample_xy(env, env_ids: torch.Tensor, xy_range: dict[str, tuple[float, float]]) -> torch.Tensor:
    num = env_ids.numel()
    x_min, x_max = xy_range["x"]
    y_min, y_max = xy_range["y"]
    x = torch.empty(num, device=env.device).uniform_(x_min, x_max)
    y = torch.empty(num, device=env.device).uniform_(y_min, y_max)
    return torch.stack([x, y], dim=-1)


def reset_bowl_and_object_non_overlapping(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    bowl_bottom_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    bowl_wall_cfg: SceneEntityCfg = SceneEntityCfg("bowl_wall"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bowl_xy_range: dict[str, tuple[float, float]] | None = None,
    object_xy_range: dict[str, tuple[float, float]] | None = None,
    min_xy_distance: float = 0.12,
) -> None:
    """Reset bowl (bottom + wall) and object with XY non-overlap constraint."""
    if bowl_xy_range is None:
        bowl_xy_range = {"x": (0.28, 0.52), "y": (-0.22, 0.22)}
    if object_xy_range is None:
        object_xy_range = {"x": (0.22, 0.48), "y": (-0.22, 0.22)}

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    bowl_bottom: RigidObject = env.scene[bowl_bottom_cfg.name]
    bowl_wall: RigidObject = env.scene[bowl_wall_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    bowl_bottom_state = bowl_bottom.data.default_root_state[env_ids].clone()
    bowl_wall_state = bowl_wall.data.default_root_state[env_ids].clone()
    obj_state = obj.data.default_root_state[env_ids].clone()

    bowl_xy = _sample_xy(env, env_ids, bowl_xy_range)
    obj_xy = _sample_xy(env, env_ids, object_xy_range)

    max_resamples = 32
    for _ in range(max_resamples):
        too_close = torch.norm(obj_xy - bowl_xy, dim=-1) < min_xy_distance
        if not torch.any(too_close):
            break
        count = int(too_close.sum().item())
        resampled = _sample_xy(env, env_ids[too_close], object_xy_range)
        obj_xy[too_close] = resampled[:count]

    bowl_bottom_state[:, 0:2] = bowl_xy
    bowl_wall_state[:, 0:2] = bowl_xy
    obj_state[:, 0:2] = obj_xy

    # Keep z/quat defaults from cfg and clear reset velocities.
    bowl_bottom_state[:, 7:13] = 0.0
    bowl_wall_state[:, 7:13] = 0.0
    obj_state[:, 7:13] = 0.0

    bowl_bottom.write_root_state_to_sim(bowl_bottom_state, env_ids=env_ids)
    bowl_wall.write_root_state_to_sim(bowl_wall_state, env_ids=env_ids)
    obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
