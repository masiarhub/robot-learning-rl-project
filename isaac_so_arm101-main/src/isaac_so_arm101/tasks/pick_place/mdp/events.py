from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from pxr import UsdShade, Gf, Usd
import omni.usd
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BLOCK_COLORS = [
    (0.0, 0.0, 1.0),   # blue
    (0.5, 0.0, 0.5),   # purple
    (1.0, 0.5, 0.0),   # orange
    (1.0, 0.0, 0.0),   # red
    (1.0, 1.0, 0.0),   # yellow
    (0.0, 0.8, 0.0),   # green
]

def _set_object_color(env, env_ids: torch.Tensor, object_cfg: SceneEntityCfg) -> None:
    stage = omni.usd.get_context().get_stage()
    indices = torch.randint(len(BLOCK_COLORS), (len(env_ids),))
    for i, env_id in enumerate(env_ids.tolist()):
        color = BLOCK_COLORS[indices[i].item()]
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
        if not prim.IsValid():
            continue
        for desc in Usd.PrimRange(prim):
            if not desc.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(desc)
            diffuse = shader.GetInput("diffuseColor")
            if diffuse:
                diffuse.Set(Gf.Vec3f(*color))
                break

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
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    bowl_xy_range: dict[str, tuple[float, float]] | None = None,
    object_xy_range: dict[str, tuple[float, float]] | None = None,
    min_xy_distance: float = 0.12,
) -> None:
    """Reset bowl and object with XY non-overlap constraint."""
    if bowl_xy_range is None:
        bowl_xy_range = {"x": (0.28, 0.52), "y": (-0.22, 0.22)}
    if object_xy_range is None:
        object_xy_range = {"x": (0.22, 0.48), "y": (-0.22, 0.22)}
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    obj_state = obj.data.default_root_state[env_ids].clone()
    bowl_xy = _sample_xy(env, env_ids, bowl_xy_range)
    obj_xy = _sample_xy(env, env_ids, object_xy_range)
    max_resamples = 32
    for _ in range(max_resamples):
        too_close = torch.norm(obj_xy - bowl_xy, dim=-1) < min_xy_distance
        if not torch.any(too_close):
            break
        count = int(too_close.sum().item())
        obj_xy[too_close] = _sample_xy(env, env_ids[too_close], object_xy_range)[:count]
    bowl_state[:, 0:2] = bowl_xy
    obj_state[:, 0:2] = obj_xy
    bowl_state[:, 7:13] = 0.0
    obj_state[:, 7:13] = 0.0
    bowl.write_root_state_to_sim(bowl_state, env_ids=env_ids)
    obj.write_root_state_to_sim(obj_state, env_ids=env_ids)
    _set_object_color(env, env_ids, object_cfg)