from __future__ import annotations
from typing import TYPE_CHECKING
import math
import random
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg, EventTermCfg as _EventTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.sim import find_matching_prims, get_current_stage
from isaaclab.envs.mdp.events import randomize_rigid_body_material as _RandMaterial
from pxr import UsdShade, Gf, Sdf, Usd
import omni.usd
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_material_term_cache: dict = {}

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
        prim_path = f"/World/envs/env_{env_id}/Object"
        prim = stage.GetPrimAtPath(prim_path)
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

def randomize_dome_light(
    env,
    env_ids: torch.Tensor | None,
    prim_path: str = "/World/light",
    intensity_range: tuple[float, float] = (400.0, 1200.0),
    color_range: tuple[float, float] = (0.65, 0.85),
) -> None:
    intensity = random.uniform(*intensity_range)
    grey = random.uniform(*color_range)
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    with Sdf.ChangeBlock():
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if intensity_attr.IsValid():
            intensity_attr.Set(intensity)
        color_attr = prim.GetAttribute("inputs:color")
        if color_attr.IsValid():
            color_attr.Set(Gf.Vec3f(grey, grey, grey))

def randomize_sphere_light(
    env,
    env_ids: torch.Tensor | None,
    intensity_range: tuple[float, float] = (3000.0, 8000.0),
    color_range: tuple[float, float] = (0.65, 0.85),
    radius_range: tuple[float, float] = (0.1, 0.4),
    pos_x_range: tuple[float, float] = (-0.2, 0.5),
    pos_y_range: tuple[float, float] = (-0.4, 0.4),
    pos_z_range: tuple[float, float] = (0.3, 1.0),
) -> None:
    stage = get_current_stage()
    with Sdf.ChangeBlock():
        for env_id in env_ids.tolist():
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/SphereLight")
            if not prim.IsValid():
                continue
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if intensity_attr.IsValid():
                intensity_attr.Set(random.uniform(*intensity_range))
            c = random.uniform(*color_range)
            color_attr = prim.GetAttribute("inputs:color")
            if color_attr.IsValid():
                color_attr.Set(Gf.Vec3f(c, c, c))
            radius_attr = prim.GetAttribute("inputs:radius")
            if radius_attr.IsValid():
                radius_attr.Set(random.uniform(*radius_range))
            translate_attr = prim.GetAttribute("xformOp:translate")
            if translate_attr.IsValid():
                translate_attr.Set(Gf.Vec3d(
                    random.uniform(*pos_x_range),
                    random.uniform(*pos_y_range),
                    random.uniform(*pos_z_range),
                ))

def reset_bowl_and_cube(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    # Placement point (robot base in local frame)
    placement_point: tuple[float, float] = (0.048, 0.0),
    # Bowl: annular ring + axis constraints
    bowl_dist_range: tuple[float, float] = (0.20, 0.40),
    bowl_x_min: float = 0.148,
    bowl_y_max: float = 0.20,
    bowl_radius: float = 0.12,
    # Cube: annular ring + axis constraints
    cube_dist_range: tuple[float, float] = (0.15, 0.30),
    cube_x_min: float = 0.148,
    cube_y_max: float = 0.20,
    # Legacy flat range support (ignored if dist_range is set)
    bowl_xy_range: dict[str, tuple[float, float]] | None = None,
    object_xy_range: dict[str, tuple[float, float]] | None = None,
    min_xy_distance: float = 0.12,
    # Sampling limits
    safe_fallback_after: int = 100,
    max_placement_tries: int = 200,
    safety_positions: list | None = None,
    cube_z_rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
) -> None:
    """Reset bowl and object with annular-ring sampling and occlusion-cone constraint."""

    if safety_positions is None:
        safety_positions = [
            (0.268, +0.000),
            (0.253, +0.143),
            (0.253, -0.143),
            (0.293, +0.114),
            (0.293, -0.114),
            (0.338, +0.000),
            (0.189, +0.169),
            (0.189, -0.169),
        ]

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    n = len(env_ids)
    px = float(placement_point[0])
    py = float(placement_point[1])
    r_lo_bowl, r_hi_bowl = float(bowl_dist_range[0]), float(bowl_dist_range[1])
    r_lo_cube, r_hi_cube = float(cube_dist_range[0]), float(cube_dist_range[1])

    def _sample_annulus(count: int, r_lo: float, r_hi: float) -> torch.Tensor:
        angle = torch.rand(count, device=env.device) * (2.0 * math.pi)
        r = torch.sqrt(torch.rand(count, device=env.device) * (r_hi**2 - r_lo**2) + r_lo**2)
        return torch.stack([px + r * torch.cos(angle), py + r * torch.sin(angle)], dim=1)

    # ── Bowl: rejection-sample from annular ring ──────────────────────────
    def _check_bowl(local_xy: torch.Tensor) -> torch.Tensor:
        x, y = local_xy[:, 0], local_xy[:, 1]
        d = torch.sqrt((x - px) ** 2 + (y - py) ** 2)
        return (d >= r_lo_bowl) & (d <= r_hi_bowl) & (x >= bowl_x_min) & (torch.abs(y) <= bowl_y_max)

    bowl_local_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
    needs_resample_bowl = ~_check_bowl(bowl_local_xy)
    for _ in range(max_placement_tries):
        if not needs_resample_bowl.any():
            break
        new_xy = _sample_annulus(n, r_lo_bowl, r_hi_bowl)
        bowl_local_xy[needs_resample_bowl] = new_xy[needs_resample_bowl]
        needs_resample_bowl[needs_resample_bowl.clone()] = ~_check_bowl(bowl_local_xy)[needs_resample_bowl]

    if needs_resample_bowl.any():
        print(f"[WARNING] reset_bowl_and_object: {needs_resample_bowl.sum().item()} env(s) failed bowl placement.")

    bx_local = bowl_local_xy[:, 0]
    by_local = bowl_local_xy[:, 1]

    # ── Cube constraint helpers ───────────────────────────────────────────
    def _in_occlusion_cone(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        vc_x, vc_y = cx - px, cy - py
        vb_x, vb_y = bx_local - px, by_local - py
        d_c = torch.sqrt(vc_x**2 + vc_y**2).clamp(min=1e-9)
        vc_hat_x, vc_hat_y = vc_x / d_c, vc_y / d_c
        proj = vb_x * vc_hat_x + vb_y * vc_hat_y
        perp = torch.abs(vb_x * vc_hat_y - vb_y * vc_hat_x)
        return (perp < bowl_radius) & (proj > 0.0) & (proj < d_c)

    def _check_cube(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        d = torch.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        in_ring   = (d >= r_lo_cube) & (d <= r_hi_cube)
        x_ok      = cx >= cube_x_min
        excl_ok   = torch.sqrt((cx - bx_local) ** 2 + (cy - by_local) ** 2) > bowl_radius
        cone_ok   = ~_in_occlusion_cone(cx, cy)
        y_band_ok = torch.abs(cy) <= cube_y_max
        return in_ring & x_ok & excl_ok & cone_ok & y_band_ok

    # ── Cube Phase 1: random rejection sampling ───────────────────────────
    cube_local_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
    needs_resample = ~_check_cube(cube_local_xy[:, 0], cube_local_xy[:, 1])

    for _ in range(safe_fallback_after):
        if not needs_resample.any():
            break
        new_xy = _sample_annulus(n, r_lo_cube, r_hi_cube)
        cube_local_xy[needs_resample] = new_xy[needs_resample]
        needs_resample[needs_resample.clone()] = ~_check_cube(
            cube_local_xy[:, 0], cube_local_xy[:, 1]
        )[needs_resample]

    # ── Cube Phase 2: safety positions fallback ───────────────────────────
    if needs_resample.any():
        for sx, sy in safety_positions:
            if not needs_resample.any():
                break
            sp_x = torch.full((n,), float(sx), device=env.device)
            sp_y = torch.full((n,), float(sy), device=env.device)
            can_fix = needs_resample & _check_cube(sp_x, sp_y)
            if can_fix.any():
                cube_local_xy[can_fix, 0] = float(sx)
                cube_local_xy[can_fix, 1] = float(sy)
                needs_resample[can_fix] = False

    if needs_resample.any():
        print(f"[WARNING] reset_bowl_and_object: {needs_resample.sum().item()} env(s) failed cube placement.")

    env_origins = env.scene.env_origins[env_ids]

    # ── Write bowl state ──────────────────────────────────────────────────
    bowl_state = bowl.data.default_root_state[env_ids].clone()
    bowl_state[:, 0] = bowl_local_xy[:, 0] + env_origins[:, 0]
    bowl_state[:, 1] = bowl_local_xy[:, 1] + env_origins[:, 1]
    bowl_state[:, 2] = bowl.data.default_root_state[env_ids, 2] + env_origins[:, 2]
    bowl_state[:, 7:] = 0.0
    bowl.write_root_pose_to_sim(bowl_state[:, :7], env_ids=env_ids)
    bowl.write_root_velocity_to_sim(bowl_state[:, 7:], env_ids=env_ids)

    # ── Write cube state ──────────────────────────────────────────────────
    obj_state = obj.data.default_root_state[env_ids].clone()
    obj_state[:, 0] = cube_local_xy[:, 0] + env_origins[:, 0]
    obj_state[:, 1] = cube_local_xy[:, 1] + env_origins[:, 1]
    obj_state[:, 2] = obj.data.default_root_state[env_ids, 2] + env_origins[:, 2]

    n = len(env_ids)
    z_angle = math_utils.sample_uniform(
        cube_z_rotation_range[0], cube_z_rotation_range[1], (n,), device=env.device
    )
    z_quat = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device),
        torch.zeros(n, device=env.device),
        z_angle,
    )
    obj_state[:, 3:7] = math_utils.quat_mul(z_quat, obj.data.default_root_state[env_ids, 3:7])
    obj_state[:, 7:] = 0.0
    obj.write_root_pose_to_sim(obj_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(obj_state[:, 7:], env_ids=env_ids)

    _set_object_color(env, env_ids, object_cfg)

def randomize_rigid_body_material(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple,
    dynamic_friction_range: tuple,
    restitution_range: tuple,
    num_buckets: int,
    make_consistent: bool = False,
) -> None:
    key = asset_cfg.name
    if key not in _material_term_cache:
        cfg = _EventTermCfg(
            func=_RandMaterial,
            mode="reset",
            params={
                "asset_cfg": asset_cfg,
                "static_friction_range": static_friction_range,
                "dynamic_friction_range": dynamic_friction_range,
                "restitution_range": restitution_range,
                "num_buckets": num_buckets,
                "make_consistent": make_consistent,
            },
        )
        _material_term_cache[key] = _RandMaterial(cfg, env)
    _material_term_cache[key](
        env, env_ids,
        asset_cfg=asset_cfg,
        static_friction_range=static_friction_range,
        dynamic_friction_range=dynamic_friction_range,
        restitution_range=restitution_range,
        num_buckets=num_buckets,
        make_consistent=make_consistent,
    )
def randomize_table_friction(
    env,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float] = (0.3, 1.0),
    dynamic_friction_range: tuple[float, float] = (0.2, 0.8),
) -> None:
    """Randomize table surface friction via USD physics material."""
    import random
    stage = get_current_stage()
    static_friction = random.uniform(*static_friction_range)
    dynamic_friction = random.uniform(*dynamic_friction_range)
    
    with Sdf.ChangeBlock():
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "Table" not in path:
                continue
            # Look for physics material
            from pxr import UsdPhysics
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                mat_api = UsdPhysics.MaterialAPI(prim)
                static_attr = mat_api.GetStaticFrictionAttr()
                dynamic_attr = mat_api.GetDynamicFrictionAttr()
                if static_attr.IsValid():
                    static_attr.Set(static_friction)
                if dynamic_attr.IsValid():
                    dynamic_attr.Set(dynamic_friction)