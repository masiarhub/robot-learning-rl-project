"""
Place-into-Bowl environment for SO-101 (Project 3, Eval 1).

Adapted from squint/envs/place.py:
  - The rectangular bin is replaced by a procedural bowl (disk floor + ring of
    cylinder segments forming a low rim) so that DR over rim-radius/height
    works per-env without external STL assets.
  - The bowl XY position is exposed in obs as `target_bowl_pos` (in robot frame),
    matching the project spec ("target locations are specified as (x, y, z) in
    the robot frame").
  - Bowl XY can be FIXED at construction time via `target_bowl_pos=(x,y)` so
    evaluation rollouts can pin the location specified by the TAs.
  - Table top color forced to ~#B8ADA9 (project spec).
  - Cube size DR is collapsed to a single-value range (project spec: "Object
    with a fixed size").

Registered envs:
  SO101PlaceBowlCube-v1   (random bowl + cube positions, DR on)
  SO101PlaceBowlCubeFixed-v1  (fixed bowl pos via kwarg, item still random)
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import dacite
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from .robot.so100 import SO100
from .robot.so101 import SO101


# Project spec table color #B8ADA9 -> normalized RGB
TABLE_COLOR_RGB = (0xB8 / 255.0, 0xAD / 255.0, 0xA9 / 255.0)


@dataclass
class PlaceBowlRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for PlaceBowl task."""
    # Noisy joint positions for better sim2real
    robot_qpos_noise_std: float = np.deg2rad(5)

    # Cube: project spec says fixed size — keep range collapsed to a single value.
    # Override at construction time if your real cubes differ.
    cube_half_size_range: Sequence[float] = (0.0125, 0.0125)  # 25mm cubes

    # Bowl geometry randomization (per-env, only active when DR is on)
    bowl_inner_radius_range: Sequence[float] = (0.045, 0.060)  # 9-12 cm diameter
    bowl_rim_height_range: Sequence[float] = (0.018, 0.028)
    bowl_rim_thickness: float = 0.004

    # Item physical randomization
    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)
    randomize_item_color: bool = True  # blocks are different colors per spec


class PlaceBowl(DefaultCameraEnv):
    """
    Pick up a single cube and place it inside a bowl.

    Goal conditioning:
        - The bowl XY position (in the robot base frame) is added to the state
          observation as `target_bowl_pos`.
        - If `target_bowl_pos` is given at construction, the bowl is pinned
          there (no XY randomization for the bowl).

    Success Conditions:
        - cube center is inside bowl XY (within inner radius)
        - cube center is below the rim (z < rim_height)
        - robot is not touching the cube or the bowl
        - robot is static
    """

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = [
        "none", "state", "state_dict", "rgb",
        "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
        "rgb+depth+segmentation", "rgb+depth+segmentation+state",
    ]
    agent: Union[SO100, SO101]

    BOWL_NUM_SEGMENTS = 16  # cylinder segments forming the rim

    def __init__(
        self,
        *args,
        robot_uids: str = "so101",
        control_mode: str = "pd_joint_target_delta_pos",
        domain_randomization_config: Union[PlaceBowlRandomizationConfig, dict] = PlaceBowlRandomizationConfig(),
        domain_randomization: bool = True,
        spawn_box_pos: Sequence[float] = (0.3, 0.0),
        spawn_box_half_size: float = 0.10,
        target_bowl_pos: Optional[Sequence[float]] = None,
        **kwargs,
    ):
        # Robot-specific configuration (mirrors squint place.py)
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        else:
            raise ValueError(f"Unsupported robot_uids: {robot_uids}")

        # Resolve config
        self.domain_randomization_config = PlaceBowlRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PlaceBowlRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PlaceBowlRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size

        # Fixed bowl XY (set by TA-supplied target at eval time). When None,
        # bowl XY is sampled per-episode within the spawn box.
        self._fixed_target_bowl_pos = (
            None if target_bowl_pos is None else np.asarray(target_bowl_pos, dtype=np.float32)[:2]
        )

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    # ----- Agent loading mirrors squint place.py --------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=(
                self.domain_randomization
                and self.domain_randomization_config.robot_color == "random"
            ),
        )

    # ----- Scene loading --------------------------------------------------------

    def _load_scene(self, options: dict):
        cfg = self.domain_randomization_config

        # Table with project-spec gray color
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        try:
            # Best-effort: not all ManiSkill versions expose this; non-fatal.
            for shape in self.table_scene.table.find_component_by_type(
                sapien.render.RenderBodyComponent
            ).render_shapes:
                for part in shape.parts:
                    part.material.set_base_color(list(TABLE_COLOR_RGB) + [1.0])
        except Exception:
            pass

        # ----- Item (cube) per-env actors so size/color/friction can vary -----
        if cfg.randomize_item_color and self.domain_randomization:
            cube_colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
        else:
            cube_colors = np.zeros((self.num_envs, 3))
            cube_colors[:, 0] = 1.0  # default red
        cube_colors = np.concatenate([cube_colors, np.ones((self.num_envs, 1))], axis=-1)

        cube_half_lo, cube_half_hi = cfg.cube_half_size_range
        if self.domain_randomization and cube_half_hi > cube_half_lo:
            half_sizes = self._batched_episode_rng.uniform(low=cube_half_lo, high=cube_half_hi)
        else:
            half_sizes = np.ones(self.num_envs) * (cube_half_lo + cube_half_hi) / 2

        if self.domain_randomization:
            frictions = self._batched_episode_rng.uniform(
                low=cfg.item_friction_range[0], high=cfg.item_friction_range[1])
            densities = self._batched_episode_rng.uniform(
                low=cfg.item_density_range[0], high=cfg.item_density_range[1])
        else:
            frictions = np.ones(self.num_envs) * (cfg.item_friction_range[0] + cfg.item_friction_range[1]) / 2
            densities = np.ones(self.num_envs) * (cfg.item_density_range[0] + cfg.item_density_range[1]) / 2

        self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
        self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=float(frictions[i]),
                dynamic_friction=float(frictions[i]),
                restitution=0.0,
            )
            builder.add_box_collision(half_size=[float(half_sizes[i])] * 3, material=material,
                                      density=float(densities[i]))
            builder.add_box_visual(
                half_size=[float(half_sizes[i])] * 3,
                material=sapien.render.RenderMaterial(base_color=cube_colors[i]),
            )
            builder.initial_pose = sapien.Pose(p=[0.2, 0.0, float(half_sizes[i])])
            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # ----- Bowl per-env: thin disk floor + ring of small cylinders ---------
        if self.domain_randomization:
            inner_radii = self._batched_episode_rng.uniform(
                low=cfg.bowl_inner_radius_range[0], high=cfg.bowl_inner_radius_range[1])
            rim_heights = self._batched_episode_rng.uniform(
                low=cfg.bowl_rim_height_range[0], high=cfg.bowl_rim_height_range[1])
        else:
            inner_radii = np.ones(self.num_envs) * sum(cfg.bowl_inner_radius_range) / 2
            rim_heights = np.ones(self.num_envs) * sum(cfg.bowl_rim_height_range) / 2

        self.bowl_inner_radii = common.to_tensor(inner_radii, device=self.device)
        self.bowl_rim_heights = common.to_tensor(rim_heights, device=self.device)

        bowl_color = sapien.render.RenderMaterial(base_color=[0.95, 0.95, 0.95, 1.0])
        rim_thick = cfg.bowl_rim_thickness
        floor_thickness = 0.005
        self.bowl_floor_thickness = floor_thickness

        bowls = []
        for i in range(self.num_envs):
            r = float(inner_radii[i])
            h = float(rim_heights[i])
            outer_r = r + rim_thick
            builder = self.scene.create_actor_builder()

            # Floor disc (thin cylinder), oriented vertically (axis along z)
            floor_pose = sapien.Pose(p=[0.0, 0.0, floor_thickness / 2],
                                     q=euler2quat(0, np.pi / 2, 0))
            builder.add_cylinder_collision(radius=outer_r, half_length=floor_thickness / 2, pose=floor_pose)
            builder.add_cylinder_visual(radius=outer_r, half_length=floor_thickness / 2, pose=floor_pose,
                                        material=bowl_color)

            # Rim segments around the circle
            for s in range(self.BOWL_NUM_SEGMENTS):
                ang = 2 * np.pi * s / self.BOWL_NUM_SEGMENTS
                cx = (r + rim_thick / 2) * np.cos(ang)
                cy = (r + rim_thick / 2) * np.sin(ang)
                seg_pose = sapien.Pose(p=[cx, cy, h / 2 + floor_thickness])
                # Small upright box (axis-aligned). Width tangent to circle.
                tangent_w = (2 * np.pi * (r + rim_thick / 2) / self.BOWL_NUM_SEGMENTS) / 2
                builder.add_box_collision(
                    pose=seg_pose,
                    half_size=[rim_thick / 2, tangent_w, h / 2],
                )
                builder.add_box_visual(
                    pose=seg_pose,
                    half_size=[rim_thick / 2, tangent_w, h / 2],
                    material=bowl_color,
                )

            builder.initial_pose = sapien.Pose(p=[-0.2, 0.0, 0.0])
            builder.set_scene_idxs([i])
            bowl_actor = builder.build_static(name=f"bowl-{i}")
            bowls.append(bowl_actor)
            self.remove_from_state_dict_registry(bowl_actor)

        self.bowl = Actor.merge(bowls, name="bowl")
        self.add_to_state_dict_registry(self.bowl)

        # Greenscreen: keep robot, item, bowl in foreground
        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bowl)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self._load_camera_mount()
        self._randomize_robot_color()

        # Goal site (visual only)
        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.008, material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]),
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    # ----- Episode init ---------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

            # Random initial qpos
            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(
                Pose.create_from_pq(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot))
            )

            spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.spawn_box_pos[0], self.spawn_box_pos[1], 0]
            )

            # Sample non-overlapping XY positions for cube and bowl
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            item_radius = self.item_half_sizes.max().item() + 0.01
            bowl_radius = self.bowl_inner_radii.max().item() + 0.02

            item_xy_offset = sampler.sample(item_radius, 100)

            # Bowl: pinned or random
            if self._fixed_target_bowl_pos is not None:
                bowl_xy_offset = (
                    torch.tensor(self._fixed_target_bowl_pos, device=self.device, dtype=torch.float32)
                    .view(1, 2).expand(b, -1)
                    - spawn_center[env_idx, :2]
                )
            else:
                bowl_xy_offset = sampler.sample(bowl_radius, 100, verbose=False)

            # Item pose
            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = spawn_center[env_idx, :2] + item_xy_offset
            item_xyz[:, 2] = self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            # Bowl pose (z = 0, sits on table)
            bowl_xyz = torch.zeros((b, 3))
            bowl_xyz[:, :2] = spawn_center[env_idx, :2] + bowl_xy_offset
            bowl_xyz[:, 2] = 0.0
            self.bowl.set_pose(Pose.create_from_pq(bowl_xyz))

            # Goal site sits just above the bowl center
            goal_xyz = bowl_xyz.clone()
            goal_xyz[:, 2] = self.bowl_floor_thickness + self.item_half_sizes[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # ----- Observations ---------------------------------------------------------

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        obs = dict()
        # Always expose target_bowl_pos in the robot base frame (XY only;
        # Z is implicit since the bowl sits on the table).
        bowl_xy = self.bowl.pose.p[:, :2]
        target_bowl_pos = torch.cat(
            [bowl_xy, torch.zeros((self.num_envs, 1), device=self.device)],
            dim=-1,
        )
        obs.update(target_bowl_pos=target_bowl_pos)

        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bowl_pose=self.bowl.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bowl_pos=self.bowl.pose.p - self.agent.tcp_pos,
                item_to_bowl_pos=self.bowl.pose.p - self.item.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bowl_inner_radius=self.bowl_inner_radii,
                    bowl_rim_height=self.bowl_rim_heights,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    # ----- Termination / reward -------------------------------------------------

    def evaluate(self):
        item_pos = self.item.pose.p
        bowl_pos = self.bowl.pose.p

        offset_xy = item_pos[:, :2] - bowl_pos[:, :2]
        dist_xy = torch.linalg.norm(offset_xy, dim=-1)
        is_item_above_bowl = dist_xy < self.bowl_inner_radii

        is_item_below_rim = item_pos[:, 2] < (self.bowl_rim_heights + self.bowl_floor_thickness + 0.005)
        is_item_in_bowl = is_item_above_bowl & is_item_below_rim

        item_lifted = item_pos[..., -1] >= (self.item_half_sizes + 1e-3)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bowl = self.agent.is_touching(self.bowl)
        robot_touching_item = self.agent.is_touching(self.item)

        success = (
            is_item_in_bowl
            & (~robot_touching_item)
            & is_robot_static
            & (~robot_touching_bowl)
        )

        return {
            "dist_xy": dist_xy,
            "is_item_in_bowl": is_item_in_bowl,
            "is_item_above_bowl": is_item_above_bowl,
            "is_item_below_rim": is_item_below_rim,
            "item_vel": item_vel,
            "item_lifted": item_lifted,
            "is_item_static": is_item_static,

            "success": success,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bowl": robot_touching_bowl,
            "robot_touching_item": robot_touching_item,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reaching reward
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pose.p - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        # Place reward (use bowl center + slightly-above-rim as the goal column)
        item_pos = self.item.pose.p
        bowl_pos = self.bowl.pose.p
        goal_xyz = bowl_pos.clone()
        goal_xyz[..., 2] = self.bowl_floor_thickness + self.item_half_sizes

        item_to_goal = torch.linalg.norm(goal_xyz - item_pos, dim=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal)

        item_to_goal_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        z_far_target = (
            goal_xyz[..., 2:] + (self.bowl_rim_heights[:, None] * 2) + 0.03 - item_pos[..., 2:]
        )
        item_to_goal_z_far = torch.linalg.norm(z_far_target, dim=1)
        item_to_goal_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        close = item_to_goal_xy <= self.bowl_inner_radii
        item_to_goal_z = torch.where(close, item_to_goal_z_close, item_to_goal_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_z)
        place_reward = place_reward_final + place_reward_z

        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (
            (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)
        )

        # Grasped: 3 + place_reward
        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        # Above bowl: 4 + place + drop + openness + static
        is_item_dropped = (~info["robot_touching_item"]).float()
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], axis=1)
        static_robot_reward = 1 - torch.tanh(robot_v * 10)
        reward[info["is_item_above_bowl"]] = (
            4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward
        )[info["is_item_above_bowl"]]

        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bowl"].float()
        reward -= 1 * (~info["item_lifted"]).float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


# ----- Registered envs ---------------------------------------------------------

@register_env("SO101PlaceBowlCube-v1", max_episode_steps=50)
class PlaceBowlCube(PlaceBowl):
    """Random bowl + cube positions, full DR. Used for Eval 1 training."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_env("SO101PlaceBowlCubeFixed-v1", max_episode_steps=80)
class PlaceBowlCubeFixed(PlaceBowl):
    """Bowl pinned via target_bowl_pos kwarg; longer episodes for evaluation."""
    def __init__(self, *args, target_bowl_pos: Optional[Sequence[float]] = (0.30, 0.05), **kwargs):
        super().__init__(*args, target_bowl_pos=target_bowl_pos, **kwargs)
