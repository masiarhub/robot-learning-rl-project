"""
Targeted Pick-and-Place in a 2-block clutter (Project 3, Eval 2).

Two cubes of different known colors are placed adjacent to each other.
The policy is given:
  - the wrist camera RGB
  - the target color (one-hot over a fixed palette)
  - the target bowl XY position (in robot base frame)

Success: the *target* cube ends up inside the bowl with the gripper released
and the robot static. Picking the wrong cube into the bowl is treated as a
terminal failure so episodes do not waste budget.

Registered envs:
  SO101TargetedPlace-v1        (random target color + positions, DR on)
  SO101TargetedPlaceFixed-v1   (deterministic for eval rollouts)
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
from .place_bowl import TABLE_COLOR_RGB, PlaceBowl  # reuse bowl construction
from .robot.so100 import SO100
from .robot.so101 import SO101


# Fixed color palette used as the goal-conditioning vocabulary.
# Order is the canonical one-hot index (kept in sync with deploy.py).
COLOR_PALETTE_RGB = np.asarray(
    [
        [1.00, 0.10, 0.10],   # 0 red
        [0.10, 0.55, 1.00],   # 1 blue
        [0.10, 0.85, 0.20],   # 2 green
        [0.95, 0.85, 0.10],   # 3 yellow
    ],
    dtype=np.float32,
)
NUM_COLORS = COLOR_PALETTE_RGB.shape[0]


@dataclass
class TargetedPlaceRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)

    cube_half_size_range: Sequence[float] = (0.0125, 0.0125)
    bowl_inner_radius_range: Sequence[float] = (0.045, 0.060)
    bowl_rim_height_range: Sequence[float] = (0.018, 0.028)
    bowl_rim_thickness: float = 0.004

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)

    # Maximum XY distance between the two cubes (they must look like a "flat
    # cluster" per the spec).
    cluster_gap_max: float = 0.04
    cluster_gap_min: float = 0.025


class TargetedPlace(DefaultCameraEnv):
    """Two-cube targeted pick-and-place into a bowl."""

    SUPPORTED_ROBOTS = ["so100", "so101"]
    SUPPORTED_OBS_MODES = [
        "none", "state", "state_dict", "rgb",
        "rgb+segmentation", "rgb+state", "rgb+segmentation+state",
        "rgb+depth+segmentation", "rgb+depth+segmentation+state",
    ]
    agent: Union[SO100, SO101]
    BOWL_NUM_SEGMENTS = PlaceBowl.BOWL_NUM_SEGMENTS

    def __init__(
        self,
        *args,
        robot_uids: str = "so101",
        control_mode: str = "pd_joint_target_delta_pos",
        domain_randomization_config: Union[TargetedPlaceRandomizationConfig, dict]
            = TargetedPlaceRandomizationConfig(),
        domain_randomization: bool = True,
        spawn_box_pos: Sequence[float] = (0.3, 0.0),
        spawn_box_half_size: float = 0.10,
        target_bowl_pos: Optional[Sequence[float]] = None,
        target_color_idx: Optional[int] = None,
        color_palette: np.ndarray = COLOR_PALETTE_RGB,
        **kwargs,
    ):
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()
        else:
            raise ValueError(f"Unsupported robot_uids: {robot_uids}")

        self.domain_randomization_config = TargetedPlaceRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=TargetedPlaceRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, TargetedPlaceRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size

        self._fixed_target_bowl_pos = (
            None if target_bowl_pos is None else np.asarray(target_bowl_pos, dtype=np.float32)[:2]
        )
        self._fixed_target_color_idx = target_color_idx
        self.color_palette = np.asarray(color_palette, dtype=np.float32)
        assert self.color_palette.ndim == 2 and self.color_palette.shape[1] == 3, \
            "color_palette must be (K, 3)"
        self.num_colors = self.color_palette.shape[0]

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    # ----- Agent ----------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=(
                self.domain_randomization
                and self.domain_randomization_config.robot_color == "random"
            ),
        )

    # ----- Scene ----------------------------------------------------------------

    def _load_scene(self, options: dict):
        cfg = self.domain_randomization_config

        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        try:
            for shape in self.table_scene.table.find_component_by_type(
                sapien.render.RenderBodyComponent
            ).render_shapes:
                for part in shape.parts:
                    part.material.set_base_color(list(TABLE_COLOR_RGB) + [1.0])
        except Exception:
            pass

        # Per-env: target color index, distractor color index
        if self._fixed_target_color_idx is not None:
            target_idx = np.full(self.num_envs, int(self._fixed_target_color_idx), dtype=np.int64)
        else:
            target_idx = self._batched_episode_rng.randint(low=0, high=self.num_colors)
            target_idx = np.asarray(target_idx, dtype=np.int64)
        # Distractor: any color != target
        distractor_idx = (target_idx + 1 + self._batched_episode_rng.randint(
            low=0, high=self.num_colors - 1)) % self.num_colors
        distractor_idx = np.asarray(distractor_idx, dtype=np.int64)

        self.target_color_idx = common.to_tensor(target_idx, device=self.device).long()
        self.distractor_color_idx = common.to_tensor(distractor_idx, device=self.device).long()

        target_rgb = self.color_palette[target_idx]
        distractor_rgb = self.color_palette[distractor_idx]
        target_rgba = np.concatenate([target_rgb, np.ones((self.num_envs, 1), dtype=np.float32)], axis=-1)
        distractor_rgba = np.concatenate([distractor_rgb, np.ones((self.num_envs, 1), dtype=np.float32)], axis=-1)

        # Cube geometry / physics
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

        targets, distractors = [], []
        for i in range(self.num_envs):
            mat = sapien.pysapien.physx.PhysxMaterial(
                static_friction=float(frictions[i]),
                dynamic_friction=float(frictions[i]),
                restitution=0.0,
            )
            for color, name, store_in in (
                (target_rgba[i], f"target-{i}", targets),
                (distractor_rgba[i], f"distractor-{i}", distractors),
            ):
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[float(half_sizes[i])] * 3, material=mat,
                                          density=float(densities[i]))
                builder.add_box_visual(
                    half_size=[float(half_sizes[i])] * 3,
                    material=sapien.render.RenderMaterial(base_color=color),
                )
                builder.initial_pose = sapien.Pose(p=[0.2, 0.0, float(half_sizes[i])])
                builder.set_scene_idxs([i])
                actor = builder.build(name=name)
                store_in.append(actor)
                self.remove_from_state_dict_registry(actor)

        self.target_item = Actor.merge(targets, name="target_item")
        self.distractor_item = Actor.merge(distractors, name="distractor_item")
        self.add_to_state_dict_registry(self.target_item)
        self.add_to_state_dict_registry(self.distractor_item)

        # Bowl: copied geometry from PlaceBowl (procedural disk + rim segments)
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
            r = float(inner_radii[i]); h = float(rim_heights[i])
            outer_r = r + rim_thick
            builder = self.scene.create_actor_builder()
            floor_pose = sapien.Pose(p=[0.0, 0.0, floor_thickness / 2],
                                     q=euler2quat(0, np.pi / 2, 0))
            builder.add_cylinder_collision(radius=outer_r, half_length=floor_thickness / 2, pose=floor_pose)
            builder.add_cylinder_visual(radius=outer_r, half_length=floor_thickness / 2, pose=floor_pose,
                                        material=bowl_color)
            for s in range(self.BOWL_NUM_SEGMENTS):
                ang = 2 * np.pi * s / self.BOWL_NUM_SEGMENTS
                cx = (r + rim_thick / 2) * np.cos(ang)
                cy = (r + rim_thick / 2) * np.sin(ang)
                seg_pose = sapien.Pose(p=[cx, cy, h / 2 + floor_thickness])
                tangent_w = (2 * np.pi * (r + rim_thick / 2) / self.BOWL_NUM_SEGMENTS) / 2
                builder.add_box_collision(pose=seg_pose,
                    half_size=[rim_thick / 2, tangent_w, h / 2])
                builder.add_box_visual(pose=seg_pose,
                    half_size=[rim_thick / 2, tangent_w, h / 2], material=bowl_color)
            builder.initial_pose = sapien.Pose(p=[-0.2, 0.0, 0.0])
            builder.set_scene_idxs([i])
            bowls.append(builder.build_static(name=f"bowl-{i}"))
            self.remove_from_state_dict_registry(bowls[-1])
        self.bowl = Actor.merge(bowls, name="bowl")
        self.add_to_state_dict_registry(self.bowl)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.target_item)
            self.remove_object_from_greenscreen(self.distractor_item)
            self.remove_object_from_greenscreen(self.bowl)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )

        self._load_camera_mount()
        self._randomize_robot_color()

        # Visual goal sphere (above bowl) for debugging
        gb = self.scene.create_actor_builder()
        gb.add_sphere_visual(radius=0.008,
                             material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1]))
        gb.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = gb.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    # ----- Episode init ---------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.table_scene.table.set_pose(self.table_pose)

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
            region = [
                [-self.spawn_box_half_size, -self.spawn_box_half_size],
                [self.spawn_box_half_size, self.spawn_box_half_size],
            ]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )

            item_radius = self.item_half_sizes.max().item() + 0.01
            bowl_radius = self.bowl_inner_radii.max().item() + 0.02

            # Place target cube anywhere in the box; distractor is forced to be
            # adjacent (within cluster_gap range) to form a flat cluster.
            target_xy_offset = sampler.sample(item_radius, 100)

            cfg = self.domain_randomization_config
            ang = torch.rand(b, device=self.device) * 2 * np.pi
            gap = torch.rand(b, device=self.device) * (cfg.cluster_gap_max - cfg.cluster_gap_min) + cfg.cluster_gap_min
            distractor_xy_offset = target_xy_offset + torch.stack(
                [gap * torch.cos(ang), gap * torch.sin(ang)], dim=-1
            )

            if self._fixed_target_bowl_pos is not None:
                bowl_xy_offset = (
                    torch.tensor(self._fixed_target_bowl_pos, device=self.device, dtype=torch.float32)
                    .view(1, 2).expand(b, -1)
                    - spawn_center[env_idx, :2]
                )
            else:
                # Sample bowl far enough from cluster center
                bowl_xy_offset = sampler.sample(bowl_radius, 100, verbose=False)

            target_xyz = torch.zeros((b, 3))
            target_xyz[:, :2] = spawn_center[env_idx, :2] + target_xy_offset
            target_xyz[:, 2] = self.item_half_sizes[env_idx]
            distractor_xyz = torch.zeros((b, 3))
            distractor_xyz[:, :2] = spawn_center[env_idx, :2] + distractor_xy_offset
            distractor_xyz[:, 2] = self.item_half_sizes[env_idx]
            qs_t = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            qs_d = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.target_item.set_pose(Pose.create_from_pq(target_xyz, qs_t))
            self.distractor_item.set_pose(Pose.create_from_pq(distractor_xyz, qs_d))

            bowl_xyz = torch.zeros((b, 3))
            bowl_xyz[:, :2] = spawn_center[env_idx, :2] + bowl_xy_offset
            bowl_xyz[:, 2] = 0.0
            self.bowl.set_pose(Pose.create_from_pq(bowl_xyz))

            goal_xyz = bowl_xyz.clone()
            goal_xyz[:, 2] = self.bowl_floor_thickness + self.item_half_sizes[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # ----- Observations ---------------------------------------------------------

    def _target_color_one_hot(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            self.target_color_idx, num_classes=self.num_colors
        ).float()

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

        # Goal conditioning, ALWAYS present so the policy sees it even in obs_mode='rgb'
        bowl_xy = self.bowl.pose.p[:, :2]
        target_bowl_pos = torch.cat(
            [bowl_xy, torch.zeros((self.num_envs, 1), device=self.device)], dim=-1
        )
        obs.update(
            target_bowl_pos=target_bowl_pos,
            target_color_one_hot=self._target_color_one_hot(),
        )

        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_target_grasped=info["is_target_grasped"],
                is_distractor_grasped=info["is_distractor_grasped"],
                target_pose=self.target_item.pose.raw_pose,
                distractor_pose=self.distractor_item.pose.raw_pose,
                bowl_pose=self.bowl.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_target_pos=self.target_item.pose.p - self.agent.tcp_pos,
                tcp_to_distractor_pos=self.distractor_item.pose.p - self.agent.tcp_pos,
                target_to_bowl_pos=self.bowl.pose.p - self.target_item.pose.p,
            )
            if self.domain_randomization:
                gp = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bowl_inner_radius=self.bowl_inner_radii,
                    bowl_rim_height=self.bowl_rim_heights,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gp["gripper_stiffness"],
                    gripper_damping=gp["gripper_damping"],
                )
        return obs

    # ----- Termination / reward -------------------------------------------------

    def _in_bowl(self, item):
        offset_xy = item.pose.p[:, :2] - self.bowl.pose.p[:, :2]
        dist_xy = torch.linalg.norm(offset_xy, dim=-1)
        above = dist_xy < self.bowl_inner_radii
        below_rim = item.pose.p[:, 2] < (
            self.bowl_rim_heights + self.bowl_floor_thickness + 0.005
        )
        return above & below_rim, dist_xy

    def evaluate(self):
        target_in_bowl, dist_xy_target = self._in_bowl(self.target_item)
        distractor_in_bowl, _ = self._in_bowl(self.distractor_item)

        target_lifted = self.target_item.pose.p[..., -1] >= (self.item_half_sizes + 1e-3)
        target_vel = torch.linalg.norm(self.target_item.linear_velocity, axis=-1)
        is_target_static = target_vel <= 2e-2

        is_target_grasped = self.agent.is_grasping(self.target_item)
        is_distractor_grasped = self.agent.is_grasping(self.distractor_item)
        is_robot_static = self.agent.is_static()

        robot_touching_table = self.agent.is_touching(self.table_scene.table)
        robot_touching_bowl = self.agent.is_touching(self.bowl)
        robot_touching_target = self.agent.is_touching(self.target_item)

        success = (
            target_in_bowl
            & (~robot_touching_target)
            & is_robot_static
            & (~robot_touching_bowl)
        )
        fail = distractor_in_bowl  # wrong cube placed → terminal failure

        return {
            "dist_xy_target": dist_xy_target,
            "target_in_bowl": target_in_bowl,
            "distractor_in_bowl": distractor_in_bowl,
            "target_lifted": target_lifted,
            "is_target_static": is_target_static,
            "success": success,
            "fail": fail,
            "is_target_grasped": is_target_grasped,
            "is_distractor_grasped": is_distractor_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_table": robot_touching_table,
            "robot_touching_bowl": robot_touching_bowl,
            "robot_touching_target": robot_touching_target,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reach the TARGET (not distractor)
        tcp_to_target = torch.linalg.norm(
            self.agent.tcp_pose.p - self.target_item.pose.p, dim=1
        )
        reward = 2 * (1 - torch.tanh(5 * tcp_to_target))

        # Place reward column (above bowl center)
        target_pos = self.target_item.pose.p
        bowl_pos = self.bowl.pose.p
        goal = bowl_pos.clone()
        goal[..., 2] = self.bowl_floor_thickness + self.item_half_sizes

        item_to_goal = torch.linalg.norm(goal - target_pos, dim=1)
        place_reward = 1 - torch.tanh(5.0 * item_to_goal)

        # Grasped target: 3 + place_reward
        reward[info["is_target_grasped"]] = (3 + place_reward)[info["is_target_grasped"]]

        # Above bowl: 4 + bonuses
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, -1, :]
        gripper_openness = (
            (self.agent.robot.get_qpos()[:, -1] - gripper_min) / (gripper_max - gripper_min)
        )
        is_dropped = (~info["robot_touching_target"]).float()
        rv = torch.linalg.norm(self.agent.robot.get_qvel()[:, :-1], dim=1)
        static_r = 1 - torch.tanh(rv * 10)
        above = info["target_in_bowl"]
        reward[above] = (4 + place_reward + is_dropped + gripper_openness + static_r)[above]

        # Success
        reward[info["success"]] = 9

        # Penalties
        reward -= 6 * info["robot_touching_table"].float()
        reward -= 3 * info["robot_touching_bowl"].float()
        reward -= 1 * (~info["target_lifted"]).float()
        # Strong penalty for grasping or placing the wrong cube
        reward -= 4 * info["is_distractor_grasped"].float()
        reward -= 8 * info["distractor_in_bowl"].float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 9


@register_env("SO101TargetedPlace-v1", max_episode_steps=80)
class TargetedPlaceRandom(TargetedPlace):
    """Eval 2 training env: random target color + cluster + bowl positions."""
    pass


@register_env("SO101TargetedPlaceFixed-v1", max_episode_steps=120)
class TargetedPlaceFixed(TargetedPlace):
    """Eval 2 deterministic env: pinned target_color_idx and target_bowl_pos.

    Set the kwargs at gym.make time, e.g.:
        gym.make("SO101TargetedPlaceFixed-v1",
                 target_color_idx=0,
                 target_bowl_pos=(0.30, 0.05),
                 num_envs=1)
    """
    pass
