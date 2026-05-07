"""
Multi-Block Sequential Pick-and-Place evaluation env (Project 3, Eval 3).

Four cubes of distinct, KNOWN colors in the workspace. The policy is given a
sequence of three (target_color, target_bowl_pos) pairs that must be executed
in order. This file ships the *evaluation* env; for training, reuse the Eval 2
env (`SO101TargetedPlace-v1`) plus the sequential runner in
`sim/policies/sequential_runner.py` (Option B in the plan), or train an
end-to-end multi-step policy on this env (Option A).

The env tracks:
  - step_idx (0..2): which subgoal is currently active
  - per-step success on (color_i placed in bowl_i)
  - terminal success after the third subgoal

Per-rollout scoring (per the spec): step 0 -> 4pts, step 1 -> 4pts, step 2 -> 2pts.
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
from .place_bowl import TABLE_COLOR_RGB, PlaceBowl
from .robot.so100 import SO100
from .robot.so101 import SO101
from .targeted_pick_place import COLOR_PALETTE_RGB

# Spec: 4 distinct colors, 3 fixed bowls, 3-step sequence
NUM_BLOCKS = 4
NUM_BOWLS = 3
SEQUENCE_LEN = 3
PER_STEP_POINTS = (4, 4, 2)  # informational; reward stages match these weights


@dataclass
class MultiBlockRandomizationConfig(DefaultRandomizationConfig):
    robot_qpos_noise_std: float = np.deg2rad(5)

    cube_half_size_range: Sequence[float] = (0.0125, 0.0125)
    bowl_inner_radius_range: Sequence[float] = (0.045, 0.060)
    bowl_rim_height_range: Sequence[float] = (0.018, 0.028)
    bowl_rim_thickness: float = 0.004

    item_friction_range: Sequence[float] = (0.1, 0.5)
    item_density_range: Sequence[float] = (200, 200)


class MultiBlockSequential(DefaultCameraEnv):
    """4-cube / 3-bowl sequential pick-and-place evaluator."""

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
        domain_randomization_config: Union[MultiBlockRandomizationConfig, dict]
            = MultiBlockRandomizationConfig(),
        domain_randomization: bool = True,
        spawn_box_pos: Sequence[float] = (0.3, 0.0),
        spawn_box_half_size: float = 0.12,
        bowl_positions: Optional[Sequence[Sequence[float]]] = None,
        sequence_color_idx: Optional[Sequence[int]] = None,
        color_palette: np.ndarray = COLOR_PALETTE_RGB,
        **kwargs,
    ):
        if robot_uids == "so100":
            self.base_z_rot = np.pi / 2
            self.rest_qpos = [0, 0, 0, np.pi / 2, np.pi / 2, 0]
        elif robot_uids == "so101":
            self.base_z_rot = 0
            self.rest_qpos = SO101.keyframes["start"].qpos.tolist()

        self.domain_randomization_config = MultiBlockRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=MultiBlockRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, MultiBlockRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.spawn_box_pos = list(spawn_box_pos)
        self.spawn_box_half_size = spawn_box_half_size

        self.color_palette = np.asarray(color_palette, dtype=np.float32)
        assert self.color_palette.shape == (NUM_BLOCKS, 3), \
            f"color_palette must be ({NUM_BLOCKS}, 3) for the 4-block eval"

        # Bowls: pinned by spec ("bowl positions remain fixed within each rollout")
        if bowl_positions is None:
            # Sensible default layout: three bowls along the back of the workspace
            bowl_positions = [(0.32, -0.10), (0.32, 0.00), (0.32, 0.10)]
        bowl_positions = np.asarray(bowl_positions, dtype=np.float32)
        assert bowl_positions.shape == (NUM_BOWLS, 2)
        self.bowl_positions_world = bowl_positions

        # Sequence: which cube color goes into which bowl, in order
        if sequence_color_idx is None:
            # Default to first 3 of the palette (red, blue, green)
            sequence_color_idx = (0, 1, 2)
        self.sequence_color_idx = tuple(int(c) for c in sequence_color_idx)
        assert len(self.sequence_color_idx) == SEQUENCE_LEN

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, self.base_z_rot)),
            build_separate=(
                self.domain_randomization
                and self.domain_randomization_config.robot_color == "random"
            ),
        )

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

        # Cubes: one per palette color, per env
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
            frictions = np.ones(self.num_envs) * sum(cfg.item_friction_range) / 2
            densities = np.ones(self.num_envs) * sum(cfg.item_density_range) / 2

        self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
        self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        self.blocks = []  # list[Actor], length NUM_BLOCKS
        for c in range(NUM_BLOCKS):
            color = list(self.color_palette[c]) + [1.0]
            actors = []
            for i in range(self.num_envs):
                mat = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=float(frictions[i]),
                    dynamic_friction=float(frictions[i]),
                    restitution=0.0,
                )
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[float(half_sizes[i])] * 3, material=mat,
                                          density=float(densities[i]))
                builder.add_box_visual(
                    half_size=[float(half_sizes[i])] * 3,
                    material=sapien.render.RenderMaterial(base_color=color),
                )
                builder.initial_pose = sapien.Pose(p=[0.2 + 0.05 * c, 0.0, float(half_sizes[i])])
                builder.set_scene_idxs([i])
                actor = builder.build(name=f"block_{c}-{i}")
                actors.append(actor)
                self.remove_from_state_dict_registry(actor)
            merged = Actor.merge(actors, name=f"block_{c}")
            self.add_to_state_dict_registry(merged)
            self.blocks.append(merged)

        # Bowls (3) — static actors, pinned at TA-given positions
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

        self.bowls = []
        for b_idx in range(NUM_BOWLS):
            actors = []
            bx, by = float(self.bowl_positions_world[b_idx, 0]), float(self.bowl_positions_world[b_idx, 1])
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
                builder.initial_pose = sapien.Pose(p=[bx, by, 0.0])
                builder.set_scene_idxs([i])
                bowl_actor = builder.build_static(name=f"bowl{b_idx}-{i}")
                actors.append(bowl_actor)
                self.remove_from_state_dict_registry(bowl_actor)
            merged_bowl = Actor.merge(actors, name=f"bowl_{b_idx}")
            self.add_to_state_dict_registry(merged_bowl)
            self.bowls.append(merged_bowl)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            for blk in self.blocks:
                self.remove_object_from_greenscreen(blk)
            for bw in self.bowls:
                self.remove_object_from_greenscreen(bw)

        self.rest_qpos = common.to_tensor(self.rest_qpos, device=self.device)
        self.table_pose = Pose.create_from_pq(
            p=[-0.12 + 0.737, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)
        )
        self._load_camera_mount()
        self._randomize_robot_color()

        # Per-env step pointer (0..SEQUENCE_LEN). Stored on device.
        self.step_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Per-step done flag, used so reward only fires once per step.
        self.step_done = torch.zeros((self.num_envs, SEQUENCE_LEN), dtype=torch.bool, device=self.device)

        # Sequence: tuple -> tensor of color indices, one per env (same across envs for now)
        seq = torch.tensor(self.sequence_color_idx, dtype=torch.long, device=self.device)
        self.sequence = seq.unsqueeze(0).expand(self.num_envs, -1).contiguous()  # (N, 3)

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

            block_radius = self.item_half_sizes.max().item() * 2 + 0.005
            for c in range(NUM_BLOCKS):
                xy = sampler.sample(block_radius, 200, verbose=False)
                xyz = torch.zeros((b, 3))
                xyz[:, :2] = spawn_center[env_idx, :2] + xy
                xyz[:, 2] = self.item_half_sizes[env_idx]
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.blocks[c].set_pose(Pose.create_from_pq(xyz, qs))

            # Reset the sequence counter for these envs
            self.step_idx[env_idx] = 0
            self.step_done[env_idx] = False

    # ----- Helpers --------------------------------------------------------------

    def _block_in_bowl(self, block: Actor, bowl: Actor) -> torch.Tensor:
        offset_xy = block.pose.p[:, :2] - bowl.pose.p[:, :2]
        dist_xy = torch.linalg.norm(offset_xy, dim=-1)
        above = dist_xy < self.bowl_inner_radii
        below = block.pose.p[:, 2] < (
            self.bowl_rim_heights + self.bowl_floor_thickness + 0.005
        )
        return above & below

    def _current_target_block(self):
        # gather by current step's color index
        cur_idx = self.step_idx.clamp(max=SEQUENCE_LEN - 1)  # (N,)
        cur_color = self.sequence.gather(1, cur_idx.unsqueeze(-1)).squeeze(-1)  # (N,)
        return cur_color  # (N,) color index

    def _current_bowl_idx(self):
        return self.step_idx.clamp(max=SEQUENCE_LEN - 1)  # (N,)

    # ----- Observations ---------------------------------------------------------

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            qpos = qpos + torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        cur_color = self._current_target_block()
        cur_bowl = self._current_bowl_idx()
        cur_color_one_hot = torch.nn.functional.one_hot(cur_color, num_classes=NUM_BLOCKS).float()

        # Stack bowl positions for the *current* subgoal
        bowl_positions = torch.stack(
            [bw.pose.p[:, :2] for bw in self.bowls], dim=1
        )  # (N, NUM_BOWLS, 2)
        cur_bowl_xy = bowl_positions.gather(
            1, cur_bowl.view(-1, 1, 1).expand(-1, 1, 2)
        ).squeeze(1)  # (N, 2)
        target_bowl_pos = torch.cat(
            [cur_bowl_xy, torch.zeros((self.num_envs, 1), device=self.device)], dim=-1
        )

        obs = dict(
            target_bowl_pos=target_bowl_pos,
            target_color_one_hot=cur_color_one_hot,
            step_idx=self.step_idx.float().unsqueeze(-1),
        )

        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                tcp_pose=self.agent.tcp_pose.raw_pose,
                step_done=self.step_done.float(),
                **{f"block_{c}_pose": self.blocks[c].pose.raw_pose for c in range(NUM_BLOCKS)},
                **{f"bowl_{b}_pose": self.bowls[b].pose.raw_pose for b in range(NUM_BOWLS)},
            )
            if self.domain_randomization:
                gp = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bowl_inner_radius=self.bowl_inner_radii,
                    bowl_rim_height=self.bowl_rim_heights,
                    gripper_stiffness=gp["gripper_stiffness"],
                    gripper_damping=gp["gripper_damping"],
                )
        return obs

    # ----- Termination / reward -------------------------------------------------

    def evaluate(self):
        """Per-env scoring + step advancement.

        We update step_idx in-place when the current subgoal newly completes,
        so the next env step sees the next subgoal's goal-conditioning.
        """
        # For each subgoal, check if its (color, bowl) pair is now satisfied.
        per_step_now = torch.zeros((self.num_envs, SEQUENCE_LEN), dtype=torch.bool, device=self.device)
        for s in range(SEQUENCE_LEN):
            color_for_step = self.sequence[:, s]  # (N,)
            bowl_for_step = s
            # gather block pose by color index
            block_xy = torch.stack([self.blocks[c].pose.p[:, :2] for c in range(NUM_BLOCKS)], dim=1)
            block_z = torch.stack([self.blocks[c].pose.p[:, 2] for c in range(NUM_BLOCKS)], dim=1)
            sel_xy = block_xy.gather(1, color_for_step.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
            sel_z = block_z.gather(1, color_for_step.unsqueeze(-1)).squeeze(-1)
            offset = sel_xy - self.bowls[bowl_for_step].pose.p[:, :2]
            dist = torch.linalg.norm(offset, dim=-1)
            inside = (dist < self.bowl_inner_radii) & (
                sel_z < (self.bowl_rim_heights + self.bowl_floor_thickness + 0.005)
            )
            per_step_now[:, s] = inside

        is_robot_static = self.agent.is_static()

        # Newly-achieved (was not done before, is now, AND is the active step)
        newly_done = per_step_now & (~self.step_done)
        # Only count newly-done at the active step
        active_mask = torch.zeros_like(newly_done)
        for s in range(SEQUENCE_LEN):
            active_mask[:, s] = (self.step_idx == s) & is_robot_static
        step_now_completed = newly_done & active_mask  # (N, 3)

        # Persist
        self.step_done = self.step_done | step_now_completed
        # Advance step counter for every env that just completed its active step
        advanced = step_now_completed.any(dim=-1)
        self.step_idx[advanced] = (self.step_idx[advanced] + 1).clamp(max=SEQUENCE_LEN)

        all_done = self.step_done.all(dim=-1)

        # Penalty: any non-target block sitting in any bowl is bad
        wrong_in_bowl = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for s in range(SEQUENCE_LEN):
            for c in range(NUM_BLOCKS):
                if c == self.sequence_color_idx[s]:
                    continue
                # Block c in bowl s
                bxy = self.blocks[c].pose.p[:, :2]
                bz = self.blocks[c].pose.p[:, 2]
                offset = bxy - self.bowls[s].pose.p[:, :2]
                dist = torch.linalg.norm(offset, dim=-1)
                inside = (dist < self.bowl_inner_radii) & (
                    bz < (self.bowl_rim_heights + self.bowl_floor_thickness + 0.005)
                )
                wrong_in_bowl = wrong_in_bowl | inside

        return {
            "success": all_done,
            "step_idx": self.step_idx,
            "step_done": self.step_done,
            "step_completed_this_call": step_now_completed,
            "wrong_in_bowl": wrong_in_bowl,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Reward shaping for the *active* subgoal: reach + place toward
        # active block's bowl. Weighted by per-step points (4,4,2).
        N = self.num_envs
        cur_color = self._current_target_block()  # (N,)
        cur_bowl_idx = self._current_bowl_idx()    # (N,)

        # Active block pose
        block_p = torch.stack([self.blocks[c].pose.p for c in range(NUM_BLOCKS)], dim=1)  # (N,K,3)
        active_block = block_p.gather(1, cur_color.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (N,3)

        # Active bowl pose
        bowl_p = torch.stack([bw.pose.p for bw in self.bowls], dim=1)  # (N,B,3)
        active_bowl = bowl_p.gather(1, cur_bowl_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)  # (N,3)

        tcp = self.agent.tcp_pose.p
        tcp_to_block = torch.linalg.norm(tcp - active_block, dim=-1)
        reach = 2 * (1 - torch.tanh(5 * tcp_to_block))

        goal = active_bowl.clone()
        goal[:, 2] = self.bowl_floor_thickness + self.item_half_sizes
        block_to_goal = torch.linalg.norm(active_block - goal, dim=-1)
        place = 1 - torch.tanh(5.0 * block_to_goal)

        reward = reach + place

        # Step completion bonuses (use spec weights /2 to keep magnitudes sane)
        step_bonus = torch.zeros(N, device=self.device)
        for s in range(SEQUENCE_LEN):
            step_bonus = step_bonus + info["step_completed_this_call"][:, s].float() * (PER_STEP_POINTS[s])

        reward = reward + step_bonus
        reward[info["success"]] = 12.0  # 4+4+2 + a small bonus implicitly

        # Penalty: wrong block in any bowl
        reward -= 4 * info["wrong_in_bowl"].float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12.0


@register_env("SO101MultiBlockSeq-v1", max_episode_steps=300)
class MultiBlockSeqEval(MultiBlockSequential):
    """Eval 3 environment: 4 blocks, 3-step sequence, fixed bowls."""
    pass
