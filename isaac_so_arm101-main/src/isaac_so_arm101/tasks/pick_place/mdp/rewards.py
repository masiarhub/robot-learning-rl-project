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
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    start_height: float = 0.12,
    saturation_height: float = 0.2,
    min_reward: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous reward for lifting the object.

    Below start_height: 0.0.
    At start_height: min_reward (jump — avoids rewarding pushes below threshold).
    Between start_height and saturation_height: linearly interpolated from min_reward to 1.0.
    Above saturation_height: 1.0.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    ramp = torch.clamp((height - start_height) / (saturation_height - start_height), 0.0, 1.0)
    above = (height > start_height).float()
    return above * (min_reward + (1.0 - min_reward) * ramp)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for grasping the object (close to object and gripper closed)."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Distance between end-effector and object
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # Check if gripper is closed (this would need to be passed as a parameter)
    # For now, reward based on proximity
    grasp_distance = 0.03  # 3cm threshold
    return torch.where(object_ee_distance < grasp_distance, 1.0, 0.0)

def object_released_in_zone(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward releasing the object when it's above the bowl."""
    target: RigidObject = env.scene[target_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    distance = torch.norm(target.data.root_pos_w[:, :3] - object.data.root_pos_w[:, :3], dim=1)
    near_bowl = distance < threshold

    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_obj_dist = torch.norm(object.data.root_pos_w - ee_w, dim=1)
    gripper_open = ee_obj_dist > 0.05

    return (near_bowl & gripper_open).float()

def object_bowl_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for moving the lifted object close to the bowl — continuous tanh gradient."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    goal_pos_w = bowl.data.root_pos_w[:, :3].clone()
    goal_pos_w[:, 2] += 0.05  # target 5cm above bowl

    distance = torch.norm(goal_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    return (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def gripper_aperture_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    saturation_pos: float = 0.15,
    close_joint_pos: float = -0.1,
    contact_force_saturation: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
) -> torch.Tensor:
    """Reward for keeping the gripper open while approaching the cube.

    reward = aperture_frac × proximity × no_contact

    aperture_frac: 0 at close_joint_pos, 1 when gripper ≥ saturation_pos.
                   Saturates early so the gripper only needs to be "open enough".
    proximity:     tanh-kernel nearness (std ~5 cm) — ~0 when far, prevents rewarding
                   opening the gripper away from the cube.
    no_contact:    soft gate via tanh; decays smoothly from 1 → 0 as lateral contact
                   force grows, avoiding the reward cliff that a hard binary gate would
                   create at the aperture→grasp transition.

    Args:
        std:                    EE-cube distance at which proximity = 1 - tanh(1) ≈ 0.24 (m).
        saturation_pos:         Gripper joint angle at which aperture_frac saturates at 1.0 (rad).
        close_joint_pos:        Gripper joint angle considered fully closed (rad).
        contact_force_saturation: Lateral force (N) at which no_contact ≈ 0 (tanh saturation).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_pos_w, dim=-1)
    proximity = 1.0 - torch.tanh(dist / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    aperture_frac = torch.clamp(
        (gripper_pos - close_joint_pos) / (saturation_pos - close_joint_pos),
        0.0, 1.0,
    )

    # Soft contact gate: fades aperture reward out as lateral (XY) finger force grows.
    # Uses only horizontal components so a top-down press (force mainly in Z) is ignored.
    # tanh decay avoids the hard cliff a binary gate would create at the aperture→grasp
    # transition, preventing the policy from oscillating at the boundary.
    force_matrix = cube_sensor.data.force_matrix_w_history   # [N, H, 1, 2, 3]
    finger_forces = force_matrix[:, :, 0, :, :]              # [N, H, 2, 3]
    xy_forces = finger_forces[..., :2]                        # [N, H, 2, 2]
    max_lateral = xy_forces.norm(dim=-1).max(dim=1)[0].max(dim=-1)[0]  # [N]
    no_contact = 1.0 - torch.tanh(max_lateral / contact_force_saturation)

    return aperture_frac * proximity * no_contact

def object_grasped_contact_continuous(
    env: ManagerBasedRLEnv,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 4.0,
    debug_print_interval: int = 0,
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
) -> torch.Tensor:
    """Continuous grasp quality reward based on bilateral cube-contact force.

    reward = tanh(min(left_force, right_force) / force_saturation) × balance_factor

    Uses force_matrix_w_history from a ContactSensor placed on the cube (plain
    RigidObject), filtered to the two gripper finger links. Shape: [N, H, 1, 2, 3].
    This gives true cube-specific contact forces per finger with no table contamination.
    (Robot-side force_matrix_w_history is broken for articulation bodies in Isaac Lab.)
    """
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]
    # [N, H, 1_cube_body, 2_fingers, 3] → pick the single cube body, keep finger dim
    force_matrix = cube_sensor.data.force_matrix_w_history
    finger_forces = force_matrix[:, :, 0, :, :]          # [N, H, 2, 3]
    finger_force_norms = finger_forces.norm(dim=-1).max(dim=1)[0]  # [N, 2]

    left_force  = finger_force_norms[:, 0]   # force on cube from gripper_link
    right_force = finger_force_norms[:, 1]   # force on cube from moving_jaw

    if debug_print_interval > 0 and env.common_step_counter % debug_print_interval == 0:
        print(
            f"[GRASP FORCE] step={env.common_step_counter}"
            f"  left={left_force[0].item():.2f}N"
            f"  right={right_force[0].item():.2f}N"
        )

    min_force = torch.minimum(left_force, right_force)
    force_reward = torch.tanh(min_force / force_saturation)

    ratio = torch.maximum(left_force, right_force) / (
        torch.minimum(left_force, right_force) + 1e-6
    )
    balance_factor = torch.clamp(1.0 - (ratio - 1.0) / (force_balance_ratio - 1.0), 0.0, 1.0)

    # Opposition factor: a real grip has the two jaw forces pointing toward each other
    # (anti-parallel → dot < 0). A push has both forces in roughly the same direction
    # (dot > 0). clamp(-cos_sim, 0, 1) gives 1 for a perfect clamp, 0 for a pure push.
    best_t = finger_forces.norm(dim=-1).sum(dim=-1).argmax(dim=1)  # [N]
    idx = best_t[:, None, None, None].expand(-1, 1, 2, 3)
    peak = finger_forces.gather(dim=1, index=idx).squeeze(1)  # [N, 2, 3]
    f_left  = peak[:, 0, :]  # [N, 3] force on cube from fixed jaw
    f_right = peak[:, 1, :]  # [N, 3] force on cube from moving jaw
    cos_sim = (f_left * f_right).sum(dim=-1) / (
        f_left.norm(dim=-1) * f_right.norm(dim=-1) + 1e-6
    )
    opposition = (-cos_sim).clamp(0.0, 1.0)

    return force_reward * balance_factor * opposition

def lifting_object_grasped(
    env: ManagerBasedRLEnv,
    start_height: float = 0.012,
    saturation_height: float = 0.02,
    min_reward: float = 0.0,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
) -> torch.Tensor:
    """Lift reward gated by grasp quality.

    reward = object_is_lifted(...) × object_grasped_contact_continuous(...)

    Both factors are in [0, 1], so the agent only receives lift reward proportional
    to how well it is actually grasping the cube. Pushing/bumping the cube up without
    a bilateral grip yields near-zero reward — the opposition and balance factors in
    the grasp term collapse to zero for push-style contact.
    """
    lift = object_is_lifted(env, start_height, saturation_height, min_reward, object_cfg)
    grasp = object_grasped_contact_continuous(env, force_saturation, force_balance_ratio, 0, cube_sensor_cfg)
    return lift * grasp

def robot_table_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_table"),
) -> torch.Tensor:
    """Penalty when any upper-arm link touches the table.

    Uses force_matrix_w_history from the table-side sensor (sensor on table, filter=arm links).
    Shape: [N, H, 1_table, n_arm_links, 3].
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = sensor.data.force_matrix_w_history  # [N, H, 1, n_links, 3]
    max_force = force_matrix[:, :, 0, :, :].norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()

def robot_table_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_table"),
) -> torch.Tensor:
    """Penalty when any upper-arm link touches the table.

    Uses force_matrix_w_history from the table-side sensor (sensor on table, filter=arm links).
    Shape: [N, H, 1_table, n_arm_links, 3].
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = sensor.data.force_matrix_w_history  # [N, H, 1, n_links, 3]
    max_force = force_matrix[:, :, 0, :, :].norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()

def time_penalty_if_not_lifted(
    env: ManagerBasedRLEnv,
    start_height: float = 0.05,
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty that grows linearly with episode time if cube is not lifted and not grasped.
    
    Returns a value in [-1, 0]:
      - 0   when cube is lifted OR grasped
      - scales toward -1 as episode progresses with no grasp/lift
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]

    # Time fraction [0, 1]
    time_frac = (env.episode_length_buf / env.max_episode_length).clamp(0.0, 1.0)

    # Is the cube lifted?
    lifted = (obj.data.root_pos_w[:, 2] > start_height).float()

    # Is the cube grasped? (any contact force on cube from fingers)
    force_matrix = cube_sensor.data.force_matrix_w_history  # [N, H, 1, 2, 3]
    finger_forces = force_matrix[:, :, 0, :, :].norm(dim=-1).max(dim=1)[0]  # [N, 2]
    grasped = (finger_forces.sum(dim=-1) > 0.5).float()  # any meaningful contact

    not_progressing = (1.0 - torch.clamp(lifted + grasped, 0.0, 1.0))

    return (time_frac * not_progressing)