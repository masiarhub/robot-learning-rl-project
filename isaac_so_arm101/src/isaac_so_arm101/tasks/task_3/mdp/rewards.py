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

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul, subtract_frame_transforms
from isaaclab.sensors import ContactSensor

from .._wrist_cam import (
    OFFSET_POS as _CAM_OFFSET_POS,
    OFFSET_QUAT_WXYZ as _CAM_OFFSET_QUAT_WXYZ,
    TAN_HALF_HFOV as _TAN_HALF_HFOV,
    TAN_HALF_VFOV as _TAN_HALF_VFOV,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def project_to_wrist_image(
    env: ManagerBasedRLEnv,
    points_w: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-frame points into wrist camera normalised image coordinates (NDC).

    Uses analytic FK + camera offset — works even when the wrist_camera sensor is
    not present in the scene (e.g. during Phase 1a teacher training).  The camera
    projection follows the OpenGL convention: the camera looks along its local -Z axis.

    Args:
        env: The RL environment instance.
        points_w: Points to project in world frame, shape (num_envs, 3).
        robot_cfg: Config for the robot articulation.

    Returns:
        u:       Horizontal NDC, shape (num_envs,).  u ∈ [-1, 1] means left→right.
        v:       Vertical   NDC, shape (num_envs,).  v ∈ [-1, 1] means bottom→top.
        in_view: Boolean (num_envs,) — True when the point is inside the image frustum.

    Calibration note:
        Compare the printed u/v values with the visual position of the cube in
        step_000_wrist.png from debug/debug_wrist_cam.py.  If they disagree,
        update _CAM_OFFSET_QUAT_WXYZ above.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.find_bodies(["gripper_link"])[0][0]

    gripper_pos_w  = robot.data.body_pos_w[:, body_idx, :]   # (B, 3)
    gripper_quat_w = robot.data.body_quat_w[:, body_idx, :]  # (B, 4) wxyz

    # Camera position in world frame: cam_pos_w = gripper_pos_w + R_gripper * offset_pos
    offset_pos = torch.tensor(_CAM_OFFSET_POS, device=env.device, dtype=gripper_pos_w.dtype)
    offset_pos_b = offset_pos.unsqueeze(0).expand(env.num_envs, -1)  # (B, 3)
    cam_pos_w = gripper_pos_w + quat_apply(gripper_quat_w, offset_pos_b)  # (B, 3)

    # Camera world orientation: cam_quat_w = gripper_quat_w ⊗ cam_local_quat
    cam_local_q = torch.tensor(
        _CAM_OFFSET_QUAT_WXYZ, device=env.device, dtype=gripper_quat_w.dtype
    ).unsqueeze(0).expand(env.num_envs, -1)                                           # (B, 4)
    cam_quat_w = quat_mul(gripper_quat_w, cam_local_q)                                # (B, 4)

    # Transform points from world frame → camera local frame.
    # subtract_frame_transforms(t_w_b, q_w_b, t_w_p) → t_b_p
    pts_cam, _ = subtract_frame_transforms(cam_pos_w, cam_quat_w, points_w)           # (B, 3)

    # Project onto image plane (OpenGL: camera looks along -Z → points in front have z < 0).
    z = pts_cam[:, 2]
    in_front = z < -1e-3
    safe_neg_z = (-z).clamp(min=1e-3)          # avoid division by zero

    u = pts_cam[:, 0] / (safe_neg_z * _TAN_HALF_HFOV)   # (B,) right = positive
    v = pts_cam[:, 1] / (safe_neg_z * _TAN_HALF_VFOV)   # (B,) up   = positive

    in_view = in_front & (u.abs() <= 1.0) & (v.abs() <= 1.0)

    return u, v, in_view


def cube_initial_visibility_reward(
    env: ManagerBasedRLEnv,
    max_steps: int = 20,
    std_offset: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward keeping the cube near the centre of the wrist camera during the first
    `max_steps` of each episode.

    Returns:
      +1.0  when cube is at the image centre.
       0.0  when cube is at NDC radius `std_offset` from centre (default: halfway to edge).
      -1.0  when cube is off-screen or behind the camera.
       0.0  after step `max_steps` (reward deactivates).

    This reward is purely geometric — it works for both camera environments (Phases 2/3,
    CamPPO) and the teacher (Phase 1a), where no camera sensor is rendered.  During teacher
    training it shapes the arm to orient the wrist camera toward the cube before grasping,
    so the distillation student inherits "look first" behaviour through BC.

    Args:
        max_steps:  Episode steps for which the reward is active.
        std_offset: NDC radius at which the score decays to zero.
        robot_cfg:  Config for the robot articulation.
        object_cfg: Config for the cube rigid object.
    """
    # episode_length_buf is incremented by ManagerBasedRLEnv before reward computation.
    t = env.episode_length_buf                   # (B,) 1-based step index
    active_mask = (t <= max_steps).float()       # (B,)

    obj: RigidObject = env.scene[object_cfg.name]
    cube_pos_w = obj.data.root_pos_w[:, :3]      # (B, 3)

    u, v, in_view = project_to_wrist_image(env, cube_pos_w, robot_cfg=robot_cfg)

    offset = (u.pow(2) + v.pow(2)).sqrt()        # NDC distance from image centre (B,)

    vis_score = torch.where(
        in_view,
        torch.clamp(1.0 - offset / std_offset, min=-1.0, max=1.0),
        torch.full_like(offset, -1.0),
    )

    return vis_score * active_mask


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
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # World-frame position of the cube: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # World-frame position of the end-effector (first target frame, index 0): (num_envs, 3)
    # The EE frame is offset from gripper_link by [0.01, 0, -0.09] to approximate the fingertip centre.
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Euclidean distance between EE and object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    # tanh kernel: maps distance → reward in (0, 1].
    # At distance=0  → reward=1.0 (EE is on the object).
    # At distance=std → reward ≈ 0.76  (one std away, reward is already high).
    # At distance=3*std → reward ≈ 0.10 (almost no reward beyond 3× std).
    # std=0.05 m (5 cm) keeps the gradient tight so the agent is rewarded
    # for being very close, not just in the neighbourhood.
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # The command is expressed in the robot base frame (body frame).
    # For pick-and-place it is hardcoded to (0.2, -0.25, 0.12) — 12 cm above the bowl.
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]  # desired position in robot body frame: (num_envs, 3)

    # Transform desired position from robot body frame → world frame.
    # combine_frame_transforms applies the robot's root translation + rotation.
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # Distance between the object and the goal position in the world frame: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # Gate the reward by minimal_height: the agent only receives goal-tracking reward
    # once the object is off the table. Without this gate, the agent could maximise
    # goal-tracking by sliding the object across the table surface toward the goal XY,
    # without ever picking it up.
    # Two instances of this function are used in RewardsCfg with different stds:
    #   - coarse  (std=0.3, height=0.05, weight=16): broad gradient, guides transport.
    #   - fine    (std=0.05, height=0.08, weight=5):  tight gradient, rewards precision
    #             once the object is well above the table and close to the goal.
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_bowl_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    height_offset: float = 0.0,
    debug_vis: bool = False,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for moving the object close to the bowl, gated by a minimum lift height.

    The target position is bowl_pos + [0, 0, height_offset] in world frame, so the agent
    is rewarded for bringing the cube to a point above the bowl rather than into it.
    Reads bowl position directly from the scene — no command manager needed.
    When debug_vis=True, a green sphere is drawn at each env's goal position every step.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    goal_pos_w = bowl.data.root_pos_w[:, :3].clone()
    goal_pos_w[:, 2] += height_offset

    if debug_vis:
        if not hasattr(env, "_bowl_goal_marker"):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/BowlGoalMarker",
                markers={
                    "goal": sim_utils.SphereCfg(
                        radius=0.025,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0), opacity=0.5
                        ),
                    )
                },
            )
            env._bowl_goal_marker = VisualizationMarkers(marker_cfg)
        env._bowl_goal_marker.visualize(goal_pos_w)

    distance = torch.norm(goal_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    return (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def cube_in_bowl(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    z_min: float = -0.00,
    consecutive_steps: int = 3,
    ee_min_height_above_bowl: float = 0.07,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary success reward: cube is inside bowl, gripper is open, and EE has retreated upward."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    cube_pos = obj.data.root_pos_w[:, :3]
    bowl_pos = bowl.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    # C1 — XY proximity
    c1 = torch.norm(cube_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold

    # C2 — Cube height inside bowl
    c2 = (cube_pos[:, 2] > bowl_pos[:, 2] + z_min) & (cube_pos[:, 2] < bowl_pos[:, 2] + z_max)


    # C3 — EE has moved upward away from bowl
    c3 = ee_pos[:, 2] > (bowl_pos[:, 2] + ee_min_height_above_bowl)

    satisfied = c1 & c2 & c3

  # Lazy-init counter
    if not hasattr(env, "_cube_in_bowl_steps_reward"):
        env._cube_in_bowl_steps_reward = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)

    # Previous count
    prev = env._cube_in_bowl_steps_reward

    # Update count
    env._cube_in_bowl_steps_reward = torch.where(
        satisfied,
        env._cube_in_bowl_steps_reward + 1,
        torch.zeros_like(env._cube_in_bowl_steps_reward),
    )

    # Trigger reward only on the first step where we hit consecutive_steps
    just_succeeded = (prev == consecutive_steps - 1) & (env._cube_in_bowl_steps_reward == consecutive_steps)

    # Return float tensor 0/1
    return just_succeeded.float()


def gripper_aperture_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    saturation_pos: float = 0.8,
    close_joint_pos: float = 0.2,
    contact_force_saturation: float = 0.25,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
) -> torch.Tensor:
    """Reward for keeping the gripper open while approaching the cube.

    reward = aperture_frac × proximity × no_contact

    aperture_frac: 0 at close_joint_pos, 1 when gripper ≥ saturation_pos.
                   Saturates early so the gripper only needs to be "open enough".
    proximity:     tanh-kernel nearness (std ~5 cm) — ~0 when far, prevents rewarding
                   opening the gripper away from the cube.
    no_contact:    soft gate via tanh; decays smoothly from 1 → 0 as full 3D contact
                   force grows, avoiding the reward cliff that a hard binary gate would
                   create at the aperture→grasp transition.

    Args:
        std:                    EE-cube distance at which proximity = 1 - tanh(1) ≈ 0.24 (m).
        saturation_pos:         Gripper joint angle at which aperture_frac saturates at 1.0 (rad).
        close_joint_pos:        Gripper joint angle considered fully closed (rad).
        contact_force_saturation: 3D force (N) at which no_contact ≈ 0 (tanh saturation).
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

    force_matrix = cube_sensor.data.force_matrix_w_history   # [N, H, 1, 2, 3]
    finger_forces = force_matrix[:, :, 0, :, :]              # [N, H, 2, 3]
    max_force = finger_forces.norm(dim=-1).max(dim=1)[0].max(dim=-1)[0]  # [N] full 3D
    no_contact = 1.0 - torch.tanh(max_force / contact_force_saturation)

    return aperture_frac * proximity * no_contact


def object_grasped_contact(
    env: ManagerBasedRLEnv,
    min_force_per_finger: float = 0.3,
    force_balance_ratio: float = 4.0,
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
) -> torch.Tensor:
    """Binary reward: both gripper fingers making balanced contact with the cube.

    Uses force_matrix_w_history from the cube-side ContactSensor. See
    object_grasped_contact_continuous for details on why this sensor is used.
    """
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]
    force_matrix = cube_sensor.data.force_matrix_w_history  # [N, H, 1, 2, 3]
    finger_forces = force_matrix[:, :, 0, :, :]             # [N, H, 2, 3]
    finger_force_norms = finger_forces.norm(dim=-1).max(dim=1)[0]  # [N, 2]

    left_force  = finger_force_norms[:, 0]
    right_force = finger_force_norms[:, 1]

    both_touching = (left_force > min_force_per_finger) & (right_force > min_force_per_finger)
    ratio = torch.maximum(left_force, right_force) / (
        torch.minimum(left_force, right_force) + 1e-6
    )
    balanced = ratio < force_balance_ratio

    return (both_touching & balanced).float()


def object_grasped_contact_continuous(
    env: ManagerBasedRLEnv,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 4.0,
    debug_print_interval: int = 0,
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
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

    if debug_print_interval > 0 and env.common_step_counter % debug_print_interval == 0:
        i = 0
        print(
            f"[GRASP] step={env.common_step_counter}"
            f"  left={left_force[i].item():.5f}N  right={right_force[i].item():.5f}N"
            f"  force_rew={force_reward[i].item():.5f}"
            f"  balance={balance_factor[i].item():.5f}"
            f"  opposition={opposition[i].item():.5f}"
            f"  → grasp={(force_reward * balance_factor * opposition)[i].item():.6f}"
        )

    return force_reward * balance_factor * opposition


def robot_body_cube_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red_body"),
) -> torch.Tensor:
    """Penalty when any non-gripper arm link touches the cube.

    Uses force_matrix_w_history from the cube-side sensor (sensor on cube, filter=arm links).
    Shape: [N, H, 1_cube, n_arm_links, 3]. True cube-specific contact — no table contamination.
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


def robot_bowl_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_bowl"),
) -> torch.Tensor:
    """Penalty when any robot link touches the bowl.

    Uses force_matrix_w_history from the bowl-side sensor (sensor on Bowl/.*, filter=robot links).
    Shape: [N, H, n_bowl_prims, n_robot_links, 3] — max over all dims to get per-env signal.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix = sensor.data.force_matrix_w_history  # [N, H, n_bowl_prims, n_links, 3]
    max_force = force_matrix.norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()


def debug_grasp_state(
    env: ManagerBasedRLEnv,
    print_interval: int = 50,
    lift_start_height: float = 0.015,
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    robot_gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Debug-only reward (weight≈0) that prints grasp diagnostics for env 0.

    Prints every `print_interval` steps:
      - fixed_jaw / moving_jaw: true cube-contact force per finger from cube-side sensor [N]
      - cube_z: cube height in world frame [m]
      - gripper: gripper joint angle [rad] (open≈0.2, closed≈−0.1)
      - ee_dist: EE-to-cube distance [m]
    """
    if print_interval <= 0 or env.common_step_counter % print_interval != 0:
        return torch.zeros(env.num_envs, device=env.device)

    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]
    force_matrix = cube_sensor.data.force_matrix_w_history  # [N, H, 1, 2, 3]
    fixed_jaw_f  = force_matrix[0, :, 0, 0, :].norm(dim=-1).max().item()
    moving_jaw_f = force_matrix[0, :, 0, 1, :].norm(dim=-1).max().item()

    obj: RigidObject = env.scene[object_cfg.name]
    cube_z = obj.data.root_pos_w[0, 2].item()

    robot: Articulation = env.scene[robot_gripper_cfg.name]
    gripper_j = robot.data.joint_pos[0, robot_gripper_cfg.joint_ids[0]].item()

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_dist = torch.norm(ee_frame.data.target_pos_w[0, 0, :] - obj.data.root_pos_w[0, :3]).item()

    print(
        f"[GRASP DBG step={env.common_step_counter:6d}] "
        f"fixed_jaw={fixed_jaw_f:6.2f}N  moving_jaw={moving_jaw_f:6.2f}N  "
        f"cube_z={cube_z:.4f}m  gripper={gripper_j:+.3f}rad  "
        f"ee_dist={ee_dist:.4f}m  lifted={'YES' if cube_z > lift_start_height else 'no'}"
    )

    return torch.zeros(env.num_envs, device=env.device)


def lifting_object_grasped(
    env: ManagerBasedRLEnv,
    start_height: float = 0.012,
    saturation_height: float = 0.02,
    min_reward: float = 0.0,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
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


def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Combined reward for reaching the object AND lifting it.

    NOTE: not currently used in RewardsCfg. Could replace the separate
    reaching_object + lifting_object terms to avoid rewarding the agent
    for hovering near the object without grasping it.
    """
    # Smooth approach signal: high when EE is close to object (tanh kernel, see object_ee_distance).
    reach_reward = object_ee_distance(env, std, object_cfg, ee_frame_cfg)
    # Binary lift gate: 1 if object is above minimal_height, else 0 (see object_is_lifted).
    lift_reward = object_is_lifted(env, minimal_height, object_cfg)
    # Multiplicative combination: the agent only gets reaching credit after lifting.
    # This prevents a degenerate strategy where the agent earns reaching reward
    # by resting the EE on top of the stationary cube indefinitely.
    return reach_reward * lift_reward


##
# Task 3 — target-aware four-cube reward functions
##


def _get_target_color_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """[N] int64: per-env target cube ID (0=red, 1=blue, 2=green, 3=yellow)."""
    if not hasattr(env, "_target_color_id"):
        env._target_color_id = torch.randint(0, 4, (env.num_envs,), dtype=torch.int64, device=env.device)
    return env._target_color_id


# Keep _target_is_red as an alias used by terminations.py until fully replaced.
def _target_is_red(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _get_target_color_id(env) == 0


def reaching_target_cube_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """tanh-kernel reward for reaching the *target* cube."""
    tid = _get_target_color_id(env)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    N = env.num_envs
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos = all_pos[torch.arange(N, device=tid.device), tid]  # [N, 3]
    return 1.0 - torch.tanh(torch.norm(target_pos - ee_w, dim=-1) / std)


def target_gripper_aperture_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    saturation_pos: float = 0.8,
    close_joint_pos: float = 0.2,
    contact_force_saturation: float = 0.25,
    debug_print_interval: int = 0,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    red_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    blue_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_blue"),
    green_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_green"),
    yellow_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_yellow"),
) -> torch.Tensor:
    """Gripper-aperture reward near the *target* cube; soft contact gate via target cube sensor."""
    tid = _get_target_color_id(env)
    N = env.num_envs
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos = all_pos[torch.arange(N, device=tid.device), tid]
    target_dist = torch.norm(target_pos - ee_pos_w, dim=-1)
    proximity = 1.0 - torch.tanh(target_dist / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    aperture_frac = torch.clamp(
        (gripper_pos - close_joint_pos) / (saturation_pos - close_joint_pos), 0.0, 1.0
    )

    def _contact_force(sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        fm = sensor.data.force_matrix_w_history   # [N, H, 1, 2, 3]
        ff = fm[:, :, 0, :, :]                    # [N, H, 2, 3]
        return ff.norm(dim=-1).max(dim=1)[0].max(dim=-1)[0]  # [N] full 3D force

    all_force = torch.stack(
        [_contact_force(red_sensor_cfg), _contact_force(blue_sensor_cfg),
         _contact_force(green_sensor_cfg), _contact_force(yellow_sensor_cfg)],
        dim=1,
    )  # [N, 4]
    target_force = all_force[torch.arange(N, device=tid.device), tid]
    no_contact = 1.0 - torch.tanh(target_force / contact_force_saturation)

    reward = aperture_frac * proximity * no_contact

    if debug_print_interval > 0 and env.common_step_counter % debug_print_interval == 0:
        i = 0
        print(
            f"[APERTURE] step={env.common_step_counter}"
            f"  gripper_joint={gripper_pos[i].item():.3f}rad"
            f"  aperture_frac={aperture_frac[i].item():.3f}"
            f"  dist_to_cube={target_dist[i].item():.3f}m"
            f"  proximity={proximity[i].item():.3f}"
            f"  contact_force={target_force[i].item():.5f}N"
            f"  no_contact={no_contact[i].item():.5f}"
            f"  → aperture_rew={reward[i].item():.6f}"
        )

    return reward


def target_cube_grasped_reward(
    env: ManagerBasedRLEnv,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 4.0,
    debug_print_interval: int = 0,
    red_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    blue_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_blue"),
    green_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_green"),
    yellow_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_yellow"),
) -> torch.Tensor:
    """Continuous grasp-quality reward on the *target* cube only."""
    tid = _get_target_color_id(env)
    N = env.num_envs

    def _grasp(sensor_cfg, dbg):
        return object_grasped_contact_continuous(
            env, force_saturation=force_saturation,
            force_balance_ratio=force_balance_ratio,
            debug_print_interval=dbg, cube_sensor_cfg=sensor_cfg,
        )

    all_grasp = torch.stack(
        [_grasp(red_sensor_cfg, debug_print_interval), _grasp(blue_sensor_cfg, 0),
         _grasp(green_sensor_cfg, 0), _grasp(yellow_sensor_cfg, 0)],
        dim=1,
    )  # [N, 4]
    return all_grasp[torch.arange(N, device=tid.device), tid]


def target_cube_lifted_reward(
    env: ManagerBasedRLEnv,
    start_height: float = 0.012,
    saturation_height: float = 0.02,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 3.0,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    red_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    blue_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_blue"),
    green_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_green"),
    yellow_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_yellow"),
) -> torch.Tensor:
    """Lift reward gated by grasp quality on the *target* cube only."""
    tid = _get_target_color_id(env)
    N = env.num_envs

    def _lift(obj_cfg, sensor_cfg):
        return lifting_object_grasped(
            env, start_height=start_height, saturation_height=saturation_height,
            min_reward=0.0, force_saturation=force_saturation,
            force_balance_ratio=force_balance_ratio,
            object_cfg=obj_cfg, cube_sensor_cfg=sensor_cfg,
        )

    all_lift = torch.stack(
        [_lift(red_object_cfg, red_sensor_cfg), _lift(blue_object_cfg, blue_sensor_cfg),
         _lift(green_object_cfg, green_sensor_cfg), _lift(yellow_object_cfg, yellow_sensor_cfg)],
        dim=1,
    )  # [N, 4]
    return all_lift[torch.arange(N, device=tid.device), tid]


def target_cube_to_bowl_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    minimal_height: float = 0.05,
    height_offset: float = 0.0,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Transport reward: *target* cube distance to bowl, gated by minimum lift height."""
    tid = _get_target_color_id(env)
    N = env.num_envs

    def _dist(obj_cfg):
        return object_bowl_distance(env, std, minimal_height, height_offset,
                                    bowl_cfg=bowl_cfg, object_cfg=obj_cfg)

    all_dist = torch.stack(
        [_dist(red_object_cfg), _dist(blue_object_cfg),
         _dist(green_object_cfg), _dist(yellow_object_cfg)],
        dim=1,
    )  # [N, 4]
    return all_dist[torch.arange(N, device=tid.device), tid]


def target_cube_in_bowl_reward(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    z_min: float = 0.0,
    consecutive_steps: int = 5,
    ee_min_height_above_bowl: float = 0.07,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sparse success reward: target cube inside bowl, no wrong cube inside, gripper retreated."""
    tid = _get_target_color_id(env)
    N = env.num_envs
    bowl: RigidObject = env.scene[bowl_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    bowl_pos = bowl.data.root_pos_w[:, :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    def _in_bowl(obj_pos: torch.Tensor) -> torch.Tensor:
        c1 = torch.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold
        c2 = (obj_pos[:, 2] > bowl_pos[:, 2] + z_min) & (obj_pos[:, 2] < bowl_pos[:, 2] + z_max)
        c3 = ee_pos[:, 2] > (bowl_pos[:, 2] + ee_min_height_above_bowl)
        return c1 & c2 & c3

    all_in = torch.stack(
        [
            _in_bowl(env.scene[red_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[blue_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[green_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl(env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3]),
        ],
        dim=1,
    )  # [N, 4] bool
    target_in = all_in[torch.arange(N, device=tid.device), tid]

    if not hasattr(env, "_target_cube_in_bowl_steps_reward"):
        env._target_cube_in_bowl_steps_reward = torch.zeros(
            env.num_envs, dtype=torch.int32, device=env.device
        )
    prev = env._target_cube_in_bowl_steps_reward.clone()
    env._target_cube_in_bowl_steps_reward = torch.where(
        target_in,
        env._target_cube_in_bowl_steps_reward + 1,
        torch.zeros_like(env._target_cube_in_bowl_steps_reward),
    )
    just_succeeded = (prev == consecutive_steps - 1) & (
        env._target_cube_in_bowl_steps_reward == consecutive_steps
    )
    return just_succeeded.float()


def wrong_cube_grasped_penalty(
    env: ManagerBasedRLEnv,
    grasp_quality_threshold: float = 0.1,
    force_saturation: float = 5.0,
    force_balance_ratio: float = 4.0,
    red_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    blue_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_blue"),
    green_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_green"),
    yellow_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_yellow"),
) -> torch.Tensor:
    """Binary penalty when the gripper forms a real bilateral grasp on any *wrong* cube."""
    tid = _get_target_color_id(env)
    N = env.num_envs

    def _grasp(sensor_cfg):
        return object_grasped_contact_continuous(
            env, force_saturation=force_saturation,
            force_balance_ratio=force_balance_ratio,
            debug_print_interval=0, cube_sensor_cfg=sensor_cfg,
        )

    all_grasp = torch.stack(
        [_grasp(red_sensor_cfg), _grasp(blue_sensor_cfg),
         _grasp(green_sensor_cfg), _grasp(yellow_sensor_cfg)],
        dim=1,
    )  # [N, 4]
    # Zero out the target cube; max over the remaining 3.
    target_mask = torch.zeros(N, 4, device=env.device)
    target_mask.scatter_(1, tid.unsqueeze(1), 1.0)
    max_wrong_grasp = (all_grasp * (1.0 - target_mask)).max(dim=1)[0]
    return (max_wrong_grasp > grasp_quality_threshold).float()


def wrong_cube_in_bowl_penalty(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.055,
    z_max: float = 0.04,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Binary penalty when any *wrong* cube is inside the bowl (XY+Z)."""
    tid = _get_target_color_id(env)
    N = env.num_envs
    bowl: RigidObject = env.scene[bowl_cfg.name]
    bowl_pos = bowl.data.root_pos_w[:, :3]

    def _in_bowl_xy(obj_pos: torch.Tensor) -> torch.Tensor:
        c1 = torch.norm(obj_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold
        c2 = obj_pos[:, 2] < bowl_pos[:, 2] + z_max
        return (c1 & c2).float()

    all_in = torch.stack(
        [
            _in_bowl_xy(env.scene[red_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl_xy(env.scene[blue_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl_xy(env.scene[green_object_cfg.name].data.root_pos_w[:, :3]),
            _in_bowl_xy(env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3]),
        ],
        dim=1,
    )  # [N, 4]
    target_mask = torch.zeros(N, 4, device=env.device)
    target_mask.scatter_(1, tid.unsqueeze(1), 1.0)
    return (all_in * (1.0 - target_mask)).max(dim=1)[0]


def log_target_cube_lifted_pct(
    env: ManagerBasedRLEnv,
    min_height: float = 0.03,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Metric: % of envs where the *target* cube is above min_height."""
    tid = _get_target_color_id(env)
    N = env.num_envs
    all_lifted = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, 2] > min_height,
            env.scene[blue_object_cfg.name].data.root_pos_w[:, 2] > min_height,
            env.scene[green_object_cfg.name].data.root_pos_w[:, 2] > min_height,
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, 2] > min_height,
        ],
        dim=1,
    )  # [N, 4] bool
    target_lifted = all_lifted[torch.arange(N, device=tid.device), tid]
    pct = target_lifted.float().mean().item() * 100.0
    log = env.extras.setdefault("log", {})
    log["Metrics/target_cube_lifted_pct"] = pct
    return torch.zeros(env.num_envs, device=env.device)


def log_cube_lifted_pct(
    env: ManagerBasedRLEnv,
    min_height: float = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Zero-weight metric: logs what % of envs currently have the cube above min_height.

    Writes "Metrics/cube_lifted_pct" to env.extras["log"] every step so RSL-RL
    picks it up and plots it in WandB/TensorBoard alongside the reward curves.
    Use weight=0.0 in RewardsCfg — does not affect training.

    Args:
        min_height: World-frame z threshold (m). Default 0.03 m = ~3 cm above table.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = obj.data.root_pos_w[:, 2] > min_height
    pct = lifted.float().mean().item() * 100.0

    log = env.extras.setdefault("log", {})
    log["Metrics/cube_lifted_pct"] = pct

    return torch.zeros(env.num_envs, device=env.device)


def reaching_target_cube_open_gripper(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    saturation_pos: float = 0.8,
    close_joint_pos: float = 0.2,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Target-cube reaching reward gated by gripper aperture — zero when gripper is closed.

    reward = (1 - tanh(d/std)) × aperture_frac

    Prevents closed-gripper-hover local minimum: reaching reward is zero when the
    gripper is closed, so the policy must choose between open-gripper approach or
    actual bilateral grasp — hovering with a closed gripper gives nothing.
    """
    tid = _get_target_color_id(env)
    N = env.num_envs
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos = all_pos[torch.arange(N, device=tid.device), tid]
    proximity = 1.0 - torch.tanh(torch.norm(target_pos - ee_pos_w, dim=-1) / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    aperture_frac = torch.clamp(
        (gripper_pos - close_joint_pos) / (saturation_pos - close_joint_pos), 0.0, 1.0
    )

    return proximity * aperture_frac


def closed_target_gripper_no_grasp_penalty(
    env: ManagerBasedRLEnv,
    std: float = 0.12,
    saturation_pos: float = 0.8,
    close_joint_pos: float = 0.2,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    red_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_red"),
    blue_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_blue"),
    green_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_green"),
    yellow_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube_yellow"),
) -> torch.Tensor:
    """Penalty for hovering near the target cube with gripper closed but no bilateral grasp.

    penalty = proximity × closed_frac × (1 - grasp_quality)

    Zero when the grasp is good (no double-punishment during a real grasp attempt).
    """
    tid = _get_target_color_id(env)
    N = env.num_envs
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos = all_pos[torch.arange(N, device=tid.device), tid]
    proximity = 1.0 - torch.tanh(torch.norm(target_pos - ee_pos_w, dim=-1) / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    aperture_frac = torch.clamp(
        (gripper_pos - close_joint_pos) / (saturation_pos - close_joint_pos), 0.0, 1.0
    )
    closed_frac = 1.0 - aperture_frac

    def _grasp(sensor_cfg):
        return object_grasped_contact_continuous(env, cube_sensor_cfg=sensor_cfg)

    all_grasp = torch.stack(
        [_grasp(red_sensor_cfg), _grasp(blue_sensor_cfg),
         _grasp(green_sensor_cfg), _grasp(yellow_sensor_cfg)],
        dim=1,
    )  # [N, 4]
    grasp_quality = all_grasp[torch.arange(N, device=tid.device), tid]

    return proximity * closed_frac * (1.0 - grasp_quality)


def target_cube_drop_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Penalty when the *target* cube falls below minimum_height (dropped off table)."""
    tid = _get_target_color_id(env)
    N = env.num_envs
    all_dropped = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[blue_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[green_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, 2] < minimum_height,
        ],
        dim=1,
    )  # [N, 4] bool
    return all_dropped[torch.arange(N, device=tid.device), tid].float()


##
# Visual-coordinate target-cube visibility rewards / metrics
##


def target_cube_visibility_reward(
    env: ManagerBasedRLEnv,
    max_steps: int = 20,
    std_offset: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Reward keeping the TARGET cube near the wrist camera centre (4-cube Task 3).

    Returns:
      +1.0  when target cube is at the image centre.
       0.0  when cube is at NDC radius std_offset from centre.
      -1.0  when cube is off-screen or behind the camera.
       0.0  after step max_steps (set max_steps=99999 to keep always active).
    """
    t = env.episode_length_buf
    active_mask = (t <= max_steps).float()

    tid = _get_target_color_id(env)
    N = env.num_envs
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos_w = all_pos[torch.arange(N, device=tid.device), tid]  # [N, 3]

    u, v, in_view = project_to_wrist_image(env, target_pos_w, robot_cfg=robot_cfg)
    offset = (u.pow(2) + v.pow(2)).sqrt()
    vis_score = torch.where(
        in_view,
        torch.clamp(1.0 - offset / std_offset, min=-1.0, max=1.0),
        torch.full_like(offset, -1.0),
    )
    return vis_score * active_mask


def log_target_cube_visibility_pct(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    red_object_cfg: SceneEntityCfg = SceneEntityCfg("object_red"),
    blue_object_cfg: SceneEntityCfg = SceneEntityCfg("object_blue"),
    green_object_cfg: SceneEntityCfg = SceneEntityCfg("object_green"),
    yellow_object_cfg: SceneEntityCfg = SceneEntityCfg("object_yellow"),
) -> torch.Tensor:
    """Near-zero-weight metric: logs % of envs with target cube in wrist camera FOV.

    Writes Metrics/target_cube_visibility_pct to env.extras["log"] so RSL-RL
    picks it up in WandB/TensorBoard. Use weight=1e-9 in RewardsCfg.
    """
    tid = _get_target_color_id(env)
    N = env.num_envs
    all_pos = torch.stack(
        [
            env.scene[red_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[blue_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[green_object_cfg.name].data.root_pos_w[:, :3],
            env.scene[yellow_object_cfg.name].data.root_pos_w[:, :3],
        ],
        dim=1,
    )  # [N, 4, 3]
    target_pos_w = all_pos[torch.arange(N, device=tid.device), tid]  # [N, 3]

    _, _, in_view = project_to_wrist_image(env, target_pos_w, robot_cfg=robot_cfg)
    pct = in_view.float().mean().item() * 100.0

    if not hasattr(env, "extras"):
        return torch.zeros(env.num_envs, device=env.device)
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["Metrics/target_cube_visibility_pct"] = pct
    return torch.zeros(env.num_envs, device=env.device)
