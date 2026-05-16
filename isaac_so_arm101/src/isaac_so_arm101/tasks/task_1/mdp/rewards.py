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


def gripper_closed_near_object(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    open_joint_pos: float = 0.2,
    close_joint_pos: float = -0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Reward for closing the gripper when the end-effector is close to the cube.

    reward = proximity(ee_cube_dist, std) × gripper_closed_fraction

    proximity uses the same tanh kernel as reaching_object but with a much tighter
    std so the reward is essentially zero beyond a few centimetres — preventing the
    policy from exploiting this by closing the gripper while far from the cube.

    gripper_closed_fraction is 1 when the gripper joint is at close_joint_pos and
    0 when at open_joint_pos, giving continuous gradient on the gripper action.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]          # [B, 3]
    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_pos_w, dim=-1)  # [B]
    proximity = 1.0 - torch.tanh(dist / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]   # [B]
    closed_frac = torch.clamp(
        (open_joint_pos - gripper_pos) / (open_joint_pos - close_joint_pos),
        0.0, 1.0,
    )

    return proximity * closed_frac


from isaaclab.sensors import ContactSensor

def object_grasped_contact(
    env: ManagerBasedRLEnv,
    min_force_per_finger: float = 0.3,
    force_balance_ratio: float = 4.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["gripper_link", "moving_jaw_so101_v1_link"],
    ),
) -> torch.Tensor:
    """Reward for both gripper fingers making balanced contact with the cube."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    finger_forces = net_forces[:, :, robot_cfg.body_ids, :]   # [N, history, 2, 3]
    finger_force_norms = finger_forces.norm(dim=-1).max(dim=1)[0]

    left_force  = finger_force_norms[:, 0]
    right_force = finger_force_norms[:, 1]

    both_touching = (left_force > min_force_per_finger) & (right_force > min_force_per_finger)

    ratio = torch.maximum(left_force, right_force) / (
        torch.minimum(left_force, right_force) + 1e-6
    )
    balanced = ratio < force_balance_ratio

    return (both_touching & balanced).float()


def robot_body_cube_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when any non-gripper robot link touches the cube."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    body_forces = net_forces[:, :, robot_cfg.body_ids, :]  # [N, history, n_links, 3]

    max_force = body_forces.norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()


def robot_table_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_table"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty when any robot link touches the table."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    body_forces = net_forces[:, :, robot_cfg.body_ids, :]

    max_force = body_forces.norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()


def robot_bowl_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_bowl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=[
            "shoulder_link",
            "upper_arm_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "moving_jaw_so101_v1_link",
        ],
    ),
) -> torch.Tensor:
    """Penalty when robot links touch the bowl."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    body_forces = net_forces[:, :, robot_cfg.body_ids, :]  # [N, history, n_links, 3]

    max_force = body_forces.norm(dim=-1).max(dim=-1)[0].max(dim=-1)[0]  # [N]
    return (max_force > threshold).float()


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
