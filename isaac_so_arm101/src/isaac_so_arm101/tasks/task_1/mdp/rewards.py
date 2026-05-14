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
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv,
    start_height: float = 0.12,
    saturation_height: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous reward for lifting the object, linear ramp from start_height to saturation_height.

    Below start_height: reward is 0.0.
    Between start_height and saturation_height: linearly interpolated from 0.0 to 1.0.
    Above saturation_height: constant 1.0.

    Args:
        env: Environment instance.
        start_height: Height where reward starts ramping up (0.12 m).
        saturation_height: Height where reward saturates at 1.0 (0.2 m).
        object_cfg: Configuration for the object.

    Returns:
        Per-environment reward tensor of shape (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]  # z-coordinate in world frame
    return torch.clamp((height - start_height) / (saturation_height - start_height), 0.0, 1.0)


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
    ee_min_height_above_bowl: float = 0.055,
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


# def object_grasped(
#     env: ManagerBasedRLEnv,
#     std: float = 0.03,
#     gripper_closed_threshold: float = 0.30,
#     min_lift_height: float = 0.015,   # cube must be a bit above table
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
# ) -> torch.Tensor:
#     """Grasp shaping: gripper closed, EE very close, and cube slightly lifted."""

#     obj: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot: Articulation = env.scene[robot_cfg.name]

#     # Positions
#     cube_pos_w = obj.data.root_pos_w[:, :3]
#     ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

#     # Distance EE–cube
#     distance = torch.norm(cube_pos_w - ee_pos_w, dim=1)
#     proximity = 1.0 - torch.tanh(distance / std)  # tight kernel

#     # Gripper closed?
#     gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
#     gripper_closed = (gripper_pos < gripper_closed_threshold).float()

#     # Cube lifted slightly off the table?
#     cube_lifted = (cube_pos_w[:, 2] > min_lift_height).float()

#     # Reward only when: closed AND very close AND cube slightly lifted
#     return gripper_closed * proximity * cube_lifted

# def object_grasped(
#     env: ManagerBasedRLEnv,
#     half_width: float = 0.012,      # half-distance between fingers (Y-axis in gripper frame)
#     half_depth: float = 0.012,      # acceptable range along X-axis in gripper frame
#     half_height: float = 0.012,     # acceptable range along Z-axis in gripper frame
#     gripper_closed_threshold: float = 0.10,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
# ) -> torch.Tensor:
#     """Binary reward when the cube is inside the gripper workspace AND the gripper is closed.

#     Transforms the cube's world position into the end-effector's local frame and checks
#     whether it lies within a bounding box representing the region between the two jaws.
#     The gripper joint must also be below *gripper_closed_threshold* for the reward to fire.

#     The SO-ARM101 gripper jaws open/close along the local Y axis (moving jaw at +Y rotates
#     toward the fixed jaw at Y≈0).  The EE frame is anchored on ``gripper_link`` with a
#     small offset to approximate the finger centre.

#     Args:
#         env: Environment instance.
#         half_width: Half-extent along local Y (between-fingers direction).
#         half_depth: Half-extent along local X (finger-width direction).
#         half_height: Half-extent along local Z (gripper-height direction).
#         gripper_closed_threshold: Maximum joint position to count as "closed".
#             Gripper: 0 rad = closed, 0.5 rad = open.
#         object_cfg: Configuration for the cube object.
#         ee_frame_cfg: Configuration for the end-effector frame transformer.
#         robot_cfg: Configuration for the robot articulation (gripper joint).

#     Returns:
#         Per-environment reward tensor of shape (num_envs,), values in {0.0, 1.0}.
#     """
#     obj: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     robot: Articulation = env.scene[robot_cfg.name]

#     # World-frame positions: (num_envs, 3)
#     cube_pos_w = obj.data.root_pos_w[:, :3]
#     ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
#     ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # wxyz convention

#     # Transform cube position from world frame → EE local frame
#     pos_diff = cube_pos_w - ee_pos_w  # (num_envs, 3)
#     cube_pos_local = quat_apply_inverse(ee_quat_w, pos_diff)  # (num_envs, 3)

#     # Bounding-box check: cube must be inside the gripper workspace
#     inside_x = torch.abs(cube_pos_local[:, 0]) < half_depth
#     inside_y = torch.abs(cube_pos_local[:, 1]) < half_width
#     inside_z = torch.abs(cube_pos_local[:, 2]) < half_height
#     inside_box = inside_x & inside_y & inside_z

#     # Gripper must be closed
#     gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
#     gripper_closed = gripper_pos < gripper_closed_threshold

#     return (inside_box & gripper_closed).float()

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

    # [N, history, num_bodies, 3] → only finger bodies
    net_forces = contact_sensor.data.net_forces_w_history
    finger_forces = net_forces[:, :, robot_cfg.body_ids, :]   # [N, history, 2, 3]

    # Max force norm per finger over history window → [N, 2]
    finger_force_norms = finger_forces.norm(dim=-1).max(dim=1)[0]

    left_force  = finger_force_norms[:, 0]   # [N]
    right_force = finger_force_norms[:, 1]   # [N]

    # Both fingers must be pressing the cube
    both_touching = (left_force > min_force_per_finger) & (right_force > min_force_per_finger)

    # Grip must be roughly balanced (rules out one-sided scraping)
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
