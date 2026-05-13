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
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    # root_pos_w[:, 2] is the world-frame z-coordinate of the object's root body.
    # Returns 1.0 per env where the object is above minimal_height, else 0.0.
    # This is a binary (non-differentiable) gate: it provides a clear mode-switching
    # signal that tells the agent it has successfully left the table surface.
    # minimal_height is set to 0.02 m (2 cm) — just enough to confirm the object
    # is airborne, not resting on the table (object starts at z ≈ 0.01 m).
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

# def object_is_lifted(
#     env: ManagerBasedRLEnv, minimal_height: float, saturation_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     object: RigidObject = env.scene[object_cfg.name]
#     z = object.data.root_pos_w[:, 2]

#     return torch.clamp((z - minimal_height) / (saturation_height - minimal_height), 0.0, 1.0)


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
    xy_threshold: float = 0.06,
    z_max: float = 0.05,
    gripper_open_threshold: float = 0.35,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Binary success reward: 1.0 when the cube is inside the bowl AND the gripper is open.

    Three conditions must all hold simultaneously:

    C1  XY proximity   horizontal dist(cube, bowl) < xy_threshold
                       Bowl inner radius ≈ 0.06 m at scale 1.35.
                       Increase if valid placements are not being counted.

    C2  Z height       cube_z < bowl_z + z_max
                       Bowl walls are ~0.05 m tall; z_max=0.05 keeps the cube below the rim.
                       Increase for a deeper bowl; decrease to require the cube to sit low.

    C3  Gripper open   gripper joint position ≥ gripper_open_threshold
                       Open command = 0.5 rad; default threshold 0.35 filters half-open grasps.
                       Decrease if valid releases are missed; increase to require near-full open.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cube_pos = obj.data.root_pos_w[:, :3]   # (num_envs, 3)
    bowl_pos = bowl.data.root_pos_w[:, :3]  # (num_envs, 3)

    # C1 — XY proximity: cube centre within bowl inner radius.
    c1 = torch.norm(cube_pos[:, :2] - bowl_pos[:, :2], dim=1) < xy_threshold

    # C2 — Z height: cube centre below bowl_z + z_max (inside the bowl walls).
    c2 = cube_pos[:, 2] < (bowl_pos[:, 2] + z_max)

    # C3 — Gripper open: joint position at or near the open command (0.5 rad).
    # robot_cfg.joint_ids is resolved from joint_names=["gripper"] by the reward manager.
    c3 = robot.data.joint_pos[:, robot_cfg.joint_ids[0]] >= gripper_open_threshold

    return (c1 & c2 & c3).float()


def object_grasped(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    gripper_closed_threshold: float = 0.30,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Reward for closing the gripper while the EE is near the cube.

    Gripper joint: 0 rad = fully closed, 0.5 rad = fully open.
    Only fires when gripper is near-closed (< gripper_closed_threshold) AND
    the EE is close to the cube (tanh proximity kernel with given std).
    This gives an explicit gradient signal for grasping — without it the policy
    must accidentally discover that closing the gripper leads to lifting reward.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    cube_pos_w = object.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    distance = torch.norm(cube_pos_w - ee_pos_w, dim=1)
    proximity = 1 - torch.tanh(distance / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
    gripper_closed = (gripper_pos < gripper_closed_threshold).float()

    return gripper_closed * proximity


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
