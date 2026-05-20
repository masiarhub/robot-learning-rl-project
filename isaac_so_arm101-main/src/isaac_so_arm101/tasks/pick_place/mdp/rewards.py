# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(env, start_height=0.12, saturation_height=0.2, min_reward=0.0, object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    ramp = torch.clamp((height - start_height) / (saturation_height - start_height), 0.0, 1.0)
    above = (height > start_height).float()
    return above * (min_reward + (1.0 - min_reward) * ramp)


def object_ee_distance(env, std, object_cfg=SceneEntityCfg("object"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return 1 - torch.tanh(torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1) / std)


def object_released_in_zone(env, threshold, target_cfg=SceneEntityCfg("bowl_bottom"), object_cfg=SceneEntityCfg("object"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    target: RigidObject = env.scene[target_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    near_bowl = torch.norm(target.data.root_pos_w[:, :3] - object.data.root_pos_w[:, :3], dim=1) < threshold
    gripper_open = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1) > 0.05
    return (near_bowl & gripper_open).float()


def object_bowl_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    height_offset: float = 0.0,
    debug_vis: bool = False,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_bottom"),  # was "bowl"
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

def gripper_aperture_reward(
    env,
    std: float = 0.05,
    saturation_pos: float = 0.15,
    close_joint_pos: float = -0.1,
    target_open_pos: float = 0.2,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist = torch.norm(obj.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=-1)
    far_from_cube = torch.tanh(dist / std)

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][:, 0]
    open_frac = 1.0 - torch.clamp(
        torch.abs(gripper_pos - target_open_pos) / (target_open_pos - close_joint_pos),
        0.0, 1.0,
    )

    return open_frac * far_from_cube  


def gripper_force_reward(env, target_force=3.0, force_tolerance=1.5, force_max=8.0, proximity_std=0.04,
                          debug_print_interval=0, object_cfg=SceneEntityCfg("object"),
                          ee_frame_cfg=SceneEntityCfg("ee_frame"),
                          cube_sensor_cfg=SceneEntityCfg("contact_forces_cube")):
    """Reward for applying the right amount of bilateral force to the cube.

    Gaussian bell centred at target_force — peaks at 1.0, falls off with
    sigma=force_tolerance, collapses above force_max to penalise crushing.
    Gated by proximity so the policy can't earn reward squeezing air.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]

    proximity = 1.0 - torch.tanh(
        torch.norm(obj.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=-1) / proximity_std
    )

    force_matrix = cube_sensor.data.force_matrix_w_history   # [N, H, 1, 2, 3]
    finger_norms = force_matrix[:, :, 0, :, :].norm(dim=-1).max(dim=1)[0]  # [N, 2]
    left_f  = finger_norms[:, 0]
    right_f = finger_norms[:, 1]
    mean_f  = (left_f + right_f) / 2.0

    if debug_print_interval > 0 and env.common_step_counter % debug_print_interval == 0:
        print(f"[GRIPPER FORCE] step={env.common_step_counter}  left={left_f[0].item():.2f}N  right={right_f[0].item():.2f}N  mean={mean_f[0].item():.2f}N  target={target_force}N")

    gaussian     = torch.exp(-0.5 * ((mean_f - target_force) / force_tolerance) ** 2)
    not_crushing = torch.clamp(1.0 - (mean_f - force_max).clamp(0.0) / force_tolerance, 0.0, 1.0)
    force_quality = gaussian * not_crushing
    balance = torch.minimum(left_f, right_f) / (torch.maximum(left_f, right_f) + 1e-6)
    return force_quality * balance * proximity


def object_grasped_contact_continuous(env, force_saturation=5.0, force_balance_ratio=4.0, debug_print_interval=0,
                                       cube_sensor_cfg=SceneEntityCfg("contact_forces_cube")):
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]
    force_matrix  = cube_sensor.data.force_matrix_w_history
    finger_forces = force_matrix[:, :, 0, :, :]
    finger_norms  = finger_forces.norm(dim=-1).max(dim=1)[0]
    left_f, right_f = finger_norms[:, 0], finger_norms[:, 1]

    if debug_print_interval > 0 and env.common_step_counter % debug_print_interval == 0:
        print(f"[GRASP FORCE] step={env.common_step_counter}  left={left_f[0].item():.2f}N  right={right_f[0].item():.2f}N")

    min_force    = torch.minimum(left_f, right_f)
    force_reward = torch.tanh(min_force / force_saturation)
    ratio        = torch.maximum(left_f, right_f) / (torch.minimum(left_f, right_f) + 1e-6)
    balance      = torch.clamp(1.0 - (ratio - 1.0) / (force_balance_ratio - 1.0), 0.0, 1.0)

    best_t = finger_forces.norm(dim=-1).sum(dim=-1).argmax(dim=1)
    idx    = best_t[:, None, None, None].expand(-1, 1, 2, 3)
    peak   = finger_forces.gather(dim=1, index=idx).squeeze(1)
    cos_sim = (peak[:, 0, :] * peak[:, 1, :]).sum(dim=-1) / (peak[:, 0, :].norm(dim=-1) * peak[:, 1, :].norm(dim=-1) + 1e-6)
    opposition = (-cos_sim).clamp(0.0, 1.0)
    return force_reward * balance * opposition


def lifting_object_grasped(env, start_height=0.012, saturation_height=0.02, min_reward=0.0,
                             force_saturation=5.0, force_balance_ratio=3.0,
                             object_cfg=SceneEntityCfg("object"), cube_sensor_cfg=SceneEntityCfg("contact_forces_cube")):
    lift  = object_is_lifted(env, start_height, saturation_height, min_reward, object_cfg)
    grasp = object_grasped_contact_continuous(env, force_saturation, force_balance_ratio, 0, cube_sensor_cfg)
    return lift * grasp


def robot_table_contact_penalty(env, threshold=1.0, sensor_cfg=SceneEntityCfg("contact_forces_table")):
    """Penalty when upper_arm_link contacts anything (almost always the table).
    Sensor is robot-side, shape: [N, H, 1, 3] via net_forces_w_history.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    max_force = sensor.data.net_forces_w_history[:, :, 0, :].norm(dim=-1).max(dim=-1)[0]
    return (max_force > threshold).float()


def time_penalty_if_not_lifted(env, start_height=0.05, cube_sensor_cfg=SceneEntityCfg("contact_forces_cube"), object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]
    time_frac  = (env.episode_length_buf / env.max_episode_length).clamp(0.0, 1.0)
    lifted     = (obj.data.root_pos_w[:, 2] > start_height).float()
    finger_forces = cube_sensor.data.force_matrix_w_history[:, :, 0, :, :].norm(dim=-1).max(dim=1)[0]
    grasped    = (finger_forces.sum(dim=-1) > 0.5).float()
    return time_frac * (1.0 - torch.clamp(lifted + grasped, 0.0, 1.0))

def gripper_close_when_near(
    env, proximity_std: float, gripper_target: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    # distance from EE to cube
    ee_pos = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    cube_pos = env.scene["object"].data.root_pos_w
    dist = torch.norm(ee_pos - cube_pos, dim=-1)
    proximity = torch.exp(-dist**2 / proximity_std**2)  # 1.0 when close, 0 when far

    # gripper joint position (how closed it is)
    robot = env.scene[asset_cfg.name]
    gripper_pos = robot.data.joint_pos[:, asset_cfg.joint_ids[0]]
    closing = torch.clamp(gripper_target - gripper_pos, min=0.0)  # positive when closing

    return proximity * closing

def cube_moved_before_grasp_penalty(
    env,
    height_threshold: float = 0.03,
    force_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cube_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_cube"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    cube_sensor: ContactSensor = env.scene.sensors[cube_sensor_cfg.name]

    # check if cube is being grasped (both fingers have contact)
    force_matrix = cube_sensor.data.force_matrix_w_history
    finger_norms = force_matrix[:, :, 0, :, :].norm(dim=-1).max(dim=1)[0]
    both_fingers_contact = (finger_norms[:, 0] > force_threshold) & (finger_norms[:, 1] > force_threshold)

    # cube moving horizontally while on table and NOT grasped
    on_table = (obj.data.root_pos_w[:, 2] < height_threshold).float()
    horizontal_vel = torch.norm(obj.data.root_lin_vel_w[:, :2], dim=-1)
    not_grasped = (~both_fingers_contact).float()

    return on_table * horizontal_vel * not_grasped