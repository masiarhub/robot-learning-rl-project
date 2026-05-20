import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import mujoco
import gymnasium as gym
import numpy as np
from rcs.envs.base import GripperWrapper
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

import rcs

logger = logging.getLogger(__name__)

CUBE_COLORS: dict[str, np.ndarray] = {
    "red":    np.array([0.85, 0.15, 0.15, 1.0]),
    "blue":   np.array([0.15, 0.35, 0.85, 1.0]),
    "green":  np.array([0.10, 0.80, 0.15, 1.0]),
    "yellow": np.array([1.00, 0.90, 0.00, 1.0]),
    "orange": np.array([1.00, 0.50, 0.00, 1.0]),
    "violet": np.array([0.55, 0.10, 0.80, 1.0]),
}


class JointVelWrapper(gym.Wrapper):
    """Injects joint_vel into the robot observation dict.

    Reads qvel from sim.data for each named MuJoCo joint and appends
    obs[robot_name]["joint_vel"] as a (len(joint_names),) float64 array.
    """

    def __init__(self, env: gym.Env, robot_name: str, joint_names: list[str]):
        super().__init__(env)
        self.robot_name = robot_name
        self.joint_names = joint_names
        n = len(joint_names)
        spaces = dict(self.observation_space.spaces)
        robot_spaces = dict(spaces[robot_name].spaces)
        robot_spaces["joint_vel"] = gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)
        spaces[robot_name] = gym.spaces.Dict(robot_spaces)
        self.observation_space = gym.spaces.Dict(spaces)

    def _inject(self, obs: dict) -> dict:
        sim = self.get_wrapper_attr("sim")
        vel = np.array([sim.data.joint(name).qvel[0] for name in self.joint_names], dtype=np.float64)
        obs = dict(obs)
        obs[self.robot_name] = dict(obs[self.robot_name])
        obs[self.robot_name]["joint_vel"] = vel
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._inject(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._inject(obs), info


class CubeColorWrapper(gym.Wrapper):
    """Randomizes the cube geom color on each reset.

    Writes directly to model.geom_rgba so the change persists across sim resets.
    If color is None, a random color from CUBE_COLORS is chosen each episode.
    """

    _COLORS: ClassVar[dict[str, np.ndarray]] = CUBE_COLORS

    def __init__(self, env: gym.Env, color: str | None, cube_geom_name: str):
        super().__init__(env)
        self.color = color
        self.cube_geom_name = cube_geom_name

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._apply_color()
        return obs, info

    def _apply_color(self):
        color_name = self.color if self.color is not None else self.np_random.choice(list(self._COLORS))
        rgba = self._COLORS[color_name]
        sim = self.get_wrapper_attr("sim")
        geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, self.cube_geom_name)
        if geom_id >= 0:
            sim.model.geom_rgba[geom_id] = rgba


class CubeMassRandomizerWrapper(gym.Wrapper):
    """Randomizes cube body mass by ±MASS_VARIATION on each reset.

    Scales body_mass and body_inertia together so rotational dynamics stay
    consistent with the new mass. Base values are captured from the compiled
    model on the first reset.
    """

    MASS_VARIATION: ClassVar[float] = 0.30

    def __init__(self, env: gym.Env, cube_body_name: str):
        super().__init__(env)
        self.cube_body_name = cube_body_name
        self._base_mass: float | None = None
        self._base_inertia: np.ndarray | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._randomize_mass()
        return obs, info

    def _randomize_mass(self):
        sim = self.get_wrapper_attr("sim")
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, self.cube_body_name)
        if body_id < 0:
            return
        if self._base_mass is None:
            self._base_mass = float(sim.model.body_mass[body_id])
            self._base_inertia = sim.model.body_inertia[body_id].copy()
        scale = float(self.np_random.uniform(1.0 - self.MASS_VARIATION, 1.0 + self.MASS_VARIATION))
        sim.model.body_mass[body_id] = self._base_mass * scale
        sim.model.body_inertia[body_id] = self._base_inertia * scale


class PickObjSuccessWrapper(gym.Wrapper):
    """
    Wrapper to check if an object is successfully picked up.
    Obj must be lifted 10 cm above its position.
    Computes a reward between 0 and 1 based on:
    - TCP to object distance
    - cube z height
    - whether the arm is standing still once the task is solved.
    """

    # In continuous gripper mode obs["gripper"] is a normalised position in [0, 1]
    # where 0 = fully open and 1 = fully closed.  The policy's default open value is
    # ~0.195; anything above this threshold indicates an active close command.
    _CONTINUOUS_CLOSE_THRESHOLD: ClassVar[float] = 0.22

    def __init__(self, env, robot_name: str, shared2world: rcs.common.Pose,
                 obj_joint_name="box_joint", binary_gripper: bool = True):
        super().__init__(env)
        self.sim = self.env.get_wrapper_attr("sim")
        self.obj_joint_name = obj_joint_name
        self._gripper_closing = 0
        self.robot_name = robot_name
        self._gripper = self.get_wrapper_attr("gripper")[self.robot_name]
        self.shared2world = shared2world
        self._binary_gripper = binary_gripper

    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)

        gripper_val = obs[self.robot_name]["gripper"][0]
        if self._binary_gripper:
            gripper_closed = gripper_val == GripperWrapper.BINARY_GRIPPER_CLOSED[0]
        else:
            gripper_closed = gripper_val > self._CONTINUOUS_CLOSE_THRESHOLD

        if (
            self._gripper.get_normalized_width() > 0.01
            and self._gripper.get_normalized_width() < 0.99
            and gripper_closed
        ):
            self._gripper_closing += 1
        else:
            self._gripper_closing = 0

        obj_pose_in_world = rcs.common.Pose(translation=self.sim.data.joint(self.obj_joint_name).qpos[:3])

        # obj_pose = self._robot.to_pose_in_robot_coordinates(obj_pose)

        obj_pose_in_shared = self.shared2world.inverse() * obj_pose_in_world

        # tcp_to_obj_dist = np.linalg.norm(obj_pose.translation() - self._robot.get_cartesian_position().translation())
        tcp_to_obj_dist = np.linalg.norm(obj_pose_in_shared.translation() - obs[self.robot_name]["tquat"][:3])

        obj_to_goal_dist = 0.10 - min(obj_pose_in_shared.translation()[-1], 0.10)
        # obj_to_goal_dist = np.linalg.norm(cube_pose.translation() - self.home_pose.translation())

        # NOTE: 4 depends on the time passing between each step.
        is_grasped = (
            self._gripper_closing >= 4  # gripper is closing since more than 4 steps
            and gripper_closed  # command is still close
            and tcp_to_obj_dist <= 0.01  # tcp to cube center is max 1cm
        )
        success = obj_to_goal_dist <= 0.022 and info[self.robot_name]["is_grasped"]
        movement = np.linalg.norm(self.sim.data.qvel)

        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        static_reward = 1 - np.tanh(5 * movement)
        info["is_grasped"] = is_grasped
        info["success"] = success
        reward = reaching_reward + place_reward * is_grasped + static_reward * success
        reward /= 3  # type: ignore
        return obs, reward, success, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()
        # self.home_pose = self._robot.get_cartesian_position()
        return obs, info


class RandomSquareObjPos(gym.Wrapper):
    """Wrapper to an object in a simulated environment in a random spot inside a defined square.

    Works only for single robot
    """

    def __init__(
        self,
        env: gym.Env,
        center2world: rcs.common.Pose,
        include_rotation: bool = True,
        obj_joint_name: str = "box_joint",
        x_width: float = 0.2,
        y_width: float = 0.2,
    ):
        super().__init__(env)
        self.include_rotation = include_rotation
        self.obj_joint_name = obj_joint_name
        self.center2world = center2world
        self.x_width = x_width
        self.y_width = y_width

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:

        # place randomly in square
        pos_x = self.np_random.uniform(-self.x_width / 2, self.x_width / 2)
        pos_y = self.np_random.uniform(-self.y_width / 2, self.y_width / 2)

        if self.include_rotation:
            # 1. Sample a random angle between 0 and 2*pi (360 degrees)
            theta = self.np_random.uniform(0, 2 * np.pi)

            # 2. Convert the angle to a unit quaternion for the Z-axis
            qw = np.cos(theta / 2)
            qz = np.sin(theta / 2)
        else:
            # No rotation (Identity quaternion)
            qw = 1.0
            qz = 0.0

        pose_in_center_frame = rcs.common.Pose(
            translation=np.array([pos_x, pos_y, 0]), quaternion=np.array([0, 0, qz, qw])
        )
        pose_in_world_frame = self.center2world * pose_in_center_frame

        # qpos array format for a free joint: [x, y, z, qw, qx, qy, qz]
        self.get_wrapper_attr("sim").data.joint(self.obj_joint_name).qpos = np.append(
            pose_in_world_frame.translation(), pose_in_world_frame.rotation_q_wxyz()
        )

        # reset of remaining stack, must happen after our reset!
        return super().reset(seed=seed, options=options)


class ConstrainedBowlAndCubePlacer(gym.Wrapper):
    """Randomizes bowl and cube XY positions each episode with two hard constraints:

    1. Both within *max_robot_dist* metres of the robot base (XY Euclidean from origin).
    2. Cube at least *min_cube_bowl_dist* metres from the bowl centre (XY Euclidean).

    The bowl is a static MJCF body — its position is updated by writing to
    model.body_pos, which persists until the next reset.  The cube is moved via
    its free joint (same mechanism as RandomSquareObjPos).

    Rejection sampling is used; after MAX_ATTEMPTS the last valid pair is used,
    or a hard-coded safe fallback if no valid pair was ever found.
    """

    MAX_ATTEMPTS: ClassVar[int] = 200

    def __init__(
        self,
        env: gym.Env,
        cube_joint_name: str,
        bowl_body_name: str,
        cube_z: float,
        bowl_z: float,
        min_cube_bowl_dist: float = 0.12,
        max_robot_dist: float = 0.30,
        include_cube_rotation: bool = True,
    ):
        super().__init__(env)
        self.cube_joint_name = cube_joint_name
        self.bowl_body_name = bowl_body_name
        self.cube_z = cube_z
        self.bowl_z = bowl_z
        self.min_cube_bowl_dist = min_cube_bowl_dist
        self.max_robot_dist = max_robot_dist
        self.include_cube_rotation = include_cube_rotation
        self._bowl_body_id: int | None = None

    def _sample_xy_in_disk(self, radius: float) -> np.ndarray:
        # Uniform sampling in a disk via polar coordinates (square-root trick).
        r = radius * np.sqrt(self.np_random.uniform(0.0, 1.0))
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        sim = self.get_wrapper_attr("sim")

        if self._bowl_body_id is None:
            self._bowl_body_id = mujoco.mj_name2id(
                sim.model, mujoco.mjtObj.mjOBJ_BODY, self.bowl_body_name
            )

        r = self.max_robot_dist
        min_d = self.min_cube_bowl_dist

        # --- rejection sample ---
        bowl_xy = np.array([0.25, 0.10])
        cube_xy = np.array([0.20, -0.10])
        success = False
        for _ in range(self.MAX_ATTEMPTS):
            b = self._sample_xy_in_disk(r)
            c = self._sample_xy_in_disk(r)
            if np.linalg.norm(c - b) >= min_d:
                bowl_xy, cube_xy = b, c
                success = True
                break
        if not success:
            logger.warning(
                "ConstrainedBowlAndCubePlacer: no valid placement found in %d attempts "
                "(min_dist=%.2f, max_r=%.2f); using fallback positions.",
                self.MAX_ATTEMPTS, min_d, r,
            )

        # --- place bowl (static body: write model config, then recompute kinematics) ---
        if self._bowl_body_id >= 0:
            sim.model.body_pos[self._bowl_body_id][0] = float(bowl_xy[0])
            sim.model.body_pos[self._bowl_body_id][1] = float(bowl_xy[1])
            sim.model.body_pos[self._bowl_body_id][2] = self.bowl_z
            mujoco.mj_kinematics(sim.model, sim.data)

        # --- place cube (free joint: write to joint qpos) ---
        if self.include_cube_rotation:
            theta = self.np_random.uniform(0.0, 2.0 * np.pi)
            qw = float(np.cos(theta / 2))
            qz = float(np.sin(theta / 2))
        else:
            qw, qz = 1.0, 0.0

        # Free joint qpos layout: [x, y, z, qw, qx, qy, qz]
        sim.data.joint(self.cube_joint_name).qpos[:] = np.array(
            [cube_xy[0], cube_xy[1], self.cube_z, qw, 0.0, 0.0, qz],
            dtype=np.float64,
        )

        return super().reset(seed=seed, options=options)


@dataclass(kw_only=True)
class PickTaskConfig(BaseTaskConfig):
    robot_name: str
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, 0.0, 0.05]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    object_xml = rcs.OBJECT_PATHS["green_cube"]
    object_joint: str = "box_joint"
    prefix: str = "PickTask_"
    include_rotation: bool = True
    x_width: float = 0.2
    y_width: float = 0.2
    task_id: str = "pick"
    # Constrained placement: replaces RandomSquareObjPos when enabled.
    constrain_placement: bool = False
    min_cube_bowl_dist: float = 0.12    # metres — cube must be at least this far from bowl
    max_robot_dist: float = 0.30        # metres — both objects within this radius of the base
    bowl_body_name: str = "bowl_bowl_body"
    bowl_z: float = 0.003               # bowl z in world frame (fixed; only XY is randomised)


class PickTask(Task[PickTaskConfig]):

    @staticmethod
    def add_task_mujoco(cfg: PickTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        composer.add_object_world_frame(
            cfg.object_xml,
            object_prefix=cfg.prefix,
            pose=object2world,
            register_root_relative_replay_free_joints=True,
        )

    @staticmethod
    def add_task_env(cfg: PickTaskConfig, env: gym.Env, _simulation: Sim, env_cfg: SimEnvCreatorConfig) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world
        shared2world = env_cfg.shared_base_frame_to_root_frame * env_cfg.root_frame_to_world
        object_joint = cfg.prefix + cfg.object_joint
        env = PickObjSuccessWrapper(env, cfg.robot_name, shared2world, object_joint,
                                    binary_gripper=env_cfg.wrapper_cfg.binary_gripper)
        if cfg.constrain_placement:
            env = ConstrainedBowlAndCubePlacer(
                env,
                cube_joint_name=object_joint,
                bowl_body_name=cfg.bowl_body_name,
                cube_z=float(object2world.translation()[2]),
                bowl_z=cfg.bowl_z,
                min_cube_bowl_dist=cfg.min_cube_bowl_dist,
                max_robot_dist=cfg.max_robot_dist,
                include_cube_rotation=cfg.include_rotation,
            )
        else:
            env = RandomSquareObjPos(
                env, center2world=object2world, include_rotation=cfg.include_rotation,
                obj_joint_name=object_joint, x_width=cfg.x_width, y_width=cfg.y_width,
            )
        return CubeMassRandomizerWrapper(env, cube_body_name=cfg.prefix + "box_body")


rcs.TASKS["pick"] = PickTask


class MultiCubeColorWrapper(gym.Wrapper):
    """Assigns distinct random colors to multiple cube geoms on each reset.

    If colors is None, n distinct colors are sampled without replacement from
    CUBE_COLORS on every reset. Pass a fixed list to pin specific colors.
    """

    def __init__(self, env: gym.Env, cube_geom_names: list[str], colors: list[str] | None = None):
        super().__init__(env)
        self.cube_geom_names = cube_geom_names
        self.colors = colors

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._apply_colors()
        return obs, info

    def _apply_colors(self):
        n = len(self.cube_geom_names)
        if self.colors is not None:
            selected = list(self.colors[:n])
        else:
            selected = self.np_random.choice(list(CUBE_COLORS), size=n, replace=False).tolist()
        sim = self.get_wrapper_attr("sim")
        for geom_name, color_name in zip(self.cube_geom_names, selected):
            geom_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                sim.model.geom_rgba[geom_id] = CUBE_COLORS[color_name]
