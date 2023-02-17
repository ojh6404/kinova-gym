import gym
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from collections import namedtuple


# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.utils import seeding

import gym
from gym import spaces
from gym.utils import seeding

from kinova_gym.pybullet import PyBullet
from kinova_gym.envs.robots.kinova import Kinova
from kinova_gym.envs.core import PyBulletRobot

from kinova_gym.utils import distance


class KinovaReachEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        robot (PyBulletRobot): The robot.
        task (Task): The task.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 0.5,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        # control_type: str = "ee"
        control_type: str = "position"
    ) -> None:
        self.sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Kinova(self.sim, block_gripper=True, base_position=np.array(
            [-0.6, 0.0, 0.0]), control_type=control_type)
        self.render_mode = render_mode
        self.metadata["render_fps"] = 1 / self.sim.dt
        self.target_ee_pos = np.array([0.1, 0.2, 0.5])
        self.distance_threshold = 0.1
        self.max_step_length = 1000

        self.goal_range_low = np.array([-0.5 / 2, -0.5 / 2, 0])
        self.goal_range_high = np.array(
            [0.5 / 2, 0.5 / 2, 0.5])

        self.seed = 42

        # arm joint pos (7), ee_pos(3) ,target_pos(3)
        observation_space_low = np.array(
            [-1.] * self.robot.arm_num_dof + [-np.inf] * 6)
        observation_space_high = np.array(
            [1.] * self.robot.arm_num_dof + [np.inf] * 6)

        self.observation_space = spaces.Box(
            observation_space_low, observation_space_high)
        self.action_space = self.robot.action_space

        observation, _ = self.reset()  # required for init; seed can be changed later
        # self.compute_reward = self.task.compute_reward
        # self._saved_goal = dict()  # For state saving and restoring

        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = render_target_position
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(target_position=np.zeros(
                3), distance=0.9, yaw=45, pitch=-30)

    # def reset(
    #     self, seed: Optional[int] = None, options: Optional[dict] = None
    # ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # self.np_random, seed = seeding.np_random(seed=self.seed)

        # self.target_ee_pos = self.np_random(
        #     self.goal_range_low, self.goal_range_high)

        with self.sim.no_rendering():
            self.robot.reset()
            self.sim.timestep_count = 0
            # self.task.reset()
        observation = self._get_obs()
        info = {"is_success": self.is_success(
            self.robot.get_ee_position(), self.target_ee_pos)}
        return observation, info

        # def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
    def step(self, action: np.ndarray):
        self.robot.set_action(action)
        self.sim.step()

        # 13, arm_dof_pos, ee_pos, target_ee_pos
        observation = self._get_obs()

        reward = self.compute_reward(
            self.robot.get_ee_position(), self.target_ee_pos)

        # reset when timeout
        terminated = self.sim.timestep_count > self.max_step_length
        truncated = False
        info = {"is_success": self.is_success(
            self.robot.get_ee_position(), self.target_ee_pos)}
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        return self.sim.render(
            width=self.render_width,
            height=self.render_height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )

    def compute_reward(self, ee_pos, ee_target_pos) -> np.ndarray:
        d = distance(ee_pos, ee_target_pos)
        return -d.astype(np.float32)

    def is_success(self, ee_pos: np.ndarray, ee_target_pos: np.ndarray) -> np.ndarray:
        d = distance(ee_pos, ee_target_pos)
        return np.array(d < self.distance_threshold, dtype=bool)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        arm_joint_pos = np.zeros((7,)).astype(np.float32)
        for i in range(self.robot.arm_num_dof):
            arm_joint_pos[i] = self.robot.get_joint_angle(i)
        arm_joint_pos = self.scale_arm_joint_pos(arm_joint_pos=arm_joint_pos)

        ee_pos = self.robot.get_ee_position()
        target_ee_pos = self.target_ee_pos
        observation = np.concatenate(
            [arm_joint_pos, ee_pos, target_ee_pos]).astype(np.float32)

        # print('debug')
        # print(observation)

        return observation

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = seeding.np_random.uniform(
            self.goal_range_low, self.goal_range_high)
        return goal

    def _create_scene(self) -> None:
        # self.sim.create_plane(z_offset=-0.4)
        self.sim.create_plane(z_offset=0.0)
        # self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            # position=np.zeros(3),
            position=self.target_ee_pos,
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def scale_arm_joint_pos(self, arm_joint_pos) -> np.ndarray:

        arm_joint_mean = (self.robot.arm_lower_limits +
                          self.robot.arm_upper_limits) / 2.
        arm_joint_std = self.robot.arm_joint_ranges / 2.

        scaled_arm_joint_pos = (arm_joint_pos - arm_joint_mean) / arm_joint_std

        return scaled_arm_joint_pos.astype(np.float32)

    def unscale_arm_joint_pos(self, action) -> np.ndarray:

        arm_joint_mean = (self.robot.arm_lower_limits +
                          self.robot.arm_upper_limits) / 2.
        arm_joint_std = self.robot.arm_joint_ranges / 2.

        scaled_arm_joint_pos = arm_joint_mean + arm_joint_std * action

        return scaled_arm_joint_pos.astype(np.float32)
