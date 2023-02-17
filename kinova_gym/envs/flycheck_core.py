from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from collections import namedtuple

# import gymnasium as gym
import numpy as np
# from gymnasium import spaces
# from gymnasium.utils import seeding
import gym
from gym import spaces
from gym.utils import seeding

from kinova_gym.pybullet import PyBullet


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        body_name (str): The name of the robot within the simulation.
        file_name (str): Path of the urdf file.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
    """

    def __init__(
        self,
        sim: PyBullet,
        body_name: str,
        file_name: str,
        base_position: np.ndarray,
        action_space: spaces.Space,
        joint_indices: np.ndarray,
        # joint_forces: np.ndarray,
        control_type: str,
        arm_num_dof: int,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        self.arm_num_dof = arm_num_dof
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position)
            self._parse_joint_info(self.body_name)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = self.arm_torque_limits
        self.control_type = control_type

    def _load_robot(self, file_name: str, base_position: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            useFixedBase=True,
        )

    def _parse_joint_info(self, body: str):
        numJoints = self.sim.physics_client.getNumJoints(
            self.sim._bodies_idx[body])
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce', 'maxVelocity', 'controllable'])
        self.joints_info = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self.sim.physics_client.getJointInfo(
                self.sim._bodies_idx[body], i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointType = info[2]
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != self.sim.physics_client.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints_info.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dof
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dof]

        self.arm_lower_limits = np.array([
            info.lowerLimit for info in self.joints_info if info.controllable][:self.arm_num_dof])
        self.arm_upper_limits = np.array([
            info.upperLimit for info in self.joints_info if info.controllable][:self.arm_num_dof])
        self.arm_joint_ranges = np.array([
            info.upperLimit - info.lowerLimit for info in self.joints_info if info.controllable][:self.arm_num_dof])
        self.arm_torque_limits = np.array([
            info.maxForce for info in self.joints_info if info.controllable][:self.arm_num_dof])

        self.link_name_to_index = {self.sim.physics_client.getBodyInfo(
            self.sim._bodies_idx[body])[0].decode('UTF-8'): -1, }
        for _id in range(self.sim.physics_client.getNumJoints(self.sim._bodies_idx[body])):
            _name = self.sim.physics_client.getJointInfo(
                self.sim._bodies_idx[body], _id)[12].decode('UTF-8')
            self.link_name_to_index[_name] = _id

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): The action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, control_actions: np.ndarray, Kp: np.ndarray, Kd: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """

        if self.control_type in ("ee", "position"):
            self.sim.control_joints_pos(
                body=self.body_name,
                joints=self.joint_indices,
                target_angles=control_actions,
                forces=self.joint_forces,
                Kp=Kp,
                Kd=Kd,
            )
        elif self.control_type == "torque":
            self.sim.control_joints_torque(
                body=self.body_name,
                joints=self.joint_indices,
                forces=control_actions,
            )
        else:
            raise NotImplementedError

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(
            self.body_name, joints=self.joint_indices, angles=angles)

    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(
            self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics
