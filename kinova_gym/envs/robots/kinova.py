from typing import Optional
import os

import numpy as np
# from gymnasium import spaces
from gym import spaces

from kinova_gym.envs.core import PyBulletRobot
from kinova_gym.pybullet import PyBullet
from kinova_gym import ASSET_PATH


class Kinova(PyBulletRobot):
    """Kinova robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
        arm_num_dof: int = 7,
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(
            3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.action_scale = 1.0
        # self.arm_num_dof = arm_num_dof

        # control (x, y z) if "ee", else, control the 7 joints
        n_action = 3 if control_type == "ee" else 7
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0,
                                  shape=(n_action,), dtype=np.float32)
        asset_path = os.path.join(
            ASSET_PATH, "urdf/gen3_robotiq_2f_140.urdf")
        super().__init__(
            sim,
            body_name="kinova",
            file_name=asset_path,
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6]),  # TODO
            # joint_forces=np.array(self.arm_torque_limits),  # TODO
            control_type=control_type,
            arm_num_dof=arm_num_dof
        )

        # NOTE: set gripper constraint for closed loop gripper
        self._post_load(body=self.body_name)

        # self.fingers_indices = np.array([9, 10])  # TODO
        self.gripper_index = self.mimic_parent_id
        self.gripper_range = [-1.0, 1.0]
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])  # TODO
        # self.ee_link = 11
        self.ee_link = self.link_name_to_index['end_effector_link']  # TODO

        self.Kp = np.array([100.0] * self.arm_num_dof)
        self.Kd = np.array([50.0] * self.arm_num_dof)

        self.arm_torque = np.zeros_like(self.arm_torque_limits)

        for i in range(self.arm_num_dof):
            self.sim.physics_client.enableJointForceTorqueSensor(
                self.sim._bodies_idx[self.body_name], jointIndex=i, enableSensor=True)

        # NOTE: set sim physics property
        # self.sim.set_lateral_friction(
        #     self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        # self.sim.set_lateral_friction(
        #     self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        # self.sim.set_spinning_friction(
        #     self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        # self.sim.set_spinning_friction(
        #     self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def _post_load(self, body):
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': -1,
                                'left_inner_knuckle_joint': -1,
                                'right_inner_knuckle_joint': -1,
                                'left_inner_finger_joint': 1,
                                'right_inner_finger_joint': 1}
        self._setup_mimic_joints(
            body, mimic_parent_name, mimic_children_names)

    def _setup_mimic_joints(self, body, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [
            joint.id for joint in self.joints_info if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name] for joint in self.joints_info if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.sim.physics_client.createConstraint(self.sim._bodies_idx[body], self.mimic_parent_id,
                                                         self.sim._bodies_idx[body], joint_id,
                                                         jointType=self.sim.physics_client.JOINT_GEAR,
                                                         jointAxis=[0, 1, 0],
                                                         parentFramePosition=[
                                                             0, 0, 0],
                                                         childFramePosition=[0, 0, 0])
            # Note: the mysterious `erp` is of EXTREME importance
            self.sim.physics_client.changeConstraint(
                c, gearRatio=-multiplier, maxForce=100, erp=1)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            control_actions = self.ee_displacement_to_target_arm_angles(
                ee_displacement)
        elif self.control_type == "position":
            arm_joint_ctrl = action[:7]
            control_actions = self.arm_joint_ctrl_to_target_arm_angles(
                arm_joint_ctrl)
        elif self.control_type == "torque":
            arm_joint_ctrl = action[:7]
            control_actions = self.arm_joint_ctrl_to_torques(
                arm_joint_ctrl)
        else:
            raise NotImplementedError

        # arm control
        # TODO: test
        # inv_kine = self.inverse_kinematics(
        #     link=self.ee_link, position=np.array([0.1, 0.2, 0.5]), orientation=np.array([
        #         0.0, 0.0, 0.0, 1.0])
        # )

        inv_kine = self.sim.physics_client.calculateInverseKinematics(
            self.sim._bodies_idx[self.body_name], self.ee_link, np.array([0.1, 0.2, 0.5]))

        print('debug')
        # print(inv_kine[:7])
        # # print(self.arm_lower_limits)
        # # print(self.arm_upper_limits)
        # print(self.joint_forces)
        # print(self.joint_indices)
        # print(self.body_name)
        for i in self.joint_indices:
            self.arm_torque[i] = self.sim.physics_client.getJointState(
                self.sim._bodies_idx[self.body_name], i)[3]
        # print(self.sim.physics_client.getJointStates(
        #     self.sim._bodies_idx[self.body_name], self.joint_indices))
        print(self.arm_torque)

        # print(self.get_ee_position())
        self.control_joints(control_actions=inv_kine[:7],
                            Kp=self.Kp, Kd=self.Kd)

        # self.control_joints(control_actions=control_actions,
        #                     Kp=self.Kp, Kd=self.Kd)
        # TODO
        if self.block_gripper:
            self.close_gripper()
        else:
            self.move_gripper(action=action[-1])

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * \
            0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([
                                                                                 1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array(
            [self.get_joint_angle(joint=i) for i in range(self.arm_num_dof)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def arm_joint_ctrl_to_torques(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Torques of the 7 arm joints.
        """
        torques = self.arm_torque_limits * self.action_scale * arm_joint_ctrl
        return torques

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate(
                (ee_position, ee_velocity, [fingers_width]))
        else:
            observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        gripper_angle = self.sim.get_joint_angle(
            self.body_name, self.gripper_index)
        return gripper_angle

    def reset_gripper(self):
        self.open_gripper()

    def open_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def move_gripper(self, action):
        # action to gripper angle
        gripper_upper_limit = self.joints_info[self.gripper_index].upperLimit
        gripper_lower_limit = self.joints_info[self.gripper_index].lowerLimit
        gripper_std = (gripper_upper_limit - gripper_lower_limit) / 2.
        gripper_mean = (gripper_upper_limit + gripper_lower_limit) / 2.
        # open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        open_angle = gripper_mean + gripper_std * action

        # Control the mimic gripper joint(s)
        self.sim.physics_client.setJointMotorControl2(self.sim._bodies_idx[self.body_name], self.gripper_index, self.sim.physics_client.POSITION_CONTROL, targetPosition=open_angle,
                                                      force=self.joints_info[self.gripper_index].maxForce, maxVelocity=self.joints_info[self.gripper_index].maxVelocity)
