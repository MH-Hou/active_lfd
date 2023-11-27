import numpy as np
from gym import spaces
import math

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class PandaRobot(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim,
        block_gripper,
        base_position,
        control_type="ee",
    ):
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 14  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 4
        if self.control_type == 'ee':
            action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        else:
            # the action space is the target joint angles
            action_space = spaces.Box(-2.0 * math.pi, 2.0 * math.pi, shape=(n_action,), dtype=np.float32)


        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        # self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        self.neutral_joint_values = np.array([0.00, -0.785, 0.00, -2.355, 0.00, 1.57, 0.785, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action):
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # if self.control_type == "ee":
        #     ee_displacement = action[:3]
        #     target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
        # else:
        #     target_arm_angles = action[:7]
        #     target_arm_angle_vels = action[9:-1]

        # if self.block_gripper:
        #     target_fingers_width = 0
        # else:
        #     target_fingers_width = action[7]
        #     target_fingers_vel = action[-1]

        target_angles = action[:9]
        target_vels = action[9:]
        self.sim.physics_client.setJointMotorControlArray(
            self.sim._bodies_idx[self.body_name],
            jointIndices=self.joint_indices,
            controlMode=self.sim.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            targetVelocities=target_vels,
            forces=self.joint_forces,
        )

    def ee_displacement_to_target_arm_angles(self, ee_displacement):
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles



    def get_obs(self):
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self):
        self.set_joint_neutral()

    def set_joint_neutral(self):
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self):
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self):
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self):
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)