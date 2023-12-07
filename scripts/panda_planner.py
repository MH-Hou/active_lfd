#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from tf.transformations import quaternion_from_euler
import math
import numpy as np
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from six.moves import input


class PandaPlanner():
    def __init__(self, target_object_dim=None):
        # initialize rosnode and moveit commander
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("panda_planner", anonymous=True)

        # preparations for moveit
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        self.move_group.set_num_planning_attempts(45)
        self.gripper_joints = [self.robot.get_joint("panda_finger_joint1"),
                               self.robot.get_joint("panda_finger_joint2")]

        if target_object_dim is None:
            target_object_dim = [0.05, 0.05, 0.05]
        self.target_object_dim = target_object_dim

        self.build_the_scene()

    def build_the_scene(self):
        # add the desk
        desk = self.create_collision_object(id='desk',
                                       dimensions=[2.3, 1.0, 0.01],
                                       pose=[1.0, 0.0, -0.005])
        self.scene.add_object(desk)

        # add the obstacle object on the desk
        obstacle = self.create_collision_object(id='obstacle',
                                           dimensions=[0.3, 0.05, 0.2],
                                           pose=[0.3, 0.0, 0.1])
        self.scene.add_object(obstacle)

        # add the object to pick and place
        target = self.create_collision_object(id='target',
                                         dimensions=[0.05, 0.05, 0.05],
                                         pose=[0.3, 0.3, 0.025])
        self.scene.add_object(target)

    def clean_the_scene(self):
        self.scene.remove_world_object('desk')
        self.scene.remove_world_object('obstacle')
        self.scene.remove_world_object('target')

    def create_collision_object(self, id, dimensions, pose, frame_id="panda_link0"):
        object = CollisionObject()
        object.id = id
        object.header.frame_id = frame_id

        solid = SolidPrimitive()
        solid.type = solid.BOX
        solid.dimensions = dimensions
        object.primitives = [solid]

        object_pose = Pose()
        object_pose.position.x = pose[0]
        object_pose.position.y = pose[1]
        object_pose.position.z = pose[2]

        object.primitive_poses = [object_pose]
        object.operation = object.ADD

        return object

    def attach_target_object(self):
        eef_link = self.move_group.get_end_effector_link()
        grasping_group = "panda_hand"
        touch_links = self.robot.get_link_names(group=grasping_group)
        self.scene.attach_box(eef_link, 'target', touch_links=touch_links)

    def detach_target_object(self):
        eef_link = self.move_group.get_end_effector_link()
        self.scene.remove_attached_object(eef_link, name='target')

    def plan_pick_and_place(self, target_pick_pos, target_place_pos):
        planned_trajectory = {'pos_traj': [], 'vel_traj': []}

        planned_trajectory = self.plan_pick(planned_trajectory=planned_trajectory, target_pick_pos=target_pick_pos)

        planned_trajectory = self.plan_place(planned_trajectory=planned_trajectory, target_place_pos=target_place_pos)

        return planned_trajectory

    def plan_pick(self, planned_trajectory, target_pick_pos):
        # define pre-grasp pose
        pre_grasp_pose = Pose()
        pre_grasp_pose.position.x = target_pick_pos[0]
        pre_grasp_pose.position.y = target_pick_pos[1]
        pre_grasp_pose.position.z = self.target_object_dim[2] + 0.15
        pre_grasp_ori_euler = [math.pi, 0.0, math.pi / 4.0]
        orientation = quaternion_from_euler(*pre_grasp_ori_euler)
        pre_grasp_pose.orientation.x = orientation[0]
        pre_grasp_pose.orientation.y = orientation[1]
        pre_grasp_pose.orientation.z = orientation[2]
        pre_grasp_pose.orientation.w = orientation[3]

        # define grasp pose
        grasp_pose = Pose()
        grasp_pose.position.x = target_pick_pos[0]
        grasp_pose.position.y = target_pick_pos[1]
        grasp_pose.position.z = self.target_object_dim[2] + 0.1
        grasp_ori_euler = [math.pi, 0.0, math.pi / 4.0]
        orientation = quaternion_from_euler(*grasp_ori_euler)
        grasp_pose.orientation.x = orientation[0]
        grasp_pose.orientation.y = orientation[1]
        grasp_pose.orientation.z = orientation[2]
        grasp_pose.orientation.w = orientation[3]

        """ *************************************************** """

        print('[Pick]: Pre-grasp ...')
        start_joint_values = self.move_group.get_current_joint_values() + [0.02, 0.02]
        start_eef_pose = self.move_group.get_current_pose().pose
        plan = self.reach_pose_linear(start_joint_values=start_joint_values, start_eef_pose=start_eef_pose, target_pose=pre_grasp_pose)
        planned_trajectory = self.append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='open')

        """ *************************************************** """

        print('[Pick]: Open the gripper ...')
        planned_trajectory = self.append_gripper_trajectory(gripper_state='open', planned_trajectory=planned_trajectory)

        """ *************************************************** """

        print('[Pick]: Grasp ...')
        start_joint_values = planned_trajectory['pos_traj'][-1]
        start_eef_pose = pre_grasp_pose
        plan = self.reach_pose_linear(start_joint_values=start_joint_values, start_eef_pose=start_eef_pose, target_pose=grasp_pose)
        planned_trajectory = self.append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='open')

        """ *************************************************** """

        print("[Pick]: Attach the target object ...")
        self.attach_target_object()

        """ *************************************************** """

        print('[Pick]: Close the gripper ...')
        planned_trajectory = self.append_gripper_trajectory(gripper_state='close', planned_trajectory=planned_trajectory)

        """ *************************************************** """

        print('[Pick]: All finished')

        return planned_trajectory

    def plan_place(self, planned_trajectory, target_place_pos):
        # define place pose
        place_pose = Pose()
        place_pose.position.x = target_place_pos[0]
        place_pose.position.y = target_place_pos[1]
        place_pose.position.z = self.target_object_dim[2] + 0.1
        place_ori_euler = [math.pi, 0.0, math.pi / 4.0]
        orientation = quaternion_from_euler(*place_ori_euler)
        place_pose.orientation.x = orientation[0]
        place_pose.orientation.y = orientation[1]
        place_pose.orientation.z = orientation[2]
        place_pose.orientation.w = orientation[3]

        # define post-place pose
        post_place_pose = Pose()
        post_place_pose.position.x = target_place_pos[0]
        post_place_pose.position.y = target_place_pos[1]
        post_place_pose.position.z = self.target_object_dim[2] + 0.15
        post_place_ori_euler = [math.pi, 0.0, math.pi / 4.0]
        orientation = quaternion_from_euler(*post_place_ori_euler)
        post_place_pose.orientation.x = orientation[0]
        post_place_pose.orientation.y = orientation[1]
        post_place_pose.orientation.z = orientation[2]
        post_place_pose.orientation.w = orientation[3]

        """ *************************************************** """

        print('[Place]: Place ...')
        start_joint_values = planned_trajectory['pos_traj'][-1]
        plan = self.reach_pose(start_joint_values=start_joint_values, target_pose=place_pose, whether_to_attach=True)
        planned_trajectory = self.append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='close')

        """ *************************************************** """

        print('[Place]: Open the gripper...')
        planned_trajectory = self.append_gripper_trajectory(gripper_state='open', planned_trajectory=planned_trajectory)

        """ *************************************************** """

        print('[Place]: Detach the target object...')
        self.detach_target_object()

        """ *************************************************** """

        print('[Place]: Post-place ...')
        start_joint_values = planned_trajectory['pos_traj'][-1]
        start_eef_pose = place_pose
        plan = self.reach_pose_linear(start_joint_values=start_joint_values, start_eef_pose=start_eef_pose, target_pose=post_place_pose)
        planned_trajectory = self.append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='open')

        """ *************************************************** """

        print('[Place]: All finished')

        return planned_trajectory

    def reach_pose_linear(self, start_joint_values, start_eef_pose, target_pose):
        waypoints = []

        # manually set the starting joint values for the plan
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.robot.get_joint_names(group='panda_arm')[:-1] + self.robot.get_joint_names(group='panda_hand')[1:]
        joint_state.position = start_joint_values
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.move_group.set_start_state(moveit_robot_state)

        # add the starting eef pose
        waypoints.append(start_eef_pose)

        # add the end pose
        waypoints.append(target_pose)

        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints=waypoints,
                                                             eef_step=0.01,
                                                             jump_threshold=0.00,
                                                             avoid_collisions=True,
                                                             path_constraints=None)

        return plan

    def reach_pose(self, start_joint_values, target_pose, whether_to_attach=False, tolerance=0.001):
        # manually set the starting joint values for the plan
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.robot.get_joint_names(group='panda_arm')[:-1] + self.robot.get_joint_names(group='panda_hand')[1:]
        joint_state.position = start_joint_values
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.move_group.set_start_state(moveit_robot_state)

        if whether_to_attach:
            self.attach_target_object()

        # set target eef pose and plan the trajectory
        self.move_group.set_pose_target(target_pose)
        self.move_group.set_goal_position_tolerance(tolerance)

        success, plan, _, _ = self.move_group.plan()

        if success:
            print('[Reach pose]: Found successful plan!')
        else:
            print("[Reach pose]: Unsuccessful plan")

        return plan

    def append_planned_trajectory(self, plan, planned_trajectory, gripper_state):
        pos_traj = planned_trajectory['pos_traj']
        vel_traj = planned_trajectory['vel_traj']

        joint_traj = plan.joint_trajectory  # in the form of trajectory_msgs/JointTrajectory
        joint_traj_points = joint_traj.points  # a list of trajectory_msgs/JointTrajectoryPoint
        for point in joint_traj_points:
            joint_positions = list(point.positions)  # a 1d list of joint angles
            joint_velocities = list(point.velocities)  # a 1d list of joint angular velocities

            if gripper_state == 'open':
                joint_positions.extend([0.04, 0.04])
                joint_velocities.extend([0.0, 0.0])
            else:
                joint_positions.extend([0.02, 0.02])
                joint_velocities.extend([0.0, 0.0])

            pos_traj.append(joint_positions)
            vel_traj.append(joint_velocities)

        return planned_trajectory

    def append_gripper_trajectory(self, gripper_state, planned_trajectory):
        last_joint_positions = planned_trajectory['pos_traj'][-1].copy()
        last_joint_velocities = planned_trajectory['vel_traj'][-1].copy()

        if gripper_state == 'open':
            last_joint_positions[-2] = 0.04
            last_joint_positions[-1] = 0.04
            last_joint_velocities[-2] = 0.0
            last_joint_velocities[-1] = 0.0
            planned_trajectory['pos_traj'].append(last_joint_positions)
            planned_trajectory['vel_traj'].append(last_joint_velocities)
        else:
            last_joint_positions[-2] = 0.02
            last_joint_positions[-1] = 0.02
            last_joint_velocities[-2] = 0.0
            last_joint_velocities[-1] = 0.0
            planned_trajectory['pos_traj'].append(last_joint_positions)
            planned_trajectory['vel_traj'].append(last_joint_velocities)

        return planned_trajectory

    def go_home_pose(self):
        self.move_group.set_named_target('ready')
        return self.move_group.go(wait=True)

    def open_gripper(self):
        res_1 = self.gripper_joints[0].move(0.04, True)
        res_2 = self.gripper_joints[1].move(0.04, True)
        success = res_1 and res_2

        return success

    def close_gripper(self):
        res_1 = self.gripper_joints[0].move(0.02, True)
        res_2 = self.gripper_joints[1].move(0.02, True)
        success = res_1 and res_2

        return success


def main():
    panda_planner = PandaPlanner()
    target_pick_pos = [0.3, 0.3]
    target_place_pos = [0.3, -0.3]

    planned_trajectory_ = panda_planner.plan_pick_and_place(target_pick_pos=target_pick_pos, target_place_pos=target_place_pos)
    planned_trajectory = panda_planner.plan_pick_and_place(target_pick_pos=target_pick_pos, target_place_pos=target_place_pos)

    np.savetxt("/home/oem/catkin_ws/src/active_lfd/pos_traj.csv", planned_trajectory['pos_traj'], delimiter=" ")
    np.savetxt("/home/oem/catkin_ws/src/active_lfd/vel_traj.csv", planned_trajectory['vel_traj'], delimiter=" ")

    panda_planner.clean_the_scene()
    print("Cleaned the scene and quit")



if __name__ == "__main__":
    main()


