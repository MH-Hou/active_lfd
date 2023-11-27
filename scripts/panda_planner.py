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

from six.moves import input

def create_collision_object(id, dimensions, pose, frame_id="panda_link0"):
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

def build_the_scene(scene):
    # add the desk
    desk = create_collision_object(id='desk',
                                   dimensions=[2.3, 1.0, 0.01],
                                   pose=[1.0, 0.0, -0.005])
    scene.add_object(desk)

    # add the obstacle object on the desk
    obstacle = create_collision_object(id='obstacle',
                                       dimensions=[0.3, 0.05, 0.2],
                                       pose=[0.3, 0.0, 0.1])
    scene.add_object(obstacle)

    # add the object to pick and place
    target = create_collision_object(id='target',
                                     dimensions=[0.05, 0.05, 0.05],
                                     pose=[0.3, 0.3, 0.025])
    scene.add_object(target)

def reach_pose(move_group, pose, tolerance=0.001):
    move_group.set_pose_target(pose)
    move_group.set_goal_position_tolerance(tolerance)

    # return move_group.go(wait=True)

    success, plan, _, _ = move_group.plan()

    if success:
        # print('[Reach pose]: planned trajectory is:')
        # print(plan)
        move_group.execute(plan, wait=True)
    else:
        print("[Reach pose]: Unsuccessful plan")

    return plan


def reach_pose_linear(move_group, pose):
    waypoints = []

    # add the starting point
    start_pose = move_group.get_current_pose().pose
    waypoints.append(start_pose)

    # add the end point
    waypoints.append(pose)

    (plan, fraction) = move_group.compute_cartesian_path(waypoints=waypoints,
                                                         eef_step=0.01,
                                                         jump_threshold=0.00,
                                                         avoid_collisions = True,
                                                         path_constraints = None)
    # print("[Reach pose linear]: planned trajectory is:")
    # print(plan)

    move_group.execute(plan, wait=True)

    return plan


def go_home_pose(move_group):
    move_group.set_named_target('ready')
    return move_group.go(wait=True)

def open_gripper(gripper_joints):
    res_1 = gripper_joints[0].move(0.04, True)
    res_2 = gripper_joints[1].move(0.04, True)
    success = res_1 and res_2

    return success

def close_gripper(gripper_joints):
    res_1 = gripper_joints[0].move(0.02, True)
    res_2 = gripper_joints[1].move(0.02, True)
    success = res_1 and res_2

    return success

def attach_target_object(scene, robot, move_group):
    eef_link = move_group.get_end_effector_link()
    grasping_group = "panda_hand"
    touch_links = robot.get_link_names(group=grasping_group)
    scene.attach_box(eef_link, 'target', touch_links=touch_links)

def detach_target_object(scene, move_group):
    eef_link = move_group.get_end_effector_link()
    scene.remove_attached_object(eef_link, name='target')

def append_planned_trajectory(plan, planned_trajectory, gripper_state):
    pos_traj = planned_trajectory['pos_traj']
    vel_traj = planned_trajectory['vel_traj']

    joint_traj = plan.joint_trajectory# in the form of trajectory_msgs/JointTrajectory
    joint_traj_points = joint_traj.points # a list of trajectory_msgs/JointTrajectoryPoint
    for point in joint_traj_points:
        joint_positions = list(point.positions) # a 1d list of joint angles
        joint_velocities = list(point.velocities) # a 1d list of joint angular velocities

        if gripper_state == 'open':
            joint_positions.extend([0.04, 0.04])
            joint_velocities.extend([0.0, 0.0])
        else:
            joint_positions.extend([0.02, 0.02])
            joint_velocities.extend([0.0, 0.0])

        pos_traj.append(joint_positions)
        vel_traj.append(joint_velocities)

    return planned_trajectory

def append_gripper_trajectory(gripper_state, planned_trajectory):
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

def pick(planned_trajectory, scene, robot, move_group, gripper_joints, pre_grasp_pos, grasp_pos, pre_grasp_ori_euler=[math.pi, 0.0, math.pi/4.0], grasp_ori_euler=[math.pi, 0.0, math.pi/4.0]):
    # define pre-grasp pose
    pre_grasp_pose = Pose()
    pre_grasp_pose.position.x = pre_grasp_pos[0]
    pre_grasp_pose.position.y = pre_grasp_pos[1]
    pre_grasp_pose.position.z = pre_grasp_pos[2]
    orientation = quaternion_from_euler(*pre_grasp_ori_euler)
    pre_grasp_pose.orientation.x = orientation[0]
    pre_grasp_pose.orientation.y = orientation[1]
    pre_grasp_pose.orientation.z = orientation[2]
    pre_grasp_pose.orientation.w = orientation[3]

    # define grasp pose
    grasp_pose = Pose()
    grasp_pose.position.x = grasp_pos[0]
    grasp_pose.position.y = grasp_pos[1]
    grasp_pose.position.z = grasp_pos[2]
    orientation = quaternion_from_euler(*grasp_ori_euler)
    grasp_pose.orientation.x = orientation[0]
    grasp_pose.orientation.y = orientation[1]
    grasp_pose.orientation.z = orientation[2]
    grasp_pose.orientation.w = orientation[3]

    # the robot reaches for the pre-grasp pose and open the gripper
    print('[Pick]: going to pre-grasp ...')
    # reach_pose(move_group, pre_grasp_pose)
    plan = reach_pose_linear(move_group, pre_grasp_pose)
    planned_trajectory = append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='open')

    print('[Pick]: Pre-grasp finished. Going to open the gripper ...')
    open_gripper(gripper_joints=gripper_joints)
    planned_trajectory = append_gripper_trajectory(gripper_state='open', planned_trajectory=planned_trajectory)

    # the robot reaches for the grasp pose and grasp the target object
    print('[Pick]: Gripper is opened. Going to the grasp pose ...')
    # reach_pose(move_group, grasp_pose)
    plan = reach_pose_linear(move_group, grasp_pose)
    planned_trajectory = append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory, gripper_state='open')

    print("[Pick]: Reached the grasp pose. Going to attach the target object...")
    attach_target_object(scene=scene, robot=robot, move_group=move_group)

    print('[Pick]: The target object is attached. Going to close the gripper ...')
    close_gripper(gripper_joints=gripper_joints)
    planned_trajectory = append_gripper_trajectory(gripper_state='close', planned_trajectory=planned_trajectory)

    # print('[Pick]: Gripper is closed. Going to attach the target object ...')
    # move_group.attach_object('target')

    print('[Pick]: All finished')

    return planned_trajectory

def place(planned_trajectory, scene, move_group, gripper_joints, place_pos, post_place_pos, place_ori_euler=[math.pi, 0.0, math.pi/4.0], post_place_ori_euler=[math.pi, 0.0, math.pi/4.0]):
    # define place pose
    place_pose = Pose()
    place_pose.position.x = place_pos[0]
    place_pose.position.y = place_pos[1]
    place_pose.position.z = place_pos[2]
    orientation = quaternion_from_euler(*place_ori_euler)
    place_pose.orientation.x = orientation[0]
    place_pose.orientation.y = orientation[1]
    place_pose.orientation.z = orientation[2]
    place_pose.orientation.w = orientation[3]

    # define post-place pose
    post_place_pose = Pose()
    post_place_pose.position.x = post_place_pos[0]
    post_place_pose.position.y = post_place_pos[1]
    post_place_pose.position.z = post_place_pos[2]
    orientation = quaternion_from_euler(*post_place_ori_euler)
    post_place_pose.orientation.x = orientation[0]
    post_place_pose.orientation.y = orientation[1]
    post_place_pose.orientation.z = orientation[2]
    post_place_pose.orientation.w = orientation[3]

    # the robot reaches for the place pose and open the gripper
    print('[Place]: Going to reach for the place pose...')
    plan = reach_pose(move_group, place_pose)
    planned_trajectory = append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory,
                                                   gripper_state='close')

    print('[Place]: Reached the place pose. Going to open the gripper...')
    open_gripper(gripper_joints=gripper_joints)
    planned_trajectory = append_gripper_trajectory(gripper_state='open', planned_trajectory=planned_trajectory)

    # arm.detach_object('target')
    print('[Place]: The gripper was opened. Going to detach the target object...')
    detach_target_object(scene=scene, move_group=move_group)

    # the robot reaches for the post-place pose
    print('[Place]: The target object was detached. Going to the post-place pose...')
    # reach_pose(move_group, post_place_pose)
    plan = reach_pose_linear(move_group, post_place_pose)
    planned_trajectory = append_planned_trajectory(plan=plan, planned_trajectory=planned_trajectory,
                                                   gripper_state='open')

    print('[Place]: All finished')

    return planned_trajectory

def clean_the_scene(scene):
    scene.remove_world_object('desk')
    scene.remove_world_object('obstacle')
    scene.remove_world_object('target')


def main():
    # initialize rosnode and moveit commander
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_planner", anonymous=True)

    # preparations for moveit
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    gripper_joints = [robot.get_joint("panda_finger_joint1"),
                      robot.get_joint("panda_finger_joint2")]
    move_group.set_num_planning_attempts(45)
    planned_trajectory = {'pos_traj':[], 'vel_traj':[]}

    # build the scene with a desk, obstacles, and an object to grasp and place
    build_the_scene(scene=scene)

    # input(
    #     "============ Press `Enter` to grasp ..."
    # )
    print("Going to grasp...")

    # implement the pick, including pre-grasp, grasp, and post-grasp
    pre_grasp_pos = [0.3, 0.3, 0.2]
    grasp_pos = [0.3, 0.3, 0.15]
    planned_trajectory = pick(planned_trajectory=planned_trajectory, scene=scene, robot=robot, move_group=move_group,
                              gripper_joints=gripper_joints,
                              pre_grasp_pos=pre_grasp_pos, grasp_pos=grasp_pos)



    # input(
    #     "============ Press `Enter` to place ..."
    # )
    print('Going to place...')

    # implement the place, including pre-place, place, and post-place
    place_pos = [0.3, -0.3, 0.15]
    post_place_pos = [0.3, -0.3, 0.2]
    place(planned_trajectory=planned_trajectory, scene=scene, move_group=move_group,
          gripper_joints=gripper_joints,
          place_pos=place_pos, post_place_pos=post_place_pos)

    # input(
    #     "============ Press `Enter` to go home pose ..."
    # )
    print("Going to go home pose...")

    go_home_pose(move_group=move_group)

    # input(
    #     "============ Press `Enter` to clean the scene and quit ..."
    # )
    print("Going to clean the scene and quit...")
    clean_the_scene(scene=scene)

    np.savetxt("pos_traj.csv", planned_trajectory['pos_traj'], delimiter=" ")
    np.savetxt("vel_traj.csv", planned_trajectory['vel_traj'], delimiter=" ")


if __name__ == "__main__":
    main()


