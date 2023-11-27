import numpy as np

from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet

from tasks import PickAndPlaceWithObstacleTask
from robot import PandaRobot

class PickAndPlaceWithObstacleEnv(RobotTaskEnv):
    def __init__(self, render=False, reward_type="sparse", control_type="joints"):
        sim = PyBullet(render=render, background_color=np.array([150, 222, 246]))
        robot = PandaRobot(sim, block_gripper=False, base_position=np.array([0.0, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlaceWithObstacleTask(sim, reward_type=reward_type)

        super().__init__(robot, task)



if __name__ == '__main__':
    env = PickAndPlaceWithObstacleEnv(render=True)
    oracle_pos_traj = np.genfromtxt('/home/ullrich/catkin_ws/pos_traj.csv', delimiter=' ')
    oracle_pos_traj = np.array(oracle_pos_traj)
    oracle_vel_traj = np.genfromtxt('/home/ullrich/catkin_ws/vel_traj.csv', delimiter=' ')
    oracle_vel_traj = np.array(oracle_vel_traj)

    obs = env.reset()
    done = False
    step = 0
    traj_length = oracle_pos_traj.shape[0]

    while not done:
        # action = env.action_space.sample()
        joint_pos = oracle_pos_traj[step]
        joint_vel = oracle_vel_traj[step]
        action = np.concatenate((joint_pos, joint_vel))
        obs, reward, done, info = env.step(action)
        env.render('human')

        if info['is_success']:
            done = True

        if step < traj_length - 1:
            step += 1

    print("Finished")

    env.close()