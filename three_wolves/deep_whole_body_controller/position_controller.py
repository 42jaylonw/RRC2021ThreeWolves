import gym
import numpy as np

from three_wolves.deep_whole_body_controller import base_joint_controller
from trifinger_simulation import trifingerpro_limits
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pc_reward
# delete
from three_wolves.envs.utilities.env_utils import tag, clean

class DRLPositionController(base_joint_controller.BaseJointController):

    def __init__(self, kinematics, observer):
        self.kinematics = kinematics
        self.observer = observer
        action_space = (trifingerpro_limits.robot_position.high
                        - trifingerpro_limits.robot_position.low) / 20
        self.robot_action_space = gym.spaces.Box(
            low=-action_space,
            high=action_space,
        )

        # delete me
        self.t = 0
        self.tg = None

    def reset(self):
        pass

    def update(self):
        desired_speed = 0.2
        obj_goal_dist = reward_utils.ComputeDist(self.observer.dt['object_position'],
                                                 self.observer.dt['goal_position'])
        total_time = obj_goal_dist / desired_speed
        self.t = 0
        self.tg = trajectory.get_path_planner(init_pos=self.observer.dt['object_position'],
                                              tar_pos=self.observer.dt['goal_position'],
                                              start_time=0,
                                              reach_time=int(total_time/0.1))

    def get_action(self, policy_action):
        position_action = policy_action + self.observer.dt['joint_position']
        self.t += 1
        return position_action

    def get_joints_to_goal(self):
        arm_joi_pos = self.observer.dt['joint_position']
        cube_pos = self.observer.dt['object_position']
        tar_arm_pos = [
            np.add(cube_pos, [0,  0.03, 0]),  # arm_0 x+y+
            np.add(cube_pos, [0, -0.03, 0]),  # arm_1 x+y-
            np.add(cube_pos, [-0.03, 0, 0])   # arm_2 x-y+
        ]

        to_goal_joints, _error = self.kinematics.inverse_kinematics(tar_arm_pos,
                                                                    arm_joi_pos)
        return to_goal_joints

    def compute_reward(self, model_name):
        # basic reward
        grasp_reward = pc_reward.GraspStability(self.observer.dt)
        slippery_reward = pc_reward.TipSlippery(self.observer)
        orn_reward = pc_reward.OrientationStability(self.observer.search('object_rpy'))

        # goal reaching reward
        if model_name == 'tg':
            tar_arm_pos = self.tg(self.t)
            goal_reward = pc_reward.TrajectoryFollowing(self.observer.dt, tar_arm_pos) * 10
        elif model_name == 'vel':
            goal_reward = pc_reward.GoalDistReward(self.observer)
        elif model_name == 'tv':
            tar_arm_pos = self.tg(self.t)
            vel_reward = pc_reward.GoalDistReward(self.observer) * 5
            tg_reward = pc_reward.TrajectoryFollowing(self.observer.dt, tar_arm_pos) * 5
            goal_reward = tg_reward + vel_reward
        elif model_name == 'tg_closer':
            tar_arm_pos = self.tg(self.t)
            goal_reward = pc_reward.TrajectoryFollowing(self.observer.dt, tar_arm_pos, wei=-1000) * 10
            orn_reward *= 0.3
            slippery_reward *= 2
        else:
            raise NotImplemented('not support reward type')
        total_reward = goal_reward + grasp_reward + slippery_reward + orn_reward
        return total_reward

    def IsTouch(self):
        tip_force = self.observer.dt['tip_force']
        return all(tip_force > 0)

    def IsNear(self):
        cube_pos = np.array(self.observer.dt['object_position'])
        tri_distance = [reward_utils.ComputeDist(self.observer.dt['tip_0_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_1_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_2_position'], cube_pos)]
        return all(np.array(tri_distance) < 0.05)

    def IsFar(self):
        cube_pos = np.array(self.observer.dt['object_position'])
        tri_distance = [reward_utils.ComputeDist(self.observer.dt['tip_0_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_1_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_2_position'], cube_pos)]
        return any(np.array(tri_distance) > 0.1)
