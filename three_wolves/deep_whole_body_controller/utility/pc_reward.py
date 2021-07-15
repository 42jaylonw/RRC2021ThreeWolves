import numpy as np
from three_wolves.deep_whole_body_controller.utility import reward_utils as utils

def GoalDistReward(observer, desired_velocity=0.02):
    dist_to_goal = utils.ComputeDist(observer.dt['object_position'],
                                     observer.dt['goal_position'])
    if dist_to_goal < 0.02:
        # minimize distance
        reward = utils.ExpSqr(dist_to_goal, wei=-1000)*100
    else:
        # minimize toward velocity
        last_dist_to_goal = utils.ComputeDist(observer.search('object_position')[1],
                                              observer.search('goal_position')[1])
        reward = utils.FVCap(desired_velocity, dist_to_goal - last_dist_to_goal)
    return reward * 50


def TrajectoryFollowing(obs_dict, trajectory_pos):
    cur_pos = obs_dict["object_position"]
    dist = utils.ComputeDist(cur_pos, trajectory_pos)
    return utils.ExpSqr(dist, 0, wei=-30)

def OrientationStability(his_rpy):
    _delta_rpy = sum([utils.Delta(his_rpy[:, i]) for i in range(his_rpy.shape[0])])
    reward_rpy = utils.ExpSqr(_delta_rpy, 0, wei=-2)
    return reward_rpy

def GraspStability(obs_dict):
    tip_force = obs_dict['tip_force']
    tip_to_obj_positions = np.array([
        utils.ComputeDist(obs_dict['object_position'], obs_dict[f'tip_{i}_position']) for i in range(3)
    ])
    reward = 0.1 if all(tip_force > 0) and IsNear(obs_dict) else -10
    return reward

def IsNear(obs_dict):
    tip_to_obj_positions = np.array([
        utils.ComputeDist(obs_dict['object_position'], obs_dict[f'tip_{i}_position']) for i in range(3)
    ])
    return all(np.array(tip_to_obj_positions) < 0.055)
