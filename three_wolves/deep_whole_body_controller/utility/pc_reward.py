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
        reward = utils.FVCap(desired_velocity, last_dist_to_goal - dist_to_goal) * 50
    return reward


def TrajectoryFollowing(obs_dict, trajectory_pos, wei=-300):
    cur_pos = obs_dict["object_position"]
    dist = utils.ComputeDist(cur_pos, trajectory_pos)
    return utils.ExpSqr(dist, 0, wei=wei)

def OrientationStability(his_rpy):
    _delta_rpy = np.sum([utils.Delta(his_rpy[:, i]) for i in range(his_rpy.shape[0])])
    reward_rpy = utils.ExpSqr(_delta_rpy, 0, wei=-2)
    return reward_rpy

def GraspStability(obs_dict):
    tip_force = obs_dict['tip_force']
    return .1 if all(tip_force > 0) and IsNear(obs_dict) else -10

def TipSlippery(observer):
    obj_history_positions = observer.search('object_position')
    delta_tri_0 = utils.Delta([utils.ComputeDist(p0, p1) for p0, p1 in
                               zip(observer.search('tip_0_position'), obj_history_positions)])
    delta_tri_1 = utils.Delta([utils.ComputeDist(p0, p1) for p0, p1 in
                               zip(observer.search('tip_1_position'), obj_history_positions)])
    delta_tri_2 = utils.Delta([utils.ComputeDist(p0, p1) for p0, p1 in
                               zip(observer.search('tip_2_position'), obj_history_positions)])
    sum_slippery = np.sum([delta_tri_0, delta_tri_1, delta_tri_2])
    reward_slippery = utils.ExpSqr(sum_slippery, 0, wei=-2000)
    return reward_slippery

def IsNear(obs_dict):
    tip_to_obj_positions = np.array([
        utils.ComputeDist(obs_dict['object_position'], obs_dict[f'tip_{i}_position']) for i in range(3)
    ])
    return all(np.array(tip_to_obj_positions) < 0.055)
