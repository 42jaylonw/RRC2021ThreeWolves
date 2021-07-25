import sys

import gym
import numpy as np

from three_wolves.deep_whole_body_controller import base_joint_controller
from trifinger_simulation import trifingerpro_limits
from three_wolves.deep_whole_body_controller import qp_torque_optimizer
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pinocchio_utils
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pc_reward
from three_wolves.envs.utilities.env_utils import *


lib=np.zeros(3)


MinV = -10
MaxV = -10
robot_info = {
            # 'body_mass': 0.094,
            'body_mass': 1000,
            'body_inertia': np.array([0.00006619, 0., 0.,
                                      0., 0.00006619, 0.,
                                      0., 0., 0.00006619]),
            'tip_position': np.array([[0.033, 0., 0.],
                                     [0., -0.033, 0.],
                                     [-0.033, 0., 0.]]),
            'body_orientation': np.array([0., 0., 0.])
        }

# desired_acc = np.hstack([self.tg(self.t)[2], [0]*3])
desired_acc = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
contact_forces = qp_torque_optimizer.compute_contact_force(robot_info=robot_info, desired_acc=desired_acc)
print(contact_forces)
for i in range(3):
    lib[i] = sum(contact_forces[:, i])
print(lib)
#lib = contact_forces
file_name = '5'
np.save(file_name, contact_forces)
#sys.exit()
