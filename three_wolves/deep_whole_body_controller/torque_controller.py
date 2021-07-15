import gym
import numpy as np

from three_wolves.deep_whole_body_controller import base_joint_controller
# from trifinger_simulation import trifingerpro_limits
# from three_wolves.deep_whole_body_controller import qp_torque_optimizer
# from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pinocchio_utils
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pc_reward

class QPTorqueController(base_joint_controller.BaseJointController):

    def __init__(self, kinematics, observer):
        self.kinematics = kinematics
        self.observer = observer
        # min max force (0.094 * 9.8)/0.45
        self.robot_action_space = gym.spaces.Box(
            low=np.full(3, -2.047, dtype=np.float32),
            high=np.full(3, 2.047, dtype=np.float32),
        )
        self.t = 0
        self.tg = None

    def reset(self):
        pass

    def update(self):
        desired_speed = 0.2
        obj_goal_dist = reward_utils.ComputeDist(self.observer.dt['object_position'],
                                                 self.observer.dt['goal_position'])
        total_time = obj_goal_dist / desired_speed
        reach_time = int(total_time / 0.1) + 1
        self.t = 0
        self.tg = trajectory.get_acc_planner(init_pos=self.observer.dt['object_position'],
                                             tar_pos=self.observer.dt['goal_position'],
                                             start_time=0,
                                             reach_time=reach_time)

    def get_action(self, policy_action):
        f1 = policy_action
        contact_forces = self.compute_contact_forces(f1, self.tg(self.t)[2])
        motor_torque_list = []
        for i, contact_force in enumerate(contact_forces):
            jv = self.kinematics.compute_jacobian(i, self.observer.dt['joint_position'])[:3, i * 3:i * 3 + 3]
            motor_torque = np.matmul(contact_force, jv)
            motor_torque_list.append(motor_torque)
        self.t += 1
        return np.hstack(motor_torque_list)

    def compute_contact_forces(self, f1, a, robot_mass=0.094):
        F = a * robot_mass + robot_mass * 9.8
        r = np.array([np.subtract(self.observer.dt['tip_0_position'], self.observer.dt['object_position']),
                      np.subtract(self.observer.dt['tip_1_position'], self.observer.dt['object_position']),
                      np.subtract(self.observer.dt['tip_2_position'], self.observer.dt['object_position'])])
        # ZeroDivisionError
        _denominator = (r[2] - r[1])
        _denominator[abs(_denominator) < 0.001] = np.inf
        f2 = ((r[0] - r[1]) / _denominator) * f1 + (r[2] / _denominator) * F
        f3 = F - f1 - f2
        return np.array([f1, f2, f3])

    def compute_reward(self, model_name):
        tar_arm_pos = self.tg(self.t)[0]
        tg_reward = pc_reward.TrajectoryFollowing(self.observer.dt, tar_arm_pos)
        grasp_reward = pc_reward.GraspStability(self.observer.dt)
        orn_reward = pc_reward.OrientationStability(self.observer.search('object_rpy'))
        total_reward = tg_reward * 10 + grasp_reward + orn_reward
        return total_reward

    # def get_qp_action(self):
    #     robot_info = {
    #         'body_mass': 0.094,
    #         'body_inertia': np.array([0.00006619, 0, 0,
    #                                   0, 0.00006619, 0,
    #                                   0, 0, 0.00006619]),
    #         'tip_position': np.array([np.subtract(self.observer.dt['tip_0_position'], self.observer.dt['object_position']),
    #                                   np.subtract(self.observer.dt['tip_1_position'], self.observer.dt['object_position']),
    #                                   np.subtract(self.observer.dt['tip_2_position'], self.observer.dt['object_position'])])
    #     }
    #
    #     contacts = [f > 0.1 for f in self.observer.dt['tip_force']]
    #     desired_acc = np.hstack([self.tg(self.t), [0]*3])
    #     contact_forces = qp_torque_optimizer.compute_contact_force(robot_info=robot_info,
    #                                                                desired_acc=desired_acc,
    #                                                                contacts=contacts)
    #     print(contact_forces)
    #     finger_id = [0, 1, 2]
    #     motor_torque_list = []
    #     for i, contact_force in enumerate(contact_forces):
    #         jv = self.kinematics.compute_jacobian(i, self.observer.dt['joint_position'])[:3, i*3:i*3+3]
    #         motor_torque = np.matmul(contact_force, jv)
    #         motor_torque_list.append(motor_torque)
    #
    #     self.t += 1
    #
    #     return np.hstack(motor_torque_list)
