import sys
import time

import gym
import numpy as np
from trifinger_simulation import trifingerpro_limits
from three_wolves.deep_whole_body_controller import qp_torque_optimizer
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pinocchio_utils
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pc_reward
from three_wolves.envs.utilities.env_utils import *

CUBE_MASS = 0.094
CUBE_INERTIA = np.array([[0.00006619, 0, 0],
                         [0, 0.00006619, 0],
                         [0, 0, 0.00006619]])


class QPTorqueController:

    def __init__(self, kinematics, observer, step_size):
        self.step_size = step_size
        self.kinematics = kinematics
        self.observer = observer
        self.t = 0
        self.tg = None
        self.desired_contact_points = None
        self.contact_face_ids = None
        self.desired_speed = 0.1

    def reset(self):
        pass

    def reset_tg(self, init_pos, tar_pos):
        obj_goal_dist = reward_utils.ComputeDist(init_pos, tar_pos)
        total_time = obj_goal_dist / self.desired_speed
        self.t = 0
        self.tg = trajectory.get_acc_planner(init_pos=init_pos,
                                             tar_pos=tar_pos,
                                             start_time=0,
                                             reach_time=total_time)

    def update(self, contact_points, contact_face_ids):
        self.contact_face_ids = contact_face_ids
        self.desired_contact_points = contact_points
        self.reset_tg(self.observer.dt['object_position'], self.observer.dt['goal_position'])

    def get_action(self):
        # cube_pos = np.array(self.observer.dt['object_position'])
        # cube_rpy = np.array(self.observer.dt['object_rpy'])
        # real_contact_points = np.array([
        #     np.subtract(self.observer.dt[f'tip_{i}_position'], cube_pos) for i in range(3)
        # ])
        # CUBE_rotate_contact_pos = np.array([trajectory.Rotate([0, 0, self._get_clip_yaw()], c)
        #                                     for c in real_contact_points])
        # WRD_contact_pos = np.add(cube_pos, CUBE_rotate_contact_pos)

        # desired_acc = np.hstack([self.tg(self.t)[2], [0] * 3])
        # contact_forces = qp_torque_optimizer.compute_contact_force(
        #     robot_mass=CUBE_MASS,
        #     robot_inertia=CUBE_INERTIA,
        #     contact_face_ids=self.contact_face_ids,
        #     contact_position=real_contact_points,
        #     desired_acc=desired_acc,
        # )
        #
        # rotate_contact_forces = trajectory.Rotate([0, 0, self._get_clip_yaw()],
        #                                           contact_forces).reshape(3, 3)
        desired_position = self.tg(self.t)[0] + self.desired_contact_points
        desired_joint_position, _ = self.kinematics.inverse_kinematics(desired_position,
                                                                       self.observer.dt['joint_position'])
        # all_pd_torque = self.compute_pd_control_torques(joint_positions=desired_joint_position)
        #
        # motor_torques = self.kinematics.map_contact_force_to_joint_torque(rotate_contact_forces,
        #                                                                   self.observer.dt['joint_position']
        # all_torques = all_pd_torque + motor_torques
        self.t += 0.001 * self.step_size
        return desired_joint_position

    def compute_pd_control_torques(self, joint_positions):
        kp = np.array([15.0, 15.0, 9.0] * 3)
        kd = np.array([0.5, 1.0, 0.5] * 3)
        current_position = self.observer.dt['joint_position']
        current_velocity = self.observer.dt['joint_velocity']
        position_error = joint_positions - current_position

        position_feedback = np.asarray(kp) * position_error
        velocity_feedback = np.asarray(kd) * current_velocity

        joint_torques = position_feedback - velocity_feedback

        # set nan entries to zero (nans occur on joints for which the target
        # position was set to nan)
        joint_torques[np.isnan(joint_torques)] = 0.0

        return joint_torques

    def _get_clip_yaw(self, c=np.pi / 4):
        # transfer to -pi/2 to pi/2
        theta = self.observer.dt['object_rpy'][2]
        if theta < -c or theta > c:
            n = (theta + c) // (2 * c)
            beta = theta - np.pi * n / 2
        else:
            beta = theta

        return beta

    def tips_reach(self, apply_action, total_time=4.0):
        s = 3
        pre_finger_scale = np.array([[1, s, 1],
                                     [s, 1, 1],
                                     [1, s, 1],
                                     [s, 1, 1]])[self.contact_face_ids]
        P0 = self.desired_contact_points * pre_finger_scale + [0, 0, 0.05]
        P1 = self.desired_contact_points * pre_finger_scale
        P2 = self.desired_contact_points

        # pre_contact_pos = self.desired_contact_points * pre_finger_scale + [0, 0, 0.05]
        # key_points = [pre_contact_pos, self.desired_contact_points]
        # key_interval = [total_time * 0.4, total_time * 0.6]
        key_points = [P0, P1, P2]
        key_interval = np.array([0.3, 0.3, 0.4])*total_time
        for points, interval in zip(key_points, key_interval):
            _clip_yaw = self._get_clip_yaw()
            rotated_key_pos = np.array([trajectory.Rotate([0, 0, _clip_yaw], points[i]) for i in range(3)])
            tar_tip_pos = self.observer.dt['object_position'] + rotated_key_pos
            self._to_point(apply_action, tar_tip_pos, interval)

    def _to_point(self, apply_action, tar_tip_pos, total_time):
        init_tip_pos = np.hstack([self.observer.dt[f'tip_{i}_position'] for i in range(3)])
        # tar_tip_pos = self.observer.dt['object_position'] + self.desired_contact_points
        # dist = reward_utils.ComputeDist(init_tip_pos[:3], self.desired_contact_points[:3])
        # task_time = dist / self.desired_speed
        # time.sleep(total_time*0.2)
        tg = trajectory.get_path_planner(init_pos=init_tip_pos,
                                         tar_pos=tar_tip_pos.flatten(),
                                         start_time=0,
                                         reach_time=total_time * 0.8)
        t = 0
        while t < total_time:
            tg_tip_pos = tg(t)
            arm_joi_pos = self.observer.dt['joint_position']
            to_goal_joints, _error = self.kinematics.inverse_kinematics(tg_tip_pos.reshape(3, 3),
                                                                        arm_joi_pos)
            # to_goal_torques = self.compute_pd_control_torques(to_goal_joints)
            obs_dict, eval_score = apply_action(to_goal_joints)
            t += 0.001 * self.step_size
        self.observer.update(obs_dict)

    def get_reward(self):
        goal_reward = pc_reward.TrajectoryFollowing(self.observer.dt, self.tg(self.t)[0])
        return goal_reward

    # def compute_contact_forces(self, f1, a, robot_mass=0.094):
    #     F = a * robot_mass + np.array([0, 0, robot_mass * 9.8])
    #     r = np.array([np.subtract(self.observer.dt['tip_0_position'], self.observer.dt['object_position']),
    #                   np.subtract(self.observer.dt['tip_1_position'], self.observer.dt['object_position']),
    #                   np.subtract(self.observer.dt['tip_2_position'], self.observer.dt['object_position'])])
    #     # ZeroDivisionError
    #     _denominator = (r[2] - r[1])
    #     _denominator[_denominator == 0] = np.inf
    #     f2 = ((r[0] - r[1]) / _denominator) * f1 + (r[2] / _denominator) * F
    #     f3 = F - f1 - f2
    #     return np.array([f1, f2, f3])
    #
    # def compute_reward(self, model_name):
    #     tar_arm_pos = self.tg(self.t)[0]
    #     tg_reward = pc_reward.TrajectoryFollowing(self.observer.dt, tar_arm_pos)
    #     grasp_reward = pc_reward.GraspStability(self.observer.dt)
    #     orn_reward = pc_reward.OrientationStability(self.observer.search('object_rpy'))
    #     total_reward = tg_reward * 10 + grasp_reward + orn_reward
    #     return total_reward
