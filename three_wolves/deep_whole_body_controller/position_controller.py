import numpy as np
from three_wolves.deep_whole_body_controller.utility import trajectory, reward_utils, pc_reward

CUBE_MASS = 0.094
CUBE_INERTIA = np.array([[0.00006619, 0, 0],
                         [0, 0.00006619, 0],
                         [0, 0, 0.00006619]])


class PositionController:

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
        desired_position = self.tg(self.t)[0] + self.desired_contact_points
        desired_joint_position, _ = self.kinematics.inverse_kinematics(desired_position,
                                                                       self.observer.dt['joint_position'])
        self.t += 0.001 * self.step_size
        return desired_joint_position

    def _get_clip_yaw(self, c=np.pi / 4):
        # transfer to -pi/4 to pi/4
        theta = self.observer.dt['object_rpy'][2]
        if theta < -c or theta > c:
            n = (theta + c) // (2 * c)
            beta = theta - np.pi * n / 2
        else:
            beta = theta

        return beta

    def tips_reach(self, apply_action, total_time=2.0):
        s = 2
        pre_finger_scale = np.array([[1, s, 1],
                                     [s, 1, 1],
                                     [1, s, 1],
                                     [s, 1, 1]])[self.contact_face_ids]
        P0 = np.array([list(self.observer.dt[f'tip_{i}_position'][:2]) + [0.08] for i in range(3)])
        P1 = self.desired_contact_points * pre_finger_scale + [0, 0, 0.05]
        P2 = self.desired_contact_points * pre_finger_scale
        P3 = self.desired_contact_points

        key_points = [P0, P1, P2, P3]
        key_interval = np.array([0.3, 0.2, 0.2, 0.3])*total_time
        for points, interval in zip(key_points, key_interval):
            _clip_yaw = self._get_clip_yaw()
            rotated_key_pos = np.array([trajectory.Rotate([0, 0, _clip_yaw], points[i]) for i in range(3)])
            tar_tip_pos = self.observer.dt['object_position'] + rotated_key_pos
            self._to_point(apply_action, tar_tip_pos, interval)

    def _to_point(self, apply_action, tar_tip_pos, total_time):
        init_tip_pos = np.hstack([self.observer.dt[f'tip_{i}_position'] for i in range(3)])
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
            apply_action(to_goal_joints)
            t += 0.001 * self.step_size

    def get_reward(self):
        goal_reward = pc_reward.TrajectoryFollowing(self.observer.dt, self.tg(self.t)[0], wei=-500)
        return goal_reward
