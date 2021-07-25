import numpy as np
import gym
from three_wolves.deep_whole_body_controller import position_controller, torque_controller, contact_planner
from three_wolves.deep_whole_body_controller.base_joint_controller import Control_Phase
from three_wolves.deep_whole_body_controller.utility import pinocchio_utils, trajectory, reward_utils, pc_reward

class DeepWBC:
    def __init__(
            self, observer, step_size, robot_type
    ):
        self.kinematics = pinocchio_utils.Kinematics(robot_type)
        self.contact_planner = contact_planner.ContactPlanner()
        self.torque_controller = torque_controller.QPTorqueController(self.kinematics, observer, step_size)
        self.observer = observer
        self.step_size = step_size

        # runtime property
        self._phase = Control_Phase.TORQUE
        self._last_goal = None
        self.apply_action = None

    def get_action_space(self):
        return self.contact_planner.action_space

    def reset(self, apply_action):
        self.apply_action = apply_action
        self.torque_controller.reset()
        self.drop_times = 0

    def update(self, policy_action):
        # if self._last_goal is None or all(self._last_goal != self.observer.dt['goal_position']):
        self._last_goal = self.observer.dt['goal_position']
        contact_face_ids, contact_points = self.contact_planner.compute_contact_points(policy_action)
        self.torque_controller.update(contact_points, contact_face_ids)

    def step(self, policy_action):
        self.update(policy_action)
        self.torque_controller.tips_reach(self.apply_action)
        # self._last_goal = None
        self.reward = 0
        while not self.Dropped():
            if list(self._last_goal) != list(self.observer.dt['goal_position']):
                self.update(policy_action)
            cur_phase_action = self.get_action()
            self.apply_action(cur_phase_action)
            self.reward += self.torque_controller.get_reward()*0.001*self.step_size
        self.drop_times += 1

    def get_action(self):
        action = self.torque_controller.get_action()
        return action

    def get_reward(self):
        return self.reward

    def get_done(self):
        return self.drop_times >= 2

    def Dropped(self):
        tip_force = self.observer.dt['tip_force']
        cube_pos = np.array(self.observer.dt['object_position'])
        tri_distance = [reward_utils.ComputeDist(self.observer.dt['tip_0_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_1_position'], cube_pos),
                        reward_utils.ComputeDist(self.observer.dt['tip_2_position'], cube_pos)]
        is_dropped = list(tip_force).count(0.) > 1 or any(np.array(tri_distance) > 0.7)
        return is_dropped
