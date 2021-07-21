import numpy as np
import gym
from three_wolves.deep_whole_body_controller import position_controller, torque_controller
from three_wolves.deep_whole_body_controller.base_joint_controller import BaseJointController, Control_Phase


class DeepWBC(BaseJointController):

    def __init__(
            self, kinematics, observer, args
    ):
        self.position_controller = position_controller.DRLPositionController(kinematics, observer, args.action_type)
        self.torque_controller = torque_controller.QPTorqueController(kinematics, observer)
        self.observer = observer
        self.args = args

        if args.controller_type == 'position':
            self._phase = Control_Phase.POSITION
        elif args.controller_type == 'force':
            self._phase = Control_Phase.TORQUE
        else:
            raise NotImplemented()

        self._goal = None

    def get_action_space(self):
        if self._phase == Control_Phase.TORQUE:
            return self.torque_controller.robot_action_space
        elif self._phase == Control_Phase.POSITION:
            return self.position_controller.robot_action_space
        else:
            raise NotImplemented('not support')

    def reset(self):
        self.position_controller.reset()
        self.torque_controller.reset()

    def update(self):
        if self._goal is None or all(self._goal != self.observer.dt['goal_position']):
            self._goal = self.observer.dt['goal_position']
            self.position_controller.update()
            self.torque_controller.update()

    def _select_action(self, policy_action):
        if self._phase == Control_Phase.TORQUE:
            torque_action = self.torque_controller.get_action(policy_action)
            action = np.array(torque_action, dtype=np.float32)
            return {'position': None, 'torque': action}
        elif self._phase == Control_Phase.POSITION:
            position_action = self.position_controller.get_action(policy_action)
            action = np.array(position_action, dtype=np.float32)
            return {'position': action, 'torque': None}
        else:
            raise NotImplemented('not support other phase now')

    def get_action(self, policy_action):
        # if ?:
        #     self._phase = Control_Phase.TORQUE
        # else:
        #     self._phase = Control_Phase.POSITION
        return self._select_action(policy_action)

    def get_reward(self):
        if self._phase == Control_Phase.TORQUE:
            return self.torque_controller.compute_reward(self.args.model_name)
        elif self._phase == Control_Phase.POSITION:
            return self.position_controller.compute_reward(self.args.model_name)
        else:
            raise NotImplemented()

    def get_done(self):
        if self.args.action_type == 'full':
            return self.position_controller.IsTooFar()
        else:
            return self.position_controller.IsFar()

    def get_control_mode(self):
        return self._phase
