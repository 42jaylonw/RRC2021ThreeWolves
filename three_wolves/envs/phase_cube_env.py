import time

import gym
import numpy as np
from trifinger_simulation import TriFingerPlatform, visual_objects
from trifinger_simulation.tasks import move_cube_on_trajectory as task
from three_wolves.deep_whole_body_controller.utility import pinocchio_utils, trajectory
from three_wolves.envs.base_cube_env import ActionType, BaseCubeTrajectoryEnv
from three_wolves.envs.utilities.env_utils import HistoryWrapper, resetCamera
from three_wolves.deep_whole_body_controller.deep_wbc import DeepWBC
from three_wolves.deep_whole_body_controller.base_joint_controller import Control_Phase


class PhaseControlEnv(BaseCubeTrajectoryEnv):
    def __init__(self,
                 goal_trajectory,
                 visualization,
                 args,
                 action_type=ActionType.TORQUE_AND_POSITION,
                 history_num=3):
        super(PhaseControlEnv, self).__init__(
            goal_trajectory=goal_trajectory,
            action_type=action_type,
            step_size=100)

        self.visualization = visualization
        self.kinematics = pinocchio_utils.Kinematics()
        self.observer = HistoryWrapper(history_num)
        self.deep_wbc = DeepWBC(self.kinematics, self.observer, args)
        # create observation space
        _duplicate = lambda x: np.array([x] * history_num).flatten()

        spaces = TriFingerPlatform.spaces
        if self.deep_wbc.get_control_mode() == Control_Phase.TORQUE:
            self.observation_space = gym.spaces.Box(
                low=np.hstack([
                    _duplicate(spaces.robot_position.gym.low),  # joint position
                    _duplicate(spaces.robot_velocity.gym.low),  # joint velocity
                    _duplicate(spaces.robot_torque.gym.low),  # joint torque
                    _duplicate([0] * 3),  # tip force

                    _duplicate(spaces.object_position.gym.low),  # cube position
                    _duplicate([-np.pi] * 3),  # cube orientation
                    _duplicate(spaces.object_position.gym.low),  # goal position
                    _duplicate([-0.3] * 3),  # cube to goal distance

                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 0
                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 1
                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 2
                ]),
                high=np.hstack([
                    _duplicate(spaces.robot_position.gym.high),
                    _duplicate(spaces.robot_velocity.gym.high),
                    _duplicate(spaces.robot_torque.gym.high),
                    _duplicate([1] * 3),

                    _duplicate(spaces.object_position.gym.high),
                    _duplicate([np.pi] * 3),
                    _duplicate(spaces.object_position.gym.high),
                    _duplicate([0.3] * 3),

                    _duplicate(spaces.object_position.gym.high),
                    _duplicate(spaces.object_position.gym.high),
                    _duplicate(spaces.object_position.gym.high),
                ])
            )
        elif self.deep_wbc.get_control_mode() == Control_Phase.POSITION:
            self.observation_space = gym.spaces.Box(
                low=np.hstack([
                    _duplicate(spaces.robot_position.gym.low),  # joint position
                    _duplicate(spaces.robot_velocity.gym.low),  # joint velocity
                    _duplicate(spaces.robot_torque.gym.low),  # joint torque
                    _duplicate([0] * 3),  # tip force

                    _duplicate(spaces.object_position.gym.low),  # cube position
                    _duplicate([-np.pi] * 3),  # cube orientation
                    _duplicate(spaces.object_position.gym.low),  # goal position
                    _duplicate([-0.3] * 3),  # cube to goal distance

                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 0
                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 1
                    _duplicate(spaces.object_position.gym.low),  # tri-finger position 2
                ]),
                high=np.hstack([
                    _duplicate(spaces.robot_position.gym.high),
                    _duplicate(spaces.robot_velocity.gym.high),
                    _duplicate(spaces.robot_torque.gym.high),
                    _duplicate([1] * 3),

                    _duplicate(spaces.object_position.gym.high),
                    _duplicate([np.pi] * 3),
                    _duplicate(spaces.object_position.gym.high),
                    _duplicate([0.3] * 3),

                    _duplicate(spaces.object_position.gym.high),
                    _duplicate(spaces.object_position.gym.high),
                    _duplicate(spaces.object_position.gym.high),
                ])
            )
        else:
            NotImplemented()

        # action
        # self.action_space = self.deep_wbc.torque_controller.robot_action_space
        self.action_space = self.deep_wbc.get_action_space()

    def reset(self):
        """Reset the environment."""
        # hard-reset simulation
        self.goal_marker = None
        del self.platform

        # initialize simulation
        initial_robot_position = (
            TriFingerPlatform.spaces.robot_position.default
        )
        # initialize cube at the centre
        initial_object_pose = task.move_cube.Pose(
            position=task.INITIAL_CUBE_POSITION
        )

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.deep_wbc.reset()

        # get goal trajectory
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=task.move_cube._CUBE_WIDTH,
                position=trajectory[0][1],
                orientation=(0, 0, 0, 1),
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )
            resetCamera()

        self.info = {"time_index": -1, "trajectory": trajectory}
        self.step_count = 0

        # initial step
        robot_action = self._gym_action_to_robot_action(self._initial_action)
        t = self.platform.append_desired_action(robot_action)
        self.info["time_index"] += 1
        self.step_count += 1
        self.tip_force_offset = self.platform.get_robot_observation(0).tip_force
        obs_dict = self._create_observation(0)
        self.observer.reset(obs_dict)
        obs = self.observer.update(obs_dict)
        self.init_control()
        return obs

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose
        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )
        cube_pos = object_observation.position
        cube_orn = object_observation.orientation[:3]
        # compute finger positions
        finger_pos = self.kinematics.forward_kinematics(robot_observation.position)
        if self.deep_wbc.get_control_mode() == Control_Phase.TORQUE:
            observation = {
                "joint_position": robot_observation.position,  # joint position
                "joint_velocity": robot_observation.velocity,  # joint velocity
                "joint_torque": robot_observation.torque,  # joint torque
                "tip_force": np.subtract(robot_observation.tip_force, self.tip_force_offset),  # tip force

                "object_position": cube_pos,  # cube position
                "object_rpy": cube_orn,  # cube orientation
                "goal_position": active_goal,  # goal position
                "object_goal_distance": active_goal - cube_pos,  # cube to goal distance

                "tip_0_position": finger_pos[0],  # tri-finger position 0
                "tip_1_position": finger_pos[1],  # tri-finger position 1
                "tip_2_position": finger_pos[2],  # tri-finger position 2
            }
        elif self.deep_wbc.get_control_mode() == Control_Phase.POSITION:
            observation = {
                "joint_position": robot_observation.position,  # joint position
                "joint_velocity": robot_observation.velocity,  # joint velocity
                "joint_torque": robot_observation.torque,  # joint torque
                "tip_force": np.subtract(robot_observation.tip_force, self.tip_force_offset),  # tip force

                "object_position": cube_pos,  # cube position
                "object_rpy": cube_orn,  # cube orientation
                "goal_position": active_goal,  # goal position
                "object_goal_distance": active_goal - cube_pos,  # cube to goal distance

                "tip_0_position": finger_pos[0],  # tri-finger position 0
                "tip_1_position": finger_pos[1],  # tri-finger position 1
                "tip_2_position": finger_pos[2]  # tri-finger position 2
            }
        else:
            raise NotImplemented()

        return observation

    def _internal_step(self, action):
        self.step_count += 1
        # send action to robot
        robot_action = self._gym_action_to_robot_action(action)
        t = self.platform.append_desired_action(robot_action)

        # update goal visualization
        if self.visualization:
            goal_position = task.get_active_goal(self.info["trajectory"], t)
            self.goal_marker.set_state(goal_position, (0, 0, 0, 1))
            time.sleep(0.001)
        return t

    def _apply_action(self, action):
        if self.deep_wbc.get_control_mode() == Control_Phase.POSITION:
            init_joint_pos = self.observer.dt['joint_position']
            tar_joint_pos = action['position']
            tg = trajectory.get_interpolation_planner(init_pos=init_joint_pos,
                                                      tar_pos=tar_joint_pos,
                                                      start_time=0,
                                                      reach_time=self.step_size)
            for i in range(self.step_size):
                if self.step_count >= task.EPISODE_LENGTH:
                    break
                _action = tg(i + 1)
                t = self._internal_step({'position': _action, 'torque': None})
        elif self.deep_wbc.get_control_mode() == Control_Phase.TORQUE:
            for _ in range(self.step_size):
                if self.step_count >= task.EPISODE_LENGTH:
                    break
                t = self._internal_step(action)
        # Use observations of step t + 1 to follow what would be expected
        # in a typical gym environment.  Note that on the real robot, this
        # will not be possible
        self.info["time_index"] = t  # + 1

        obs_dict = self._create_observation(self.info["time_index"])

        eval_score = self.compute_reward(
            obs_dict["object_position"],
            obs_dict["goal_position"],
            self.info,
        )
        return obs_dict, eval_score

    def step(self, policy_action):
        self.deep_wbc.update()
        cur_phase_action = self.deep_wbc.get_action(policy_action)

        # current action
        obs_dict, eval_score = self._apply_action(cur_phase_action)

        self.observer.update(obs_dict)
        reward = self.deep_wbc.get_reward()
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = self.deep_wbc.get_done() or max_episode

        return self.observer.get_history_obs(), reward, done, self.info

    def render(self, mode='human'):
        pass

    def init_control(self, total_time=0.2):
        # move tri-finger to init spot
        total_step = int(total_time / 0.1)
        init_tip_pos = np.hstack([self.observer.dt[f'tip_{i}_position'] for i in range(3)])
        cube_pos = self.observer.dt['object_position']
        tar_tip_pos = np.hstack([
            np.add(cube_pos, [0.033, -0.01, 0]),  # arm_0 x+y+
            np.add(cube_pos, [0, -0.033, 0]),  # arm_1 x+y-
            np.add(cube_pos, [-0.033, 0, 0])  # arm_2 x-y+
        ])
        tg = trajectory.get_path_planner(init_pos=init_tip_pos,
                                         tar_pos=tar_tip_pos,
                                         start_time=0,
                                         reach_time=total_step)
        for t in range(total_step):
            tar_tip_pos = tg(t + 1)
            arm_joi_pos = self.observer.dt['joint_position']
            to_goal_joints, _error = self.kinematics.inverse_kinematics(tar_tip_pos.reshape(3, 3),
                                                                        arm_joi_pos)
            obs_dict, eval_score = self._apply_action({
                'position': to_goal_joints,
                'torque': None
            })
            self.observer.update(obs_dict)
        # self.info["time_index"] = t*100

class RealPhaseControlEnv(PhaseControlEnv):
    def __init__(self, goal_trajectory, visualization, args, action_type=ActionType.POSITION):
        super().__init__(goal_trajectory, visualization, args, action_type)

    # def _apply_action(self, action):
    #     if self.deep_wbc.get_control_mode() == Control_Phase.POSITION:
    #         init_joint_pos = self.observer.dt['joint_position']
    #         tar_joint_pos = action['position']
    #         tg = trajectory.get_interpolation_planner(init_pos=init_joint_pos,
    #                                                   tar_pos=tar_joint_pos,
    #                                                   start_time=0,
    #                                                   reach_time=self.step_size)
    #         for i in range(self.step_size):
    #             if self.step_count >= task.EPISODE_LENGTH:
    #                 break
    #             _action = tg(i + 1)
    #             t = self._internal_step(_action)
    #     elif self.deep_wbc.get_control_mode() == Control_Phase.TORQUE:
    #         for _ in range(self.step_size):
    #             if self.step_count >= task.EPISODE_LENGTH:
    #                 break
    #             t = self._internal_step(action)
    #     # Use observations of step t + 1 to follow what would be expected
    #     # in a typical gym environment.  Note that on the real robot, this
    #     # will not be possible
    #     self.info["time_index"] = t  # + 1
    #
    #     obs_dict = self._create_observation(self.info["time_index"])
    #
    #     eval_score = self.compute_reward(
    #         obs_dict["object_position"],
    #         obs_dict["goal_position"],
    #         self.info,
    #     )
    #     return obs_dict, eval_score

    def step(self, policy_action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(policy_action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        self.deep_wbc.update()
        cur_phase_action = self.deep_wbc.get_action(policy_action)

        # current action
        obs_dict, eval_score = self._apply_action(cur_phase_action)

        self.observer.update(obs_dict)
        # reward = self.deep_wbc.get_reward()
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = max_episode  # or self.deep_wbc.get_done()

        return self.observer.get_history_obs(), eval_score, done, self.info

    def reset(self):
        import robot_fingers
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        self.deep_wbc.reset()

        # get goal trajectory
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}
        self.step_count = 0

        # initial step
        robot_action = self._gym_action_to_robot_action(self._initial_action)
        t = self.platform.append_desired_action(robot_action)
        self.info["time_index"] += 1
        self.step_count += 1
        self.tip_force_offset = self.platform.get_robot_observation(0).tip_force
        obs_dict = self._create_observation(0)
        self.observer.reset(obs_dict)
        obs = self.observer.update(obs_dict)
        self.init_control()
        return obs


if __name__ == '__main__':
    class A(object):
        model_name = 'vel'


    env = PhaseControlEnv(goal_trajectory=None,
                          visualization=True,
                          args=A())

    observation = env.reset()
    z = 0
    is_done = False
    while True:
        observation, r, is_done, info = env.step(np.random.uniform(env.action_space.low,
                                                                   env.action_space.high) * 0)
        z = info["time_index"]
        print("reward:", r)
