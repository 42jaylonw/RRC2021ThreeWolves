import time

import gym
import numpy as np
import pybullet
from trifinger_simulation import TriFingerPlatform, visual_objects
from trifinger_simulation.tasks import move_cube_on_trajectory as task

from three_wolves.envs.base_cube_env import ActionType, BaseCubeTrajectoryEnv
from three_wolves.envs.utilities.env_utils import HistoryWrapper, resetCamera
from three_wolves.deep_whole_body_controller.deep_wbc import DeepWBC
from three_wolves.deep_whole_body_controller.base_joint_controller import Control_Phase
from  three_wolves.deep_whole_body_controller.utility import trajectory

class PhaseControlEnv(BaseCubeTrajectoryEnv):
    def __init__(self, goal_trajectory, visualization, history_num=3, robot_type='sim'):
        super(PhaseControlEnv, self).__init__(
            goal_trajectory=goal_trajectory,
            action_type=ActionType.TORQUE,
            step_size=1)
        self.visualization = visualization
        self.observer = HistoryWrapper(history_num)
        self.deep_wbc = DeepWBC(self.observer, self.step_size, robot_type)
        # create observation space
        _duplicate = lambda x: np.array([x] * history_num).flatten()

        spaces = TriFingerPlatform.spaces
        self.observation_space = gym.spaces.Box(
            low=np.hstack([
                spaces.object_position.gym.low,  # cube position
                [-2*np.pi] * 3,                    # cube rpy
                spaces.object_position.gym.low,  # goal position
                [-0.3] * 3,                      # goal-cube difference
                [0]                              # goal-cube distance
            ]),
            high=np.hstack([
                spaces.object_position.gym.high,  # cube position
                [2*np.pi] * 3,                      # cube rpy
                spaces.object_position.gym.high,  # goal position
                [0.3] * 3,                        # goal-cube difference
                [1]                               # goal-cube distance
            ])
        )
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
        # initial_object_pose = task.move_cube.Pose(
        #     position=task.INITIAL_CUBE_POSITION
        # )
        _random_obj_xy_pos = np.random.uniform(
            low=[-0.06]*2,
            high=[0.06]*2,
        )
        _random_obj_yaw_ori = np.random.uniform(-2*np.pi, 2*np.pi)
        # _random_obj_yaw_ori = np.pi
        _random_obj_yaw_ori = pybullet.getQuaternionFromEuler([0, 0, _random_obj_yaw_ori])
        random_object_pose = task.move_cube.Pose(
            position=[_random_obj_xy_pos[0],
                      _random_obj_xy_pos[1],
                      task.INITIAL_CUBE_POSITION[2]],
            orientation=_random_obj_yaw_ori
        )
        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=random_object_pose,
        )

        self.deep_wbc.reset(self.apply_action)

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
        obs, _ = self._create_observation(0)
        return obs

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        camera_observation = self.platform.get_camera_observation(t)
        object_observation = camera_observation.filtered_object_pose
        active_goal = np.asarray(
            task.get_active_goal(self.info["trajectory"], t)
        )
        cube_pos = object_observation.position
        cube_orn = pybullet.getEulerFromQuaternion(object_observation.orientation)
        # cube_pos = self.platform.cube.get_state()[0]
        # cube_orn = pybullet.getEulerFromQuaternion(self.platform.cube.get_state()[1])
        # compute finger positions
        finger_pos = self.deep_wbc.kinematics.forward_kinematics(robot_observation.position)
        obs_dict = {
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
        self.observer.update(obs_dict)
        rl_obs = np.hstack([
            cube_pos,  # cube position
            cube_orn,  # cube rpy
            active_goal,  # goal position
            active_goal - cube_pos,  # goal-cube difference
            np.linalg.norm(active_goal - cube_pos)  # goal-cube distance
        ])
        return rl_obs, obs_dict

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

    def apply_action(self, action):
        # init_joint_pos = self.observer.dt['joint_position']
        # tar_joint_pos = action['position']
        # tg = trajectory.get_interpolation_planner(init_pos=init_joint_pos,
        #                                           tar_pos=tar_joint_pos,
        #                                           start_time=0,
        #                                           reach_time=self.step_size)
        # for i in range(self.step_size):
        #     if self.step_count >= task.EPISODE_LENGTH:
        #         break
        #     _action = tg(i + 1)
        #     t = self._internal_step(_action)
        tg = trajectory.get_interpolation_planner(init_pos=self.observer.dt['joint_torque'],
                                                  tar_pos=action,
                                                  start_time=0,
                                                  reach_time=self.step_size)
        for i in range(self.step_size):
            if self.step_count >= task.EPISODE_LENGTH:
                break
            _action = tg(i + 1)
            t = self._internal_step(_action)
            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t  # + 1

        _, obs_dict = self._create_observation(self.info["time_index"])

        eval_score = self.compute_reward(
            obs_dict["object_position"],
            obs_dict["goal_position"],
            self.info,
        )
        self.info['eval_score'] = eval_score
        return obs_dict, eval_score

    def step(self, policy_action):
        self.deep_wbc.step(policy_action)
        # cur_phase_action = self.deep_wbc.get_action()
        # current action
        # obs_dict, eval_score = self.apply_action(cur_phase_action)
        # self.info['eval_score'] = eval_score

        reward = self.deep_wbc.get_reward()
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = self.deep_wbc.get_done() or max_episode

        return self._create_observation(self.info["time_index"])[0], reward, done, self.info

    def render(self, mode='human'):
        pass

class RealPhaseControlEnv(PhaseControlEnv):
    def __init__(self,
                 goal_trajectory):
        super().__init__(goal_trajectory=goal_trajectory,
                         visualization=False,
                         robot_type='real')

    def _internal_step(self, action_dict):
        self.step_count += 1
        # send action to robot
        robot_action = self._gym_action_to_robot_action(action_dict['position'])
        t = self.platform.append_desired_action(robot_action)
        return t

    def step(self, policy_action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(policy_action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        self.deep_wbc.step(policy_action)
        reward = self.deep_wbc.get_reward()
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = max_episode

        return self._create_observation(self.info["time_index"])[0], reward, done, self.info

    def reset(self):
        import robot_fingers
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        self.deep_wbc.reset(self.apply_action)

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
        obs, _ = self._create_observation(0)
        return obs


if __name__ == '__main__':
    class A(object):
        model_name = 'force'

    env = PhaseControlEnv(goal_trajectory=None,
                          visualization=True)

    observation = env.reset()
    is_done = False
    t = 0
    while t < task.EPISODE_LENGTH:
        observation, r, is_done, info = env.step([2.2/4, 0.7/4, 3.5/4,
                                                  0.5, 0.5, 0.5])
        _score = info["eval_score"]
        if t % 1 == 0:
            print("reward:", _score)
        t += 0.001*env.step_size
        if is_done:
            env.reset()
