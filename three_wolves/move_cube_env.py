import enum
import os
import typing
import gym
import robot_fingers
from trifinger_simulation import TriFingerPlatform, trifingerpro_limits, visual_objects
from trifinger_simulation.tasks import move_cube_on_trajectory as task
from trifinger_simulation import pinocchio_utils
from trifinger_simulation import finger_types_data
from three_wolves.utils import *

import numpy as np


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class BaseCubeTrajectoryEnv(gym.GoalEnv):
    """Gym environment for moving cubes with TriFingerPro."""

    def __init__(
            self,
            goal_trajectory: typing.Optional[task.Trajectory] = None,
            action_type: ActionType = ActionType.POSITION,
            step_size: int = 1,
    ):
        """Initialize.

        Args:
            goal_trajectory: Goal trajectory for the cube.  If ``None`` a new
                random trajectory is sampled upon reset.
            action_type: Specify which type of actions to use.
                See :class:`ActionType` for details.
            step_size:  Number of actual control steps to be performed in one
                call of step().
        """
        # Basic initialization
        # ====================

        if goal_trajectory is not None:
            task.validate_goal(goal_trajectory)
        self.goal = goal_trajectory

        self.action_type = action_type

        if step_size < 1:
            raise ValueError("step_size cannot be less than 1.")
        self.step_size = step_size

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                    }
                ),
                "object_observation": gym.spaces.Dict(
                    {
                        "position": object_state_space["position"],
                        "orientation": object_state_space["orientation"],
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space["position"],
                "achieved_goal": object_state_space["position"],
            }
        )

    def compute_reward(
            self,
            achieved_goal: task.Position,
            desired_goal: task.Position,
            info: dict,
    ) -> float:
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal: Current position of the object.
            desired_goal: Goal position of the current trajectory step.
            info: An info dictionary containing a field "time_index" which
                contains the time index of the achieved_goal.

        Returns:
            The reward that corresponds to the provided achieved goal w.r.t. to
            the desired goal. Note that the following should always hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        # This is just some sanity check to verify that the given desired_goal
        # actually matches with the active goal in the trajectory.
        active_goal = np.asarray(
            task.get_active_goal(
                self.info["trajectory"], self.info["time_index"]
            )
        )
        assert np.all(active_goal == desired_goal), "{}: {} != {}".format(
            info["time_index"], active_goal, desired_goal
        )

        return -task.evaluate_state(
            info["trajectory"], info["time_index"], achieved_goal
        )

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float): amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the current time index.
        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        task.seed(seed)
        return [seed]

    # def _create_observation(self, t, action):
    #     robot_observation = self.platform.get_robot_observation(t)
    #     camera_observation = self.platform.get_camera_observation(t)
    #     object_observation = camera_observation.filtered_object_pose
    #
    #     active_goal = np.asarray(
    #         task.get_active_goal(self.info["trajectory"], t)
    #     )
    #
    #     observation = {
    #         "robot_observation": {
    #             "position": robot_observation.position,
    #             "velocity": robot_observation.velocity,
    #             "torque": robot_observation.torque,
    #         },
    #         "object_observation": {
    #             "position": object_observation.position,
    #             "orientation": object_observation.orientation,
    #         },
    #         "action": action,
    #         "desired_goal": active_goal,
    #         "achieved_goal": object_observation.position,
    #     }
    #     return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _set_urdf_path(self):
        """
        Sets the paths for the URDFs to use depending upon the finger type
        """
        try:
            from ament_index_python.packages import get_package_share_directory

            self.robot_properties_path = get_package_share_directory(
                "robot_properties_fingers"
            )
        except Exception:
            self.robot_properties_path = os.path.join(
                os.path.dirname(__file__), "robot_properties_fingers"
            )
        print(self.robot_properties_path)
        urdf_file = finger_types_data.get_finger_urdf("trifingerpro")
        self.finger_urdf_path = os.path.join(
            self.robot_properties_path, "urdf", urdf_file
        )


class RLPositionHistoryEnv(BaseCubeTrajectoryEnv):
    def __init__(self, goal_trajectory, visualization, history_num=3):
        super(RLPositionHistoryEnv, self).__init__(
            goal_trajectory=goal_trajectory, action_type=ActionType.POSITION, step_size=100)

        # env params
        # self.evaluation = evaluation
        self.visualization = visualization

        # policy params
        _range = (self.action_space.high - self.action_space.low) / 20
        self.action_space = gym.spaces.Box(
            low=-_range, high=_range
        )

        # create history observer
        self.observer = HistoryWrapper(history_num)

        # create observation space
        _duplicate = lambda x: np.array([x] * history_num).flatten()
        spaces = TriFingerPlatform.spaces
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
        finger_pos = self.forward_kinematics(robot_observation.position)
        observation = {
            "joint_position": robot_observation.position,  # joint position
            "joint_velocity": robot_observation.velocity,  # joint velocity
            "joint_torque": robot_observation.torque,  # joint torque
            "tip_force": robot_observation.tip_force,  # tip force
            "achieved_goal": cube_pos,  # cube position
            "object_orientation": cube_orn,  # cube orientation
            "desired_goal": active_goal,  # goal position
            "cube_goal_distance": active_goal - cube_pos,  # cube to goal distance
            "finger_0_position": finger_pos[0],  # tri-finger position 0
            "finger_1_position": finger_pos[1],  # tri-finger position 1
            "finger_2_position": finger_pos[2]  # tri-finger position 2
        }
        return observation

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
        # self.inverse_kinematics = self.platform.simfinger.kinematics.inverse_kinematics
        # self.forward_kinematics = self.platform.simfinger.kinematics.forward_kinematics
        # _trifinger_urdf = '/userhome/robot_properties_fingers/urdf/trifingerpro_with_stage.urdf'
        # trifinger_urdf = '/userhome/robot_properties_fingers/urdf/pro/trifingerpro_with_stage.urdf'
        self._set_urdf_path()
        kinematics = pinocchio_utils.Kinematics(
            self.finger_urdf_path, ["finger_tip_link_0",
                                    "finger_tip_link_120",
                                    "finger_tip_link_240"])
        self.inverse_kinematics = kinematics.inverse_kinematics
        self.forward_kinematics = kinematics.forward_kinematics

        obs_dict = self._create_observation(0)
        self._last_dist_to_goal = compute_dist(obs_dict["desired_goal"], obs_dict['achieved_goal'])

        # script to let arm reach box
        while not self.IsNear(obs_dict):
            _pre_action = self.tri_to_cube(obs_dict)
            obs_dict, _ = self._apply_action(_pre_action)

        return self.observer.reset(obs_dict)

    def get_reward(self, obs_dict):
        # ObjectStability = -10 if self.IsWarp(obs_dict) else 0
        NotNear = -10 if not self.IsNear(obs_dict) else 0.1
        ReachReward = self.ReachReward(obs_dict) * 2
        TipForceBalance = self.GraspPenalty(obs_dict)
        SlipperyPenalty = self.SlipperyPenalty()
        TipTriangle = self.TipTriangle(obs_dict)

        total_reward = ReachReward + NotNear + TipForceBalance + SlipperyPenalty + TipTriangle
        return total_reward

    def _apply_action(self, action):
        for _ in range(self.step_size):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                break

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # update goal visualization
            if self.visualization:
                goal_position = task.get_active_goal(self.info["trajectory"], t)
                self.goal_marker.set_state(goal_position, (0, 0, 0, 1))
                # time.sleep(0.001)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            self.info["time_index"] = t + 1

        obs_dict = self._create_observation(self.info["time_index"])

        eval_score = self.compute_reward(
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
            self.info,
        )
        return obs_dict, eval_score

    def step(self, action):
        joi_pos = self._create_observation(self.info["time_index"])['joint_position']
        action = joi_pos + action
        obs_dict, eval_score = self._apply_action(action)
        self.observer.update(obs_dict)
        reward = self.get_reward(obs_dict)
        # if self.evaluation:
        #     done = max_episode
        #     # reward = eval_score
        # else:
        if self.IsFar(obs_dict):
            while not self.IsNear(obs_dict):
                _pre_action = self.tri_to_cube(obs_dict)
                obs_dict, _ = self._apply_action(_pre_action)
                if self.step_count >= task.EPISODE_LENGTH:
                    break
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = max_episode
        self.info['eval_score'] = eval_score

        return self.observer.get_history_obs(), reward, done, self.info

    def tri_to_cube(self, obs_dict):
        arm_joi_pos = obs_dict['joint_position']
        cube_pos = obs_dict['achieved_goal']
        tar_arm_pos_0 = [
            np.add(cube_pos, [0, 0.03, 0]),  # arm_0 x+y+
            np.add(cube_pos, [0, -0.03, 0]),  # arm_1 x+y-
            np.add(cube_pos, [-0.03, 0, 0])  # arm_2 x-y+
        ]

        to_cube_angle, _error = self.inverse_kinematics(tar_arm_pos_0, arm_joi_pos)
        return to_cube_angle

    def ReachReward(self, obs_dict):
        # z first then xy
        cube_pos = obs_dict['achieved_goal']
        goal_pos = obs_dict['desired_goal']
        if compute_dist(goal_pos, cube_pos) < 0.015:
            return exp_mini(cube_pos, goal_pos, wei=-10000) * 100

        _tar_vel = 4e-3  # (0.3*self.step_size) / (7*1000)
        dist_to_goal = compute_dist(goal_pos, cube_pos)
        target_reward = fVCap(_tar_vel, dist_to_goal - self._last_dist_to_goal)

        self._last_dist_to_goal = dist_to_goal
        return target_reward * 2.5e3  # -1 ~ 1

    @staticmethod
    def IsWarp(obs_dict):
        return any(np.abs(obs_dict['object_orientation'][:3]) > np.pi / 6)

    @staticmethod
    def IsNear(obs_dict):
        cube_pos = np.array(obs_dict['achieved_goal'])
        tri_distance = [compute_dist(obs_dict['finger_0_position'], cube_pos),
                        compute_dist(obs_dict['finger_1_position'], cube_pos),
                        compute_dist(obs_dict['finger_2_position'], cube_pos)]
        return all(np.array(tri_distance) < 0.055)

    @staticmethod
    def IsFar(obs_dict):
        cube_pos = np.array(obs_dict['achieved_goal'])
        tri_distance = [compute_dist(obs_dict['finger_0_position'], cube_pos),
                        compute_dist(obs_dict['finger_1_position'], cube_pos),
                        compute_dist(obs_dict['finger_2_position'], cube_pos)]
        return any(np.array(tri_distance) > 0.1)

    @staticmethod
    def IsGrasp(obs_dict):
        tip_force = obs_dict['tip_force']
        return all(tip_force > 0.1)

    def GraspPenalty(self, obs_dict):
        tip_force = obs_dict['tip_force']
        # minimize the difference
        if self.IsGrasp(obs_dict):
            gr = exp_mini(Delta(tip_force), wei=-100)  # -10 ~ 0
            return gr
        else:
            return - 5

    def SlipperyPenalty(self):
        cube_pos_his = self.observer.search('achieved_goal')
        delta_tri_0 = Delta([compute_dist(p0, p1) for p0, p1 in
                             zip(self.observer.search('finger_0_position'), cube_pos_his)])
        delta_tri_1 = Delta([compute_dist(p0, p1) for p0, p1 in
                             zip(self.observer.search('finger_1_position'), cube_pos_his)])
        delta_tri_2 = Delta([compute_dist(p0, p1) for p0, p1 in
                             zip(self.observer.search('finger_2_position'), cube_pos_his)])
        sp = - np.sum([delta_tri_0, delta_tri_1, delta_tri_2])
        return sp * 100

    @staticmethod
    def TipTriangle(obs_dict):
        tip_0 = obs_dict['finger_0_position']
        tip_1 = obs_dict['finger_1_position']
        tip_2 = obs_dict['finger_2_position']
        # z_pos_penalty = - Delta([tip_0[2], tip_1[2], tip_2[2]])
        # tip pos form a regular triangle
        t0_t1 = compute_dist(tip_0, tip_1)
        t0_t2 = compute_dist(tip_0, tip_2)
        t1_t2 = compute_dist(tip_1, tip_2)
        triangle_xyz_penalty = exp_mini(Delta([t0_t1, t0_t2, t1_t2]), wei=-1000)
        return triangle_xyz_penalty * 10


class RealRobotCubeTrajectoryEnv(RLPositionHistoryEnv):
    """Gym environment for moving cubes with real TriFingerPro."""

    def reset(self):
        # cannot reset multiple times
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformWithObjectFrontend()

        # if no goal is given, sample one randomly
        if self.goal is None:
            trajectory = task.sample_goal()
        else:
            trajectory = self.goal

        self.info = {"time_index": -1, "trajectory": trajectory}
        self.step_count = 0
        self._set_urdf_path()
        kinematics = pinocchio_utils.Kinematics(
            self.finger_urdf_path, ["finger_tip_link_0",
                                    "finger_tip_link_120",
                                    "finger_tip_link_240"])
        self.inverse_kinematics = kinematics.inverse_kinematics
        self.forward_kinematics = kinematics.forward_kinematics

        obs_dict = self._create_observation(0)
        self._last_dist_to_goal = compute_dist(obs_dict["desired_goal"], obs_dict['achieved_goal'])

        # script to let arm reach box
        while not self.IsNear(obs_dict):
            _pre_action = self.tri_to_cube(obs_dict)
            obs_dict, _ = self._apply_action(_pre_action)

        return self.observer.reset(obs_dict)

    def _apply_action(self, action):
        for _ in range(self.step_size):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                break

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            self.info["time_index"] = t

        obs_dict = self._create_observation(self.info["time_index"])

        eval_score = self.compute_reward(
            obs_dict["achieved_goal"],
            obs_dict["desired_goal"],
            self.info,
        )
        return obs_dict, eval_score


if __name__ == '__main__':
    env = RLPositionHistoryEnv(goal_trajectory=None,
                               visualization=True)

    observation = env.reset()
    t = 0
    is_done = False
    while not is_done:
        observation, reward, is_done, info = env.step(np.random.uniform(env.action_space.low,
                                                                        env.action_space.high))
        t = info["time_index"]
        print("reward:", reward)
