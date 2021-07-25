import gym
from trifinger_simulation import TriFingerPlatform
from trifinger_simulation import visual_objects
from trifinger_simulation.tasks import move_cube_on_trajectory as task
import numpy as np
from three_wolves.envs.base_cube_env import ActionType, BaseCubeTrajectoryEnv

class RLPositionHistoryEnv(BaseCubeTrajectoryEnv):
    def render(self, mode='human'):
        pass

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
                _duplicate(spaces.robot_position.gym.low),   # joint position
                _duplicate(spaces.robot_velocity.gym.low),   # joint velocity
                _duplicate(spaces.robot_torque.gym.low),     # joint torque
                _duplicate([0] * 3),                         # tip force

                _duplicate(spaces.object_position.gym.low),    # cube position
                _duplicate([-np.pi] * 3),                      # cube orientation
                _duplicate(spaces.object_position.gym.low),    # goal position
                _duplicate([-0.3] * 3),                        # cube to goal distance

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
            "joint_position": robot_observation.position,   # joint position
            "joint_velocity": robot_observation.velocity,   # joint velocity
            "joint_torque": robot_observation.torque,       # joint torque
            "tip_force": robot_observation.tip_force,       # tip force
            "achieved_goal": cube_pos,       # cube position
            "object_orientation": cube_orn,  # cube orientation
            "desired_goal": active_goal,     # goal position
            "cube_goal_distance": active_goal - cube_pos,  # cube to goal distance
            "finger_0_position": finger_pos[0],  # tri-finger position 0
            "finger_1_position": finger_pos[1],  # tri-finger position 1
            "finger_2_position": finger_pos[2]   # tri-finger position 2
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
        self.inverse_kinematics = self.platform.simfinger.kinematics.inverse_kinematics
        self.forward_kinematics = self.platform.simfinger.kinematics.forward_kinematics

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
        ReachReward = self.ReachReward(obs_dict)*2
        TipForceBalance = self.GraspPenalty(obs_dict)
        SlipperyPenalty = self.SlipperyPenalty()
        TipTriangle = self.TipTriangle(obs_dict)

        total_reward = ReachReward + NotNear + TipForceBalance + SlipperyPenalty + TipTriangle
        return total_reward

    def _apply_action(self, action):
        for _ in range(self.step_size):
            self.step_count += 1
            if self.step_count > task.EPISODE_LENGTH:
                raise RuntimeError("Exceeded number of steps for one episode.")

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
            _pre_action = self.tri_to_cube(obs_dict)
            obs_dict, _ = self._apply_action(_pre_action)
        max_episode = self.step_count >= task.EPISODE_LENGTH
        done = max_episode
        self.info['eval_score'] = eval_score

        return self.observer.get_history_obs(), reward, done, self.info

    def tri_to_cube(self, obs_dict):
        arm_joi_pos = obs_dict['joint_position']
        cube_pos = obs_dict['achieved_goal']
        tar_arm_pos_0 = [
            np.add(cube_pos, [0, 0.03,  0]),  # arm_0 x+y+
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
        return target_reward * 2.5e3    # -1 ~ 1

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
        return sp*100

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
