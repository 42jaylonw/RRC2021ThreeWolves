#!/usr/bin/python3

import json
import sys

from three_wolves.move_cube_env import RealRobotCubeTrajectoryEnv
from stable_baselines3 import SAC

def main():
    # the goal is passed as JSON string
    # goal_json = sys.argv[1]
    # print(goal_json)
    # goal_trajectory = json.loads(goal_json)

    env = RealRobotCubeTrajectoryEnv(
        goal_trajectory=None,
        visualization=False)

    # load
    log_dir = '/userhome/best_model.zip'
    policy = SAC.load(log_dir)

    observation = env.reset()
    t = 0
    is_done = False
    while not is_done:
        action = policy.predict(observation)[0]
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        print("reward:", reward)


if __name__ == "__main__":
    main()
