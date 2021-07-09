#!/usr/bin/python3

import json
import os
import sys

from three_wolves.move_cube_env import RLPositionHistoryEnv
from stable_baselines3 import SAC

def main():
    # the goal is passed as JSON string
    goal_json = sys.argv[1]
    goal_trajectory = json.loads(goal_json)
    print(goal_trajectory)
    env = RLPositionHistoryEnv(
        goal_trajectory=goal_trajectory,
        visualization=False)

    # load
    log_dir = '/userhome/position_model.zip'
    print(os.listdir('/userhome'))
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
