#!/usr/bin/python3

import json
import os.path
import sys
from three_wolves.move_cube_env import RLPositionHistoryEnv
from stable_baselines3 import SAC

def main():
    # the goal is passed as JSON string
    # goal_json = sys.argv[1]
    # goal_trajectory = json.loads(goal_json)

    env = RLPositionHistoryEnv(goal_trajectory=None,
                               visualization=True,
                               evaluation=True)
    log_dir = f"src/three_wolves/three_wolves/model_save/triangle_cube_tg/"
    policy = SAC.load(log_dir + "best_model.zip")

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