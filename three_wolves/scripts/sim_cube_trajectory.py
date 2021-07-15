#!/usr/bin/python3
import sys
import json
from three_wolves.envs import phase_cube_env
from stable_baselines3 import SAC

class M:
    def __init__(self, model_name='tg'):
        self.model_name = model_name

def main():
    # the goal is passed as JSON string
    try:
        goal_json = sys.argv[1]
        goal_trajectory = json.loads(goal_json)
    except IndexError:
        goal_trajectory = None
    model = M()
    env = phase_cube_env.PhaseControlEnv(goal_trajectory=goal_trajectory,
                                         visualization=False,
                                         args=model)
    log_filename = f"/userhome/{model.model_name}.zip"
    policy = SAC.load(log_filename)

    observation = env.reset()
    done = False
    while not done:
        action = policy.predict(observation)[0]
        observation, reward, done, info = env.step(action)
        t = info["time_index"]
        print(f"{t} reward:", reward)


if __name__ == "__main__":
    main()