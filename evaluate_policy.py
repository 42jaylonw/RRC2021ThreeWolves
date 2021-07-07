#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - The trajectory as a JSON string
 - Path to the file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import argparse
import json

from three_wolves.move_cube_env import RLPositionHistoryEnv
from stable_baselines3 import SAC

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory",
        type=json.loads,
        metavar="JSON",
        help="Goal trajectory as a JSON string.",
    )
    parser.add_argument(
        "action_log_file",
        type=str,
        help="File to which the action log is written.",
    )
    args = parser.parse_args()

    # TODO: Replace with your environment if you used a custom one.
    # env = cube_trajectory_env.SimCubeTrajectoryEnv(
    #     goal_trajectory=args.trajectory,
    #     action_type=cube_trajectory_env.ActionType.POSITION,
    #     # IMPORTANT: Do not enable visualisation here, as this will result in
    #     # invalid log files (unfortunately the visualisation slightly influence
    #     # the behaviour of the physics in pyBullet...).
    #     visualization=False,
    # )
    env = RLPositionHistoryEnv(goal_trajectory=args.trajectory,
                               visualization=False,
                               evaluation=True)

    # TODO: Replace this with your model
    # policy = RandomPolicy(env.action_space)
    # policy = PointAtTrajectoryPolicy(env.action_space, args.trajectory)
    log_dir = f"src/pkg/rl_position_control/model_save/triangle_cube_tg/"
    policy = SAC.load(log_dir + "best_model.zip")

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!
    is_done = False
    observation = env.reset()
    accumulated_reward = 0
    t = 0
    while not is_done:
        # action = policy.predict(observation)
        action = policy.predict(observation)[0]
        observation, reward, is_done, info = env.step(action)
        t = info["time_index"]
        accumulated_reward += reward

    print("Accumulated reward: {}".format(accumulated_reward))

    # store the log for evaluation
    env.platform.store_action_log(args.action_log_file)


if __name__ == "__main__":
    main()