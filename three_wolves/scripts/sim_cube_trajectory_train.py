from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import os
from three_wolves.utils import *
from three_wolves.trifinger_simulation.tasks import move_cube_on_trajectory as mct

import argparse


def get_arguments():
    parser = argparse.ArgumentParser("Tri-finger")
    # Environment
    parser.add_argument("--scenario", '-s', default="residual", type=str,
                        choices=["residual", "precise", "triangle", "script"],
                        help="type of action")
    parser.add_argument("--run-mode", '-r', type=str, choices=['train', 'test'], help="train or test model")
    parser.add_argument("--device", '-d', type=int, default=0, help="cuda device")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    log_dir = f"rl_position_control/model_save/{args.scenario}_cube_tg/"
    env_class = eval(f'{args.scenario}')
    if args.run_mode == 'train':
        env = env_class(False)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        callback = SaveOnBestTrainingRewardCallback(check_freq=int(1e3), log_dir=log_dir)
        model = SAC(MlpPolicy, env, verbose=1, device=f'cuda:{args.device}')
        model.learn(total_timesteps=int(3e8), callback=callback, log_interval=int(1e3))
    else:
        env = env_class(True)
        plot_results(log_dir)
        model = SAC.load(log_dir + "best_model.zip")
        SCORE = []
        obs = env.reset()
        for _ in range(10):
            S = 0
            for i in range(mct.EPISODE_LENGTH//100):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                S += info['score']
                if done:
                    obs = env.reset()
            SCORE.append(S)
        print(f'10 times Mean Score: {np.mean(SCORE)}')
