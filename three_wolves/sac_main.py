from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import os
from utils import SaveOnBestTrainingRewardCallback, plot_results
from move_cube_env import RLPositionHistoryEnv
from src.three_wolves.trifinger_simulation.tasks import move_cube_on_trajectory as mct
import argparse

def get_arguments():
    parser = argparse.ArgumentParser("Tri-finger")
    # Environment
    parser.add_argument("--run-mode", '-r', type=str, choices=['train', 'test'], help="train or test model")
    parser.add_argument("--device", '-d', type=int, default=0, help="cuda device")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    # log_dir = f"policy/model_save/ThreeWolves_PreStage/"
    log_dir = f"model_save/triangle_cube_tg_0706/"

    if args.run_mode == 'train':
        env = RLPositionHistoryEnv(goal_trajectory=None, visualization=False)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        callback = SaveOnBestTrainingRewardCallback(check_freq=int(1e4), log_dir=log_dir)
        model = SAC(MlpPolicy, env, verbose=1, device=f'cuda:{args.device}')
        model.learn(total_timesteps=int(3e8), callback=callback, log_interval=int(1e4))
    else:
        env = RLPositionHistoryEnv(goal_trajectory=None, visualization=True)
        plot_results(log_dir)
        model = SAC.load(log_dir + "best_model.zip")
        SCORE = []
        obs = env.reset()
        S = 0
        for i in range(mct.EPISODE_LENGTH//100):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            S += info['eval_score']
            if done:
                obs = env.reset()
        print(S)
