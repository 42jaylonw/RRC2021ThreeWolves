import os
import matplotlib.pyplot as plt
from trifinger_simulation.visual_objects import CubeMarker
import pybullet
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

U = []


def tag(xyz):
    U.append(CubeMarker(
        width=0.03,
        position=xyz,
        orientation=(0, 0, 0, 1),
        color=(1, 0, 0, 0.7),
        pybullet_client_id=0,
    ))


def clean():
    for o in U:
        pybullet.removeBody(o.body_id, physicsClientId=0)


def resetCamera():
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=0.5,
        cameraYaw=0,
        cameraPitch=-41,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=0
    )


def compute_dist(p0, p1):
    return np.linalg.norm(np.subtract(p1, p0))


def fVCap(v_tar, vel):
    return max(-v_tar, min(vel, v_tar))


def compute_acc(pos_3):
    assert pos_3.shape == (3, 3)
    vel_0 = compute_dist(pos_3[0], pos_3[1]) / 0.001
    vel_1 = compute_dist(pos_3[1], pos_3[2]) / 0.001
    acc_3 = (vel_1 - vel_0) / 0.001
    return acc_3


def exp_mini(cur, tar=0, wei=-5):
    assert wei < 0
    return np.exp(wei * np.square(np.sum(tar - cur)))


def Delta(seq):
    return np.sum(np.abs(seq - np.mean(seq)))


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


def plot_results(log_folder, title='Learning Curve'):
    from scipy.signal import savgol_filter
    R = load_results(log_folder)['r']
    T = load_results(log_folder)['t']
    _w = 7
    _window_size = len(R) // _w if (len(R) // _w) % 2 != 0 else len(R) // _w + 1
    filtered = savgol_filter(R, _window_size, 1)

    plt.title('smoothed returns')
    plt.ylabel('Returns')
    plt.xlabel('time step')
    plt.plot(T, filtered)
    plt.grid()
    plt.show()


class HistoryWrapper:
    def __init__(self,
                 history_num=3):
        self._history_obs = {}
        self.history_num = history_num

    def reset(self, init_obs_dict):
        for k, v in init_obs_dict.items():
            self._history_obs.update({k: [v]*self.history_num})
        return self.get_history_obs()

    def update(self, obs_dict):
        for k, v in obs_dict.items():
            assert len(v) == len(self._history_obs[k][0]), 'wrong shape'
            assert k in self._history_obs.keys(), 'wrong key'
            self._history_obs[k].pop()
            self._history_obs[k].insert(0, v)
            assert len(self._history_obs[k]) == self.history_num

        return self.get_history_obs()

    def get_history_obs(self):
        _obs = []
        for _, v in self._history_obs.items():
            _obs.append(np.hstack(v))
        return np.hstack(_obs)

    def search(self, k):
        return self._history_obs[k]


if __name__ == '__main__':
    his = HistoryWrapper(
        {'pos': [1, 2, 3],
         'orn': [6, 6, 6]
         }
    )
    q = his.get_history_obs()
