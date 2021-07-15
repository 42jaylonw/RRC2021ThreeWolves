import numpy as np

def compute_point_to_point_trajectory(t, T):
    assert T > 0
    a0 = 0
    a1 = 0
    a2 = 3 / T**2
    a3 = - 2 / T**3
    s = a0 + a1*t + a2*t**2 + a3*t**3
    return s
    # v = a1 + 2*a2*t + 3*a3*t**2
    # a = 2*a2 + 6*a3*t
    # return s, v, a

def get_path_planner(init_pos, tar_pos, start_time, reach_time):
    dist = tar_pos - init_pos

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            return init_pos + compute_point_to_point_trajectory(t, T) * dist
        else:
            return tar_pos
    return tg

def compute_point_to_point_acc_trajectory(t, T):
    a0 = 0
    a1 = 0
    a2 = 3 / T**2
    a3 = - 2 / T**3
    s = a0 + a1*t + a2*t**2 + a3*t**3
    # return s
    v = a1 + 2*a2*t + 3*a3*t**2
    a = 2*a2 + 6*a3*t
    return s, v, a

def get_acc_planner(init_pos, tar_pos, start_time, reach_time):
    dist = tar_pos - init_pos

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            s, t_v, t_a = compute_point_to_point_acc_trajectory(t, T)
            t_s = init_pos + s * dist
            t_v *= dist
            t_a *= dist
            return [t_s, t_v, t_a]
        else:
            return [tar_pos, np.zeros(3), np.zeros(3)]
    return tg

def get_interpolation_planner(init_pos, tar_pos, start_time, reach_time):
    interpolation = (tar_pos - init_pos) / (reach_time - start_time)

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            return init_pos + t * interpolation
        else:
            return tar_pos
    return tg


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    path = get_acc_planner(np.array([0, 0, 0]), np.array([30, -10, 20]), 20, 50)
    q = np.array([path(t) for t in range(20, 80)])
    for i in range(3):
        plt.title(['displacement', 'velocity', 'acceleration'][i])
        plt.plot(range(20, 80), q[:, i, 0])
        plt.plot(range(20, 80), q[:, i, 1])
        plt.plot(range(20, 80), q[:, i, 2])
        plt.show()
