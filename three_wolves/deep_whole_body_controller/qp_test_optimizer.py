import numpy as np
import quadprog
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

ACC_WEIGHT = np.array([1., 1., 1., 10., 10, 1.])


def compute_mass_matrix(robot_mass, robot_inertia, tip_positions):
    # yaw = 0.  # Set yaw to 0 for now as all commands are local.
    # rot_z = np.array([[np.cos(yaw), np.sin(yaw), 0.],
    #                   [-np.sin(yaw), np.cos(yaw), 0.],
    #                   [0., 0., 1.]])
    rot_z = np.eye(3)

    inv_mass = np.eye(3) / robot_mass
    inv_inertia = np.linalg.inv(robot_inertia)
    mass_mat = np.zeros((6, 9))

    for limb_id in range(3):
        mass_mat[:3, limb_id * 3:limb_id * 3 + 3] = inv_mass
        x = tip_positions[limb_id]
        tip_position_skew = np.array([[0, -x[2], x[1]],
                                      [x[2], 0, -x[0]],
                                      [-x[1], x[0], 0]])
        mass_mat[3:6,
        limb_id * 3:limb_id * 3 + 3] = rot_z.T.dot(inv_inertia).dot(tip_position_skew)
    return mass_mat


def compute_constraint_matrix(body_mass,
                              friction_coef,
                              f_min_ratio,
                              f_max_ratio):
    f_min = f_min_ratio * body_mass * 9.8 / friction_coef
    f_max = f_max_ratio * body_mass * 9.8 / friction_coef

    A = np.zeros((18, 9))
    lb = np.zeros(18)
    for limb_id in range(3):
        lb[limb_id * 2] = f_min
        lb[limb_id * 2 + 1] = -f_max
        if limb_id == 0 or limb_id == 2:
            A[limb_id * 2, limb_id * 3] = 1
            A[limb_id * 2 + 1, limb_id * 3] = -1
        else:
            A[limb_id * 2, limb_id * 3 + 1] = 1
            A[limb_id * 2 + 1, limb_id * 3 + 1] = -1

    A[6, :] = [friction_coef, 1] + [0] * 7
    A[7, :] = [friction_coef, -1] + [0] * 7
    A[8, :] = [friction_coef, 0, 1] + [0] * 6
    A[9, :] = [friction_coef, 0, -1] + [0] * 6
    A[10, :] = [0] * 3 + [1, friction_coef, 0] + [0] * 3
    A[11, :] = [0] * 3 + [-1, friction_coef, 0] + [0] * 3
    A[12, :] = [0] * 3 + [0, friction_coef, 1] + [0] * 3
    A[13, :] = [0] * 3 + [0, friction_coef, -1] + [0] * 3
    A[14, :] = [0] * 6 + [-friction_coef, 1, 0]
    A[15, :] = [0] * 6 + [-friction_coef, -1, 0]
    A[16, :] = [0] * 6 + [-friction_coef, 0, 1]
    A[17, :] = [0] * 6 + [-friction_coef, 0, -1]
    # for leg_id in range(3):
    #     row_id = 6 + leg_id * 4
    #     col_id = leg_id * 3
    #     lb[row_id:row_id + 4] = np.array([0, 0, 0, 0])
    #     A[row_id, col_id:col_id + 3] = np.array([1, 0, friction_coef])
    #     A[row_id + 1, col_id:col_id + 3] = np.array([-1, 0, friction_coef])
    #     A[row_id + 2, col_id:col_id + 3] = np.array([0, 1, friction_coef])
    #     A[row_id + 3, col_id:col_id + 3] = np.array([0, -1, friction_coef])

    return A.T, lb


def compute_objective_matrix(mass_matrix, desired_acc, acc_weight, reg_weight):
    g = np.array([0., 0., -9.8, 0., 0., 0.])
    Q = np.diag(acc_weight)
    R = np.ones(9) * reg_weight

    quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R
    linear_term = 1 * (g - desired_acc).T.dot(Q).dot(mass_matrix)
    return quad_term, linear_term


def compute_contact_force(robot_mass,
                          robot_inertia,
                          tip_positions,
                          desired_acc,

                          acc_weight=ACC_WEIGHT,
                          reg_weight=10,
                          friction_coef=0.45,
                          f_min_ratio=-100,
                          f_max_ratio=10):
    mass_matrix = compute_mass_matrix(
        # robot_info['body_mass'],
        # np.array(robot_info['body_inertia']).reshape((3, 3)),
        # robot_info['tip_position']
        robot_mass,
        robot_inertia,
        tip_positions
    )
    G, a = compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
                                    reg_weight)
    C, b = compute_constraint_matrix(robot_mass, friction_coef, f_min_ratio, f_max_ratio)
    G += 1e-4 * np.eye(9)

    try:
        result = quadprog.solve_qp(G, a, C, b)
    except Exception as r:
        print('wrong')
        result = np.zeros((3, 3))

    return -result[0].reshape((3, 3)) * 10

# def compute_contact_force(robot,
#                           desired_acc,
#                           contacts,
#                           acc_weight=ACC_WEIGHT,
#                           reg_weight=1e-4,
#                           friction_coef=0.45,
#                           f_min_ratio=0.1,
#                           f_max_ratio=10.):
#     mass_matrix = compute_mass_matrix(
#         robot.MPC_BODY_MASS,
#         np.array(robot.MPC_BODY_INERTIA).reshape((3, 3)),
#         robot.GetFootPositionsInBaseFrame())
#     G, a = compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
#                                     reg_weight)
#     C, b = compute_constraint_matrix(robot.MPC_BODY_MASS, contacts,
#                                      friction_coef, f_min_ratio, f_max_ratio)
#     G += 1e-4 * np.eye(12)
#
#     try:
#         result = quadprog.solve_qp(G, a, C, b)
#     except Exception as r:
#         print("qp_torque_optimizer: ", r)
#         result = np.load("data/safe_qp_torque_optimizer_result.npy", allow_pickle=True).tolist()
#         return -result[0].reshape((4, 3))
#
#     return -result[0].reshape((4, 3))
