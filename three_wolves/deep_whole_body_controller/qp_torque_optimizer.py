import numpy as np
import quadprog
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

ACC_WEIGHT = np.array([1., 1., 10., 1., 1., 1.])


def compute_mass_matrix(robot_mass, robot_inertia, tip_positions, cube_yaw=None):
    # todo: consider roll and pitch ?
    # yaw = cube_yaw  # Set yaw to 0 for now as all commands are local.
    # rot_z = np.array([[np.cos(cube_yaw), np.sin(cube_yaw), 0.],
    #                   [-np.sin(cube_yaw), np.cos(cube_yaw), 0.],
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
                              contact_face_ids,
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

    face_dict = {
        0: (
            [1, friction_coef, 0],
            [-1, friction_coef, 0],
            [0, friction_coef, 1],
            [0, friction_coef, -1],
        ),
        1: (
            [-friction_coef, 1, 0],
            [-friction_coef, -1, 0],
            [-friction_coef, 0, 1],
            [-friction_coef, 0, -1],
        ),
        2: (
            [1, -friction_coef, 0],
            [-1, -friction_coef, 0],
            [0, -friction_coef, 1],
            [0, -friction_coef, -1],
        ),
        3: (
            [friction_coef, 1, 0],
            [friction_coef, -1, 0],
            [friction_coef, 0, 1],
            [friction_coef, 0, -1],
        ),

    }
    # finger 0
    A[6, :] = face_dict[contact_face_ids[0]][0] + [0] * 6
    A[7, :] = face_dict[contact_face_ids[0]][1] + [0] * 6
    A[8, :] = face_dict[contact_face_ids[0]][2] + [0] * 6
    A[9, :] = face_dict[contact_face_ids[0]][3] + [0] * 6
    # finger 1
    A[10, :] = [0] * 3 + face_dict[contact_face_ids[1]][0] + [0] * 3
    A[11, :] = [0] * 3 + face_dict[contact_face_ids[1]][1] + [0] * 3
    A[12, :] = [0] * 3 + face_dict[contact_face_ids[1]][2] + [0] * 3
    A[13, :] = [0] * 3 + face_dict[contact_face_ids[1]][3] + [0] * 3
    # finger 2
    A[14, :] = [0] * 6 + face_dict[contact_face_ids[2]][0]
    A[15, :] = [0] * 6 + face_dict[contact_face_ids[2]][1]
    A[16, :] = [0] * 6 + face_dict[contact_face_ids[2]][2]
    A[17, :] = [0] * 6 + face_dict[contact_face_ids[2]][3]

    return A.T, lb


def compute_objective_matrix(mass_matrix, desired_acc, acc_weight, reg_weight):
    g = np.array([0., 0., -9.81, 0., 0., 0.])
    # g = np.array([0., 0., 0, 0., 0., 0.])
    Q = np.diag(acc_weight)
    R = np.ones(9) * reg_weight

    quad_term = mass_matrix.T.dot(Q).dot(mass_matrix) + R
    linear_term = 1 * (-g + desired_acc).T.dot(Q).dot(mass_matrix)
    return quad_term, linear_term


def compute_contact_force(robot_mass,
                          robot_inertia,
                          contact_face_ids,
                          contact_position,
                          desired_acc,
                          acc_weight=ACC_WEIGHT,
                          reg_weight=1e-6,
                          friction_coef=0.8,
                          f_min_ratio=-10,
                          f_max_ratio=10):

    mass_matrix = compute_mass_matrix(robot_mass, robot_inertia, contact_position)
    G, a = compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
                                    reg_weight)
    C, b = compute_constraint_matrix(robot_mass, friction_coef, contact_face_ids,
                                     f_min_ratio, f_max_ratio)
    G += 1e-4 * np.eye(9)

    result = quadprog.solve_qp(G, a, C, b)

    return result[0].reshape((3, 3))
