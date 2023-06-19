# -*- coding: utf-8 -*-

import numpy as np
import ikpy.chain
import h5py
import random
import math


def qr_calculation(robot, desire_x, desired_xd, actual_x, dis, hatd):
    alpha = 1
    qr_ = np.linalg.pinv(robot.jacob0(robot.q)) @ (desired_xd - alpha * (actual_x - desire_x) + dis - hatd)
    return qr_


def dis_observer(robot, qr_, desire_x, actual_x, omega, dt_):
    alpha = 1
    Lk = np.identity(6)
    omega += -Lk @ (robot.jacob0(robot.q) @ (robot.qd - qr_) - alpha * (actual_x - desire_x)) * dt_
    hatd = omega + Lk @ (actual_x - desire_x)
    return hatd, omega


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


NUM_TRAJECTORIES = 60000


NUM_JOINTS = 6
NUM_SAMPLES_TO_GENERATE = 100
dt = 1
robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/zhouyunlai/RAIKPR/assets/UR5/urdf/ur5_robot.urdf')

upper = []
lower = []
for i in range(1, len(robotic_arm.links) - 1):  # len(robotic_arm.links) = 8
    lower.append(robotic_arm.links[i].bounds[0])
    upper.append(robotic_arm.links[i].bounds[1])
upper = np.array(upper)
lower = np.array(lower)
# upper: [3.14159265, 3.14159265, 2.5, 3.14159265, 3.14159265, 3.14159265]
# lower: [-3.14159265, -3.14159265, -2.5, -3.14159265, -3.14159265, -3.14159265]


for trajectory in range(NUM_TRAJECTORIES):

    joint_0 = random.uniform(lower[0], upper[0])
    joint_0_direction = random.choice([-1, 1])
    if joint_0_direction == -1:
        max_joint_0_v = (joint_0 - lower[0]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_0_v = (upper[0] - joint_0) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_0_v = random.uniform(0, max_joint_0_v) * joint_0_direction

    joint_1 = random.uniform(lower[1], upper[1])
    joint_1_direction = random.choice([-1, 1])
    if joint_1_direction == -1:
        max_joint_1_v = (joint_1 - lower[1]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_1_v = (upper[1] - joint_1) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_1_v = random.uniform(0, max_joint_1_v) * joint_1_direction

    joint_2 = random.uniform(lower[2], upper[2])
    joint_2_direction = random.choice([-1, 1])
    if joint_2_direction == -1:
        max_joint_2_v = (joint_2 - lower[2]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_2_v = (upper[2] - joint_2) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_2_v = random.uniform(0, max_joint_2_v) * joint_2_direction

    joint_3 = random.uniform(lower[3], upper[3])
    joint_3_direction = random.choice([-1, 1])
    if joint_3_direction == -1:
        max_joint_3_v = (joint_3 - lower[3]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_3_v = (upper[3] - joint_3) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_3_v = random.uniform(0, max_joint_3_v) * joint_3_direction

    joint_4 = random.uniform(lower[4], upper[4])
    joint_4_direction = random.choice([-1, 1])
    if joint_4_direction == -1:
        max_joint_4_v = (joint_4 - lower[4]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_4_v = (upper[4] - joint_4) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_4_v = random.uniform(0, max_joint_4_v) * joint_4_direction

    joint_5 = random.uniform(lower[5], upper[5])
    joint_5_direction = random.choice([-1, 1])
    if joint_5_direction == -1:
        max_joint_5_v = (joint_5 - lower[5]) / (NUM_SAMPLES_TO_GENERATE * dt)
    else:
        max_joint_5_v = (upper[5] - joint_5) / (NUM_SAMPLES_TO_GENERATE * dt)
    joint_5_v = random.uniform(0, max_joint_5_v) * joint_5_direction

    init_joint_angles = [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5]
    joints_v = [joint_0_v, joint_1_v, joint_2_v, joint_3_v, joint_4_v, joint_5_v]

    results = []  # 存放eatv
    inputs = []  # 存放joint_angles
    for i in range(NUM_SAMPLES_TO_GENERATE):
        distance_traveled = [joint_v * (i * dt) for joint_v in joints_v]

        current_joint_angles = [
            init_joint_angles[k] + distance_traveled[k] for k in range(len(init_joint_angles))
        ]

        transformation_matrix = robotic_arm.forward_kinematics([0.0] + current_joint_angles + [0.0])
        euler_angles = rotationMatrixToEulerAngles(transformation_matrix[:3, :3])
        translation_vector = transformation_matrix[:3, 3]
        results.append(np.concatenate((euler_angles, translation_vector)))
        inputs.append(np.array(current_joint_angles))

    results = np.array(results)  # (100, 6), eatv
    inputs = np.array(inputs)  # (100, 6), joint_angles
    # 这二者行与行之间是对应的, 即FK(inputs[i]) -> results[i], 反之使用IK亦然

    train_dataset = h5py.File('ur5_tra{}.hdf5'.format(trajectory), 'w')
    train_dataset.create_dataset('results', data=results)
    train_dataset.create_dataset('inputs', data=inputs)
    train_dataset.close()
    print('生成并保存了第{}条轨迹'.format(trajectory))

# x = list(results[:, 3])
# y = list(results[:, 4])
# z = list(results[:, 5])
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()
