import numpy as np
import ikpy.chain
import h5py
import mmcv
import math
import os.path as osp


# # hdf5转npy
# trajectories_names_list = []
# for trajectory_name in mmcv.scandir('/home/zulipeng/zyl/RAIKPR/data_without_mutation/testing', '.hdf5'):
#     trajectories_names_list.append(trajectory_name)
#
# eatv_list = []
# ja_list = []
# total_number = 0
# for i in range(len(trajectories_names_list)):
#     trajectory = h5py.File(osp.join('/home/zulipeng/zyl/RAIKPR/data_without_mutation/testing', trajectories_names_list[i]), 'r')
#
#     # eatv: euler angles & translation vectors
#     # ja: joint angles
#     eatv = trajectory.get('results')[:]  # [100, 6]
#     eatv_list.append(eatv)
#
#     ja = trajectory.get('inputs')[:]  # [100, 6]
#     ja_list.append(ja)
#
#     total_number += 1
# print(f'一共{total_number}个trajectories')
#
# eatvs_npy = np.array(eatv_list)
# jas_npy = np.array(ja_list)
#
# print(eatvs_npy.shape)
# print(eatvs_npy.dtype)
# print(jas_npy.shape)
# print(jas_npy.dtype)
#
# np.save('/home/zulipeng/zyl/RAIKPR/data_without_mutation/testing/eatvs_npy.npy', eatvs_npy)
# np.save('/home/zulipeng/zyl/RAIKPR/data_without_mutation/testing/jas_npy.npy', jas_npy)



# # 读取npy文件并验证
# def qr_calculation(robot, desire_x, desired_xd, actual_x, dis, hatd):
#     alpha = 1
#     qr = np.linalg.pinv(robot.jacob0(robot.q)) @ (desired_xd - alpha * (actual_x - desire_x) + dis - hatd)
#     return qr
#
#
# def dis_observer(robot, qr, desire_x, actual_x, omega, _dt):
#     alpha = 1
#     Lk = np.identity(6)
#     omega += -Lk @ (robot.jacob0(robot.q) @ (robot.qd - qr) - alpha * (actual_x - desire_x)) * _dt
#     hatd = omega + Lk @ (actual_x - desire_x)
#     return hatd, omega
#
#
# def rotationMatrixToEulerAngles(R):
#     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6
#     if not singular:
#         x = math.atan2(R[2, 1], R[2, 2])
#         y = math.atan2(-R[2, 0], sy)
#         z = math.atan2(R[1, 0], R[0, 0])
#     else:
#         x = math.atan2(-R[1, 2], R[1, 1])
#         y = math.atan2(-R[2, 0], sy)
#         z = 0
#     return np.array([x, y, z])
#
# eatvs_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/validation/eatvs_npy.npy')
# jas_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/validation/jas_npy.npy')
# print(eatvs_npy.shape)  # (60000, 100, 6)
# print(eatvs_npy.dtype)  # float64
# print(jas_npy.shape)  # (60000, 100, 6)
# print(jas_npy.dtype)  # float64
#
# robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/zulipeng/zyl/RAIKPR/assets/UR5/urdf/ur5_robot.urdf')
#
# transformation_matrix = robotic_arm.forward_kinematics([0] + list(jas_npy[10000][50]) + [0])
# euler_angles = rotationMatrixToEulerAngles(transformation_matrix[:3, :3])
# translation_vector = transformation_matrix[:3, 3]
#
# duizhao = list(eatvs_npy[10000][50])
#
# print('debugger')



# # 将trajectory中的translation vectors画在三维坐标系中
# x = list(eatv[:, 3])
# y = list(eatv[:, 4])
# z = list(eatv[:, 5])
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
