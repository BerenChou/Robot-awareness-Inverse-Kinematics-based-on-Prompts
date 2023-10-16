import numpy as np
import ikpy.chain
import h5py
import mmcv
import math
import os.path as osp
import torch


# # hdf5转npy
# trajectories_names_list = []
# for trajectory_name in mmcv.scandir('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training', '.hdf5'):
#     trajectories_names_list.append(trajectory_name)
#
# eatv_list = []
# ja_list = []
# total_number = 0
# for i in range(len(trajectories_names_list)):
#     trajectory = h5py.File(osp.join('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training', trajectories_names_list[i]), 'r')
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
# np.save('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/eatvs_npy_40w.npy', eatvs_npy)
# np.save('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/jas_npy_40w.npy', jas_npy)


# # 读取npy文件并验证
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
# eatvs_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/eatvs_npy_40w.npy')
# jas_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/jas_npy_40w.npy')
# print(eatvs_npy.shape)  # (400000, 100, 6)
# print(eatvs_npy.dtype)  # float64
# print(jas_npy.shape)  # (400000, 100, 6)
# print(jas_npy.dtype)  # float64
#
# robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/zulipeng/zyl/RAIKPR/data_without_mutation/assets/UR5/urdf/ur5_robot.urdf')
#
# transformation_matrix = robotic_arm.forward_kinematics([0] + list(jas_npy[390000][50]) + [0])
# euler_angles = rotationMatrixToEulerAngles(transformation_matrix[:3, :3])
# translation_vector = transformation_matrix[:3, 3]
#
# duizhao = list(eatvs_npy[390000][50])
#
# print('debugger')


# eatvs_npy1 = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/eatvs_npy_40w.npy').reshape(-1, 6)
# # max: [3.14158891 1.56858005 3.14158663 0.9486867  0.9486645  1.03169828]
# # min: [-3.14158291 -1.5678904  -3.14158191 -0.94717112 -0.94884787 -0.85334423]
#
# eatvs_npy2 = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/validation/eatvs_npy.npy').reshape(-1, 6)
# # max: [3.14157793 1.56530594 3.1413108  0.9442901  0.94244734 1.03058602]
# # min: [-3.14088957 -1.5565625  -3.14158695 -0.93854148 -0.94470803 -0.85235611]
#
# eatvs_npy3 = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/testing/eatvs_npy.npy').reshape(-1, 6)
# # max: [3.14158826 1.55225135 3.14154608 0.93857907 0.94655879 1.03069814]
# # min: [-3.14114441 -1.56717048 -3.14156884 -0.93996165 -0.94215589 -0.85319351]
#
# print(np.max(eatvs_npy1, axis=0))
# print(np.max(eatvs_npy2, axis=0))
# print(np.max(eatvs_npy3, axis=0))


# tra_length=100
# mask = abs(
#     torch.arange(1, tra_length + 1).reshape(1, tra_length).repeat(tra_length, 1) -
#     torch.arange(1, tra_length + 1).reshape(tra_length, 1).repeat(1, tra_length)
# ) < 11  # 1 + 2 * (i - 1)个位置将会被attend, 即当i = 11时, 21个位置将会被attend
# mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).numpy()
# print('debugger')


# tra_length = 20
# mask1 = (torch.triu(torch.ones(tra_length, tra_length)) == 1).transpose(0, 1)
# mask1 = mask1.float().masked_fill(mask1 == 0, float('-inf')).masked_fill(mask1 == 1, float(0.0)).numpy()
# mask2 = abs(
#     torch.arange(1, tra_length + 1).reshape(1, tra_length).repeat(tra_length, 1) -
#     torch.arange(1, tra_length + 1).reshape(tra_length, 1).repeat(1, tra_length)
# ) < 4  # 1 + 2 * (i - 1)个位置将会被attend, 即当i = 11时, 21个位置将会被attend
# mask2 = mask2.float().masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, float(0.0)).numpy()
# attn_mask = mask1 + mask2
# print('debugger')


# eatvs_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/eatvs_npy.npy')[:, :, 5]  # (60000, 100)
# # max: [3.14149875 1.56183109 3.14120492 0.9425432  0.94796457 1.03054962]
# index_list = []
# for i in range(eatvs_npy.shape[0]):
#     if any(eatvs_npy[i] > 1.03):
#         index_list.append(i)
# print(index_list)
