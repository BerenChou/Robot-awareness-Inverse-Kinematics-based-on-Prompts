import torch
import ikpy.chain
import numpy as np


# alpha = torch.tensor([torch.pi/2, 0, 0, torch.pi/2, -torch.pi/2, 0], dtype=torch.float32, requires_grad=True)
# a = torch.tensor([0, -0.425, -0.39225, 0, 0, 0], dtype=torch.float32, requires_grad=True)
# d = torch.tensor([0.089159, 0, 0, 0.10915, 0.09465, 0.0823], dtype=torch.float32, requires_grad=True)
# def h_t_m(alpha_i, a_i, d_i, ja_i):
#     return torch.tensor([
#         [torch.cos(ja_i), -torch.sin(ja_i) * torch.cos(alpha_i),  torch.sin(ja_i) * torch.sin(alpha_i),  a_i * torch.cos(ja_i)],
#         [torch.sin(ja_i),  torch.cos(ja_i) * torch.cos(alpha_i), -torch.cos(ja_i) * torch.sin(alpha_i),  a_i * torch.sin(ja_i)],
#         [0,                torch.sin(alpha_i),                    torch.cos(alpha_i),                    d_i],
#         [0,                0,                                     0,                                     1],
#     ], dtype=torch.float32, requires_grad=True)
#
# def forward_kinematics_UR5(_jas):
#     base = torch.tensor([
#         [-1,  0,  0,  0],
#         [ 0, -1,  0,  0],
#         [ 0,  0,  1,  0],
#         [ 0,  0,  0,  1],
#     ], dtype=torch.float32, requires_grad=True)
#
#     T01 = h_t_m(alpha[0], a[0], d[0], _jas[0])
#     T12 = h_t_m(alpha[1], a[1], d[1], _jas[1])
#     T23 = h_t_m(alpha[2], a[2], d[2], _jas[2])
#     T34 = h_t_m(alpha[3], a[3], d[3], _jas[3])
#     T45 = h_t_m(alpha[4], a[4], d[4], _jas[4])
#     T56 = h_t_m(alpha[5], a[5], d[5], _jas[5])
#
#     last = torch.tensor([
#         [ 0, -1,  0,  0],
#         [ 0,  0, -1,  0],
#         [ 1,  0,  0,  0],
#         [ 0,  0,  0,  1],
#     ], dtype=torch.float32, requires_grad=True)
#     return base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ last


# base = torch.tensor([
#         [-1,  0,  0,  0],
#         [ 0, -1,  0,  0],
#         [ 0,  0,  1,  0],
#         [ 0,  0,  0,  1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T01 = torch.tensor([
#         [torch.cos(_jas[0]),  0,  torch.sin(_jas[0]),  0],
#         [torch.sin(_jas[0]),  0, -torch.cos(_jas[0]),  0],
#         [0,                   1,  0,                   0.089159],
#         [0,                   0,  0,                   1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T12 = torch.tensor([
#         [torch.cos(_jas[1]), -torch.sin(_jas[1]),  0,  -0.425 * torch.cos(_jas[1])],
#         [torch.sin(_jas[1]),  torch.cos(_jas[1]),  0,  -0.425 * torch.sin(_jas[1])],
#         [0,                   0,                   1,   0],
#         [0,                   0,                   0,   1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T23 = torch.tensor([
#         [torch.cos(_jas[2]), -torch.sin(_jas[2]),  0, -0.39225 * torch.cos(_jas[2])],
#         [torch.sin(_jas[2]),  torch.cos(_jas[2]),  0, -0.39225 * torch.sin(_jas[2])],
#         [0,                   0,                   1,  0],
#         [0,                   0,                   0,  1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T34 = torch.tensor([
#         [torch.cos(_jas[3]),  0,  torch.sin(_jas[3]),  0],
#         [torch.sin(_jas[3]),  0, -torch.cos(_jas[3]),  0],
#         [0,                   1,  0,                   0.10915],
#         [0,                   0,  0,                   1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T45 = torch.tensor([
#         [torch.cos(_jas[4]),  0, -torch.sin(_jas[4]),  0],
#         [torch.sin(_jas[4]),  0,  torch.cos(_jas[4]),  0],
#         [0,                  -1,  0,                   0.09465],
#         [0,                   0,  0,                   1],
#     ], dtype=torch.float32, requires_grad=True)
#
# T56 = torch.tensor([
#         [torch.cos(_jas[5]), -torch.sin(_jas[5]),  0,  0],
#         [torch.sin(_jas[5]),  torch.cos(_jas[5]),  0,  0],
#         [0,                   0,                   1,  0.0823],
#         [0,                   0,                   0,  1],
#     ], dtype=torch.float32, requires_grad=True)
#
# last = torch.tensor([
#         [ 0, -1,  0,  0],
#         [ 0,  0, -1,  0],
#         [ 1,  0,  0,  0],
#         [ 0,  0,  0,  1],
#     ], dtype=torch.float32, requires_grad=True)


# _jas = torch.tensor([-1.974077, -0.11979904, -2.2081947, 0.45669574, 1.817192, 1.049178], dtype=torch.float32, requires_grad=True)
# # --base------------------------------------------------
# base = torch.tensor([
#     [-1,  0,  0,  0],
#     [ 0, -1,  0,  0],
#     [ 0,  0,  1,  0],
#     [ 0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
#
# # --T01------------------------------------------------
# T01_sin_index = torch.tensor([
#     [0,  0,  1,  0],
#     [1,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T01_sin = torch.sin(_jas[0]) * T01_sin_index
# T01_cos_index = torch.tensor([
#     [1,  0,  0,  0],
#     [0,  0, -1,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T01_cos = torch.cos(_jas[0]) * T01_cos_index
# T01_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  1,  0,  0.089159],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T01 = T01_sin + T01_cos + T01_number
#
# # --T12------------------------------------------------
# T12_sin_index = torch.tensor([
#     [0, -1,  0,  0],
#     [1,  0,  0,  -0.425],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T12_sin = torch.sin(_jas[1]) * T12_sin_index
# T12_cos_index = torch.tensor([
#     [1,  0,  0,  -0.425],
#     [0,  1,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T12_cos = torch.cos(_jas[1]) * T12_cos_index
# T12_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  1,  0],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T12 = T12_sin + T12_cos + T12_number
#
# # --T23------------------------------------------------
# T23_sin_index = torch.tensor([
#     [0, -1,  0,  0],
#     [1,  0,  0, -0.39225],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T23_sin = torch.sin(_jas[2]) * T23_sin_index
# T23_cos_index = torch.tensor([
#     [1,  0,  0, -0.39225],
#     [0,  1,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T23_cos = torch.cos(_jas[2]) * T23_cos_index
# T23_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  1,  0],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T23 = T23_sin + T23_cos + T23_number
#
# # --T34------------------------------------------------
# T34_sin_index = torch.tensor([
#     [0,  0,  1,  0],
#     [1,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T34_sin = torch.sin(_jas[3]) * T34_sin_index
# T34_cos_index = torch.tensor([
#     [1,  0,  0,  0],
#     [0,  0, -1,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T34_cos = torch.cos(_jas[3]) * T34_cos_index
# T34_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  1,  0,  0.10915],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T34 = T34_sin + T34_cos + T34_number
#
# # --T45------------------------------------------------
# T45_sin_index = torch.tensor([
#     [0,  0, -1,  0],
#     [1,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T45_sin = torch.sin(_jas[4]) * T45_sin_index
# T45_cos_index = torch.tensor([
#     [1,  0,  0,  0],
#     [0,  0,  1,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T45_cos = torch.cos(_jas[4]) * T45_cos_index
# T45_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0, -1,  0,  0.09465],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T45 = T45_sin + T45_cos + T45_number
#
# # --T56------------------------------------------------
# T56_sin_index = torch.tensor([
#     [0, -1,  0,  0],
#     [1,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T56_sin = torch.sin(_jas[5]) * T56_sin_index
# T56_cos_index = torch.tensor([
#     [1,  0,  0,  0],
#     [0,  1,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
# ], dtype=torch.float32, requires_grad=True)
# T56_cos = torch.cos(_jas[5]) * T56_cos_index
# T56_number = torch.tensor([
#     [0,  0,  0,  0],
#     [0,  0,  0,  0],
#     [0,  0,  1,  0.0823],
#     [0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
# T56 = T56_sin + T56_cos + T56_number
#
# # --last------------------------------------------------
# last = torch.tensor([
#     [ 0, -1,  0,  0],
#     [ 0,  0, -1,  0],
#     [ 1,  0,  0,  0],
#     [ 0,  0,  0,  1],
# ], dtype=torch.float32, requires_grad=True)
#
# output = base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ last
# print(output)
# # tensor([[-0.1117, -0.7836,  0.6112, -0.0041],
# #         [ 0.3598, -0.6052, -0.7102, -0.2367],
# #         [ 0.9263,  0.1405,  0.3495,  0.5293],
# #         [ 0.0000,  0.0000,  0.0000,  1.0000]])
# output_sum = torch.sum(output)
# output_sum.backward()
# print(_jas.grad)


# def forward_kinematics_UR5(_jas):
#     # --base------------------------------------------------
#     base = torch.tensor([
#         [-1, 0, 0, 0],
#         [0, -1, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     # --T01------------------------------------------------
#     T01_sin_index = torch.tensor([
#         [0, 0, 1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T01_sin = torch.sin(_jas[0]) * T01_sin_index
#
#     T01_cos_index = torch.tensor([
#         [1, 0, 0, 0],
#         [0, 0, -1, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T01_cos = torch.cos(_jas[0]) * T01_cos_index
#
#     T01_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 1, 0, 0.089159],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T01 = T01_sin + T01_cos + T01_number
#
#     # --T12------------------------------------------------
#     T12_sin_index = torch.tensor([
#         [0, -1, 0, 0],
#         [1, 0, 0, -0.425],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T12_sin = torch.sin(_jas[1]) * T12_sin_index
#
#     T12_cos_index = torch.tensor([
#         [1, 0, 0, -0.425],
#         [0, 1, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T12_cos = torch.cos(_jas[1]) * T12_cos_index
#
#     T12_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T12 = T12_sin + T12_cos + T12_number
#
#     # --T23------------------------------------------------
#     T23_sin_index = torch.tensor([
#         [0, -1, 0, 0],
#         [1, 0, 0, -0.39225],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T23_sin = torch.sin(_jas[2]) * T23_sin_index
#
#     T23_cos_index = torch.tensor([
#         [1, 0, 0, -0.39225],
#         [0, 1, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T23_cos = torch.cos(_jas[2]) * T23_cos_index
#
#     T23_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T23 = T23_sin + T23_cos + T23_number
#
#     # --T34------------------------------------------------
#     T34_sin_index = torch.tensor([
#         [0, 0, 1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T34_sin = torch.sin(_jas[3]) * T34_sin_index
#
#     T34_cos_index = torch.tensor([
#         [1, 0, 0, 0],
#         [0, 0, -1, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T34_cos = torch.cos(_jas[3]) * T34_cos_index
#
#     T34_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 1, 0, 0.10915],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T34 = T34_sin + T34_cos + T34_number
#
#     # --T45------------------------------------------------
#     T45_sin_index = torch.tensor([
#         [0, 0, -1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T45_sin = torch.sin(_jas[4]) * T45_sin_index
#
#     T45_cos_index = torch.tensor([
#         [1, 0, 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T45_cos = torch.cos(_jas[4]) * T45_cos_index
#
#     T45_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, -1, 0, 0.09465],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T45 = T45_sin + T45_cos + T45_number
#
#     # --T56------------------------------------------------
#     T56_sin_index = torch.tensor([
#         [0, -1, 0, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T56_sin = torch.sin(_jas[5]) * T56_sin_index
#
#     T56_cos_index = torch.tensor([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#     ], dtype=torch.float32)
#     T56_cos = torch.cos(_jas[5]) * T56_cos_index
#
#     T56_number = torch.tensor([
#         [0, 0, 0, 0],
#         [0, 0, 0, 0],
#         [0, 0, 1, 0.0823],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     T56 = T56_sin + T56_cos + T56_number
#
#     # --last------------------------------------------------
#     last = torch.tensor([
#         [0, -1, 0, 0],
#         [0, 0, -1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 1],
#     ], dtype=torch.float32)
#
#     return base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ last
#
#
# ur5 = ikpy.chain.Chain.from_urdf_file('/home/zulipeng/zyl/RAIKPR/data_without_mutation/assets/UR5/urdf/ur5_robot.urdf')
# jas_npy = np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/jas_npy_6w_samples.npy').astype(np.float32)  # (6w, 6)
# wrong_list = []
# for i in range(jas_npy.shape[0]):
#     jas = jas_npy[i]
#
#     transformation_matrix_ikpy = ur5.forward_kinematics([0.0] + list(jas) + [0.0])
#     transformation_matrix_def = forward_kinematics_UR5(
#         torch.from_numpy(jas)
#     ).cpu().numpy()
#
#     if np.sum(abs(transformation_matrix_ikpy - transformation_matrix_def)) > 0.01:
#         wrong_list.append(i)
#
# print('检查了{}个'.format(i))
# print(wrong_list)
# print(len(wrong_list))
