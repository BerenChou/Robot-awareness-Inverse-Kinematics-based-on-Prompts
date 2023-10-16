from torch.utils.data import Dataset
import torch
import numpy as np


# 用于Transformer模型的Dataset
class Transformer_Dataset(Dataset):
    def __init__(self, eatvs_npy_path, jas_npy_path, normalization_value_for_radian):
        self.eatvs_npy = torch.from_numpy(
            np.load(eatvs_npy_path).astype(np.float32)
        )
        # shape为(轨迹数量, 单条轨迹长度, 6), 最后一个维度的6个值中, 前3个为euler angles(ea), 后3个为translation vector(tv)

        self.jas_npy = torch.from_numpy(
            np.load(jas_npy_path).astype(np.float32)
        )
        # shape为(轨迹数量, 单条轨迹长度, 6), 最后一个维度的6个值对应于UR5机械臂6个关节的弧度制角度值

        if normalization_value_for_radian != 1.0:  # 即当其为torch.pi时
            self.eatvs_npy[:, :, :3] = self.eatvs_npy[:, :, :3] / normalization_value_for_radian
            self.jas_npy = self.jas_npy / normalization_value_for_radian

    def __getitem__(self, index):
        return self.eatvs_npy[index], self.jas_npy[index]

    def __len__(self):
        assert self.eatvs_npy.shape == self.jas_npy.shape
        return self.eatvs_npy.shape[0]


# following Nerf
def pe(eatv):  # (6,)
    ea_0= [
        [np.sin(2 ** i * np.pi * eatv[0]), np.cos(2 ** i * np.pi * eatv[0])] for i in range(4)
    ]
    ea_1 = [
        [np.sin(2 ** i * np.pi * eatv[1]), np.cos(2 ** i * np.pi * eatv[1])] for i in range(4)
    ]
    ea_2 = [
        [np.sin(2 ** i * np.pi * eatv[2]), np.cos(2 ** i * np.pi * eatv[2])] for i in range(4)
    ]

    tv_0 = [
        [np.sin(2 ** i * np.pi * eatv[3]), np.cos(2 ** i * np.pi * eatv[3])] for i in range(10)
    ]
    tv_1 = [
        [np.sin(2 ** i * np.pi * eatv[4]), np.cos(2 ** i * np.pi * eatv[4])] for i in range(10)
    ]
    tv_2 = [
        [np.sin(2 ** i * np.pi * eatv[5]), np.cos(2 ** i * np.pi * eatv[5])] for i in range(10)
    ]

    ea_0 = [item for sublist in ea_0 for item in sublist]
    ea_1 = [item for sublist in ea_1 for item in sublist]
    ea_2 = [item for sublist in ea_2 for item in sublist]
    tv_0 = [item for sublist in tv_0 for item in sublist]
    tv_1 = [item for sublist in tv_1 for item in sublist]
    tv_2 = [item for sublist in tv_2 for item in sublist]

    ea_0 = np.array(ea_0)
    ea_1 = np.array(ea_1)
    ea_2 = np.array(ea_2)
    tv_0 = np.array(tv_0)
    tv_1 = np.array(tv_1)
    tv_2 = np.array(tv_2)

    return np.concatenate([ea_0, ea_1, ea_2, tv_0, tv_1, tv_2])  # (84,), 20 * 3 + 8 * 3


# 用于MLP模型的Dataset
class MLP_Dataset(Dataset):
    def __init__(self, eatvs_npy_path, jas_npy_path, normalization_value_for_radian):

        # # 借鉴Nerf, 对输入进行编码, 但没什么效果
        # self.eatvs_npy = np.load(eatvs_npy_path)  # shape为(样本数量, 6), 最后一维6个值中, 前3为ea, 后3为tv
        # self.eatvs_npy_pe = np.random.rand(self.eatvs_npy.shape[0], 84)
        # for i in range(self.eatvs_npy.shape[0]):
        #     self.eatvs_npy_pe[i] = pe(self.eatvs_npy[i])
        # self.eatvs_npy = torch.from_numpy(self.eatvs_npy_pe.astype(np.float32))
        # # shape为(样本数量, 84)

        # 不对输入进行编码
        self.eatvs_npy = torch.from_numpy(
            np.load(eatvs_npy_path).astype(np.float32)
        )  # shape为(样本数量, 6), 最后一维6个值中, 前3为ea, 后3为tv

        self.jas_npy = torch.from_numpy(
            np.load(jas_npy_path).astype(np.float32)
        )  # shape为(样本数量, 6), 最后一个维度的6个值对应于UR5机械臂6个关节的弧度制角度值

        if normalization_value_for_radian != 1.0:  # 即当其为torch.pi时
            self.eatvs_npy[:, :3] = self.eatvs_npy[:, :3] / normalization_value_for_radian
            self.jas_npy = self.jas_npy / normalization_value_for_radian

    def __getitem__(self, index):
        return self.eatvs_npy[index], self.jas_npy[index]

    def __len__(self):
        assert self.eatvs_npy.shape == self.jas_npy.shape
        return self.eatvs_npy.shape[0]


# from einops import rearrange
# class Chunk_Dataset(Dataset):
#     def __init__(self, eatvs_npy_path, jas_npy_path):
#         self.eatvs_npy = torch.from_numpy(
#             np.load(eatvs_npy_path).astype(np.float32)
#         )
#         self.eatvs_npy = rearrange(self.eatvs_npy, 'b (l1 l2) j -> b l1 (l2 j)', l1=20)
#
#         self.jas_npy = torch.from_numpy(
#             np.load(jas_npy_path).astype(np.float32)
#         )
#         self.jas_npy = rearrange(self.jas_npy, 'b (l1 l2) j -> b l1 (l2 j)', l1=20)
#
#     def __getitem__(self, index):
#         return self.eatvs_npy[index], self.jas_npy[index]
#
#     def __len__(self):
#         return self.eatvs_npy.shape[0]
