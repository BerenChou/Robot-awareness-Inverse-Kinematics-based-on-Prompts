import torch
import torch.nn as nn
import random


class NeighborhoodLoss(nn.Module):
    def __init__(self, sub_trajectory_length, inherent_neighborhood):
        super(NeighborhoodLoss, self).__init__()
        self.sub_trajectory_length = sub_trajectory_length
        self.inherent_neighborhood = inherent_neighborhood

    def forward(self, predictions):  # (B, 100, 6)
        batch_size = predictions.shape[0]

        start_timestamp = random.randint(0, predictions.shape[1] - 1 - self.sub_trajectory_length)  # 0~79里面随机选一个数

        loss = 0.0
        for i in range(batch_size):
            for j in range(start_timestamp, start_timestamp + self.sub_trajectory_length):
                loss += torch.sum(
                    (predictions[i][j] - predictions[i][j+1]) ** 2
                )
        # torch.sum()在6个关节上加和. 两个for循环, 第一个在batch上加和, 第二个在sub_trajectory_length个interval上加和

        loss = loss / batch_size - self.inherent_neighborhood
        return loss


# from dataset import TraDataset
#
#
# tra_dataset = TraDataset('data/training')
#
# all_joint_angle_v = 0.0
#
# for i in range(len(tra_dataset)):
#     ja = tra_dataset[i][1]  # (100, 6)
#
#     all_joint_angle_v += float(
#         torch.sum((ja[0] - ja[1]) ** 2) * 20
#     )
#
#     if i == 59999:
#         print('已经全部加和完成')
#
# print(all_joint_angle_v / 60000)  # 0.049172985116200045
