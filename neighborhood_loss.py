import torch
import torch.nn as nn


class NeighborhoodLoss(nn.Module):
    def __init__(self, inherent_neighborhood=0.2592):
        super(NeighborhoodLoss, self).__init__()
        self.inherent_neighborhood = inherent_neighborhood

    def forward(self, predictions):  # (B, 100, 6)
        loss = 0.0
        for i in range(predictions.shape[0]):

            for j in range(predictions.shape[1]-1):  # [0, 98]
                loss += torch.sum((predictions[i][j] - predictions[i][j+1]) ** 2)

        loss = loss / predictions.shape[0] - self.inherent_neighborhood
        return loss


# from dataset import TraDataset
#
#
# tra_dataset = TraDataset('data/training')
# all_joint_angle_v = 0.0
#
# for i in range(len(tra_dataset)):
#     data = tra_dataset[i][1].numpy()  # (100, 6)
#
#     all_joint_angle_v += float((data[0][0] - data[1][0]) ** 2)
#
#     if i == 39999:
#         print('已经全部加和完成')
#
# print(all_joint_angle_v)  # 17.45305134985816, 17.45305134985816 / 40000 * 6 * 99 = 0.2591778125453937
