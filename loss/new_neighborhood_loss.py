import torch
import torch.nn as nn


class NeighborhoodLoss(nn.Module):
    def __init__(self):
        super(NeighborhoodLoss, self).__init__()

    def forward(self, ja_predictions):  # (B, 100, 6)
        ja_predictions_except_last_row = ja_predictions[:, :-1]
        ja_predictions_except_first_row = ja_predictions[:, 1:]
        return abs(torch.mean(abs(ja_predictions_except_first_row - ja_predictions_except_last_row)) - 0.0129632)


# data_without_mutation_jas = np.load('data_without_mutation/training/jas_npy.npy')
# ja_predictions_except_last_row = data_without_mutation_jas[:, :-1]
# ja_predictions_except_first_row = data_without_mutation_jas[:, 1:]
# print(np.mean(abs(ja_predictions_except_first_row - ja_predictions_except_last_row)))  # 0.0129632
