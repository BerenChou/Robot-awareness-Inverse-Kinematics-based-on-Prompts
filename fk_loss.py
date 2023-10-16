import torch
import torch.nn as nn


def rotationMatrixToEulerAngles(R):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2]).unsqueeze(0)
        y = torch.atan2(-R[2, 0], sy).unsqueeze(0)
        z = torch.atan2(R[1, 0], R[0, 0]).unsqueeze(0)
    else:
        x = torch.atan2(-R[1, 2], R[1, 1]).unsqueeze(0)
        y = torch.atan2(-R[2, 0], sy).unsqueeze(0)
        z = torch.zeros(1, dtype=torch.float32, requires_grad=False).unsqueeze(0).to(y.device)
    return torch.cat((x, y, z))


class FKLoss(nn.Module):
    def __init__(self, device, loss_name='L1Loss'):
        super(FKLoss, self).__init__()

        if loss_name == 'L1Loss':
            self.standard_loss = nn.L1Loss()
        elif loss_name == 'MSELoss':
            self.standard_loss = nn.MSELoss()

        # --base------------------------------------------------
        self.base = torch.tensor([
            [-1,  0, 0, 0],
            [ 0, -1, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T01------------------------------------------------
        self.T01_sin_index = torch.tensor([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T01_cos_index = torch.tensor([
            [1, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 0,  0, 0],
            [0, 0,  0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T01_number = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0.089159],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T12------------------------------------------------
        self.T12_sin_index = torch.tensor([
            [0, -1, 0,  0],
            [1,  0, 0, -0.425],
            [0,  0, 0,  0],
            [0,  0, 0,  0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T12_cos_index = torch.tensor([
            [1, 0, 0, -0.425],
            [0, 1, 0,  0],
            [0, 0, 0,  0],
            [0, 0, 0,  0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T12_number = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T23------------------------------------------------
        self.T23_sin_index = torch.tensor([
            [0, -1, 0,  0],
            [1,  0, 0, -0.39225],
            [0,  0, 0,  0],
            [0,  0, 0,  0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T23_cos_index = torch.tensor([
            [1, 0, 0, -0.39225],
            [0, 1, 0,  0],
            [0, 0, 0,  0],
            [0, 0, 0,  0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T23_number = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T34------------------------------------------------
        self.T34_sin_index = torch.tensor([
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T34_cos_index = torch.tensor([
            [1, 0,  0, 0],
            [0, 0, -1, 0],
            [0, 0,  0, 0],
            [0, 0,  0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T34_number = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0.10915],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T45------------------------------------------------
        self.T45_sin_index = torch.tensor([
            [0, 0, -1, 0],
            [1, 0,  0, 0],
            [0, 0,  0, 0],
            [0, 0,  0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T45_cos_index = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T45_number = torch.tensor([
            [0,  0, 0, 0],
            [0,  0, 0, 0],
            [0, -1, 0, 0.09465],
            [0,  0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --T56------------------------------------------------
        self.T56_sin_index = torch.tensor([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 0, 0],
            [0,  0, 0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T56_cos_index = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=torch.float32, device=device, requires_grad=False)
        self.T56_number = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0.0823],
            [0, 0, 0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

        # --last------------------------------------------------
        self.last = torch.tensor([
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [1,  0,  0, 0],
            [0,  0,  0, 1],
        ], dtype=torch.float32, device=device, requires_grad=False)

    def forward(self, pred_ja_batch, gt_eatv_batch):  # (B, 100, 6)
        pred_ja_batch = pred_ja_batch.reshape(-1, 6)
        gt_eatv_batch = gt_eatv_batch.reshape(-1, 6)

        pred_eatv_list = []
        for i in range(pred_ja_batch.shape[0]):
            T01_sin = torch.sin(pred_ja_batch[i][0]) * self.T01_sin_index
            T01_cos = torch.cos(pred_ja_batch[i][0]) * self.T01_cos_index
            T01 = T01_sin + T01_cos + self.T01_number

            T12_sin = torch.sin(pred_ja_batch[i][1]) * self.T12_sin_index
            T12_cos = torch.cos(pred_ja_batch[i][1]) * self.T12_cos_index
            T12 = T12_sin + T12_cos + self.T12_number

            T23_sin = torch.sin(pred_ja_batch[i][2]) * self.T23_sin_index
            T23_cos = torch.cos(pred_ja_batch[i][2]) * self.T23_cos_index
            T23 = T23_sin + T23_cos + self.T23_number

            T34_sin = torch.sin(pred_ja_batch[i][3]) * self.T34_sin_index
            T34_cos = torch.cos(pred_ja_batch[i][3]) * self.T34_cos_index
            T34 = T34_sin + T34_cos + self.T34_number

            T45_sin = torch.sin(pred_ja_batch[i][4]) * self.T45_sin_index
            T45_cos = torch.cos(pred_ja_batch[i][4]) * self.T45_cos_index
            T45 = T45_sin + T45_cos + self.T45_number

            T56_sin = torch.sin(pred_ja_batch[i][5]) * self.T56_sin_index
            T56_cos = torch.cos(pred_ja_batch[i][5]) * self.T56_cos_index
            T56 = T56_sin + T56_cos + self.T56_number

            h_t_m = self.base @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ self.last

            eatv = torch.concat(
                (rotationMatrixToEulerAngles(h_t_m[:3, :3]), h_t_m[:3, 3])
            )

            pred_eatv_list.append(eatv)

        pred_eatv_batch = torch.stack(pred_eatv_list)

        return self.standard_loss(pred_eatv_batch, gt_eatv_batch)


# if __name__ == '__main__':
#     import numpy as np
#     device = torch.device('cuda:0')
#
#     jas_data = torch.from_numpy(
#         np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/jas_npy_6w.npy')[:512]
#     ).to(device).requires_grad_(True)
#
#     eatvs_data = torch.from_numpy(
#         np.load('/home/zulipeng/zyl/RAIKPR/data_without_mutation/training/eatvs_npy_40w.npy')[200000:200512]
#     ).to(device)
#
#     fkloss = FKLoss(device=device)
#     output = fkloss(jas_data, eatvs_data)
#     print(output)
#
#     output.backward()
#     print(jas_data.grad.shape)
