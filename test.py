import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Transformer_Dataset
import ikpy.chain
import numpy as np
import ikpy.chain
import math
import matplotlib.pyplot as plt


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


def test(model):
    dataset = Transformer_Dataset(eatvs_data_source, jas_data_source, normalization_value_for_radian)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True,
                            drop_last=True, persistent_workers=True)

    loss_f = nn.L1Loss()

    with torch.no_grad():
        model.eval()

        one_epoch_eatv2ja_loss = []
        one_epoch_input_eatv = []
        one_epoch_pred_ja = []

        for _, (eatv_batch, ja_batch) in enumerate(dataloader):
            eatv_batch = eatv_batch.to(device)
            ja_batch = ja_batch.to(device)

            pred_ja_batch = model(eatv_batch)
            eatv2ja_loss = loss_f(pred_ja_batch, ja_batch)

            one_epoch_eatv2ja_loss.append(eatv2ja_loss.item())
            one_epoch_input_eatv.append(eatv_batch)
            one_epoch_pred_ja.append(pred_ja_batch)

        eatv2ja_loss = sum(one_epoch_eatv2ja_loss) / len(one_epoch_eatv2ja_loss)
        print("eatv2ja_loss = {:3f}.".format(eatv2ja_loss))

        all_input_eatv = torch.concat(one_epoch_input_eatv, dim=0)
        all_pred_ja = torch.concat(one_epoch_pred_ja, dim=0)

        robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/zulipeng/zyl/RAIKPR/data_without_mutation/assets/UR5/urdf/ur5_robot.urdf')
        for i in range(all_pred_ja.shape[0]):
            for j in range(all_pred_ja.shape[1]):
                transformation_matrix = robotic_arm.forward_kinematics(
                    [0.0] + [float(value) for value in all_pred_ja[i][j]] + [0.0]
                )
                euler_angles = list(rotationMatrixToEulerAngles(transformation_matrix[:3, :3]))
                translation_vector = list(transformation_matrix[:3, 3])
                all_pred_ja[i][j] = torch.tensor(euler_angles + translation_vector)
        all_pred_eatv = all_pred_ja

        difference_between_pred_eatv_and_gt_eatv = torch.mean(
            abs(
                (all_input_eatv - all_pred_eatv).reshape(-1, 6)
            ), dim=0)
        difference_between_pred_eatv_and_gt_eatv[:3] = difference_between_pred_eatv_and_gt_eatv[:3] * normalization_value_for_radian * 57.3
        print(difference_between_pred_eatv_and_gt_eatv)

        np.save('processed_pre_eatv_and_input_eatv/' + which_set + '_all_input_eatv.npy', all_input_eatv.cpu().numpy())
        np.save('processed_pre_eatv_and_input_eatv/' + which_set + '_all_pred_eatv.npy', all_pred_eatv.cpu().numpy())


# 加载模型权重, 在某个集合上测试, 汇报指标, 并保存1.预测出来的关节角使用ikpy库求得的FK的值(eatv) 2.GT的eatv. 1和2顺序相同, 可以在下面画图函数里面绘制图像.
if __name__ == '__main__':

    which_set = 'training'
    eatvs_data_source = 'data_without_mutation/' + which_set + '/eatvs_npy.npy'
    jas_data_source = 'data_without_mutation/' + which_set + '/jas_npy.npy'
    normalization_value_for_radian = 1.0
    device = torch.device('cuda:0')

    ik_transformer = torch.load('/home/zulipeng/zyl/RAIKPR/saved_ckpt/2023-07-04-14:28_eatv2jaloss0.485595_epoch745.pth').to(device)
    test(ik_transformer)


# # 将all_input_eatv与all_pred_eatv中的translation vectors画在三维坐标系中并保存到trajectory_images文件夹
# if __name__ == '__main__':
#
#     which_set = 'training'
#     all_input_eatv = np.load('processed_pre_eatv_and_input_eatv/' + which_set + '_all_input_eatv.npy')
#     all_pred_eatv = np.load('processed_pre_eatv_and_input_eatv/' + which_set + '_all_pred_eatv.npy')
#
#     for i in range(all_input_eatv.shape[0]):
#         x = list(all_input_eatv[i, :, 3])
#         y = list(all_input_eatv[i, :, 4])
#         z = list(all_input_eatv[i, :, 5])
#
#         x_pred = list(all_pred_eatv[i, :, 3])
#         y_pred = list(all_pred_eatv[i, :, 4])
#         z_pred = list(all_pred_eatv[i, :, 5])
#
#         fig = plt.figure()
#
#         ax = plt.axes(projection='3d')
#         ax.plot3D(x, y, z, 'green', label='Curve 1')
#         ax.plot3D(x_pred, y_pred, z_pred, 'red', label='Curve 2')
#         ax.text(x[0], y[0], z[0], 'Start', color='green', fontsize=12)
#         ax.text(x[-1], y[-1], z[-1], 'End', color='green', fontsize=12)
#         ax.legend()
#         ax.set_title('Multiple 3D Curves')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.savefig('trajectory_images/' + which_set + '/' + str(i) + '.png')
#         if i > 0 and i % 1000 == 0:
#             print('保存了1000张图片!')
#         plt.close()
