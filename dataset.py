from torch.utils.data import Dataset
import h5py
import mmcv
import os.path as osp
import torch


class TraDataset(Dataset):
    def __init__(self, trajectories_path):
        """
        Args:
            trajectories_path: 存放轨迹文件的路径
        """
        self.trajectories_path = trajectories_path

        self.trajectories_names_list = []
        for trajectory_name in mmcv.scandir(self.trajectories_path, '.hdf5'):
            self.trajectories_names_list.append(trajectory_name)

    def __getitem__(self, index):
        trajectory = h5py.File(osp.join(self.trajectories_path, self.trajectories_names_list[index]), 'r')
        # eatv: euler angles & translation vectors
        # ja: joint angles
        eatv = torch.Tensor(trajectory.get('results')[:])  # torch.Size([100, 6])
        ja = torch.Tensor(trajectory.get('inputs')[:])  # torch.Size([100, 6])
        return eatv, ja

    def __len__(self):
        return len(self.trajectories_names_list)


# if __name__ == '__main__':
#     import ikpy.chain
#     from torch.utils.data import DataLoader
#     from data.data_generator import rotationMatrixToEulerAngles
#
#     robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/zulipeng/zyl/RAIKPR/assets/UR5/urdf/ur5_robot.urdf')
#
#     tra_dataset = TraDataset('data/training')
#     train_loader = DataLoader(tra_dataset, batch_size=200, shuffle=True, num_workers=2)
#
#     i = 0
#     for _, (eatv_batch, ja_batch) in enumerate(train_loader):
#         if i == 0:
#             # print(str(eatv_batch.shape) + '  ' + str(ja_batch.shape))
#             # torch.Size([200, 100, 6])  torch.Size([200, 100, 6])
#             transformation_matrix = robotic_arm.forward_kinematics([0] + list(ja_batch[0][99]) + [0])
#             euler_angles = rotationMatrixToEulerAngles(transformation_matrix[:3, :3])
#             translation_vector = transformation_matrix[:3, 3]
#             _eatv = eatv_batch[0][99]
#         i += 1
#     print('共{}个iterations'.format(i))
