import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TraDataset
from torch.backends import cudnn
import logging
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler
from neighborhood_loss import NeighborhoodLoss
import matplotlib.pyplot as plt
import ikpy.chain
# from utils.rotation_matrix2euler_angles import rotationMatrixToEulerAngles


class Args:
    def __init__(self):
        self.training_trajectories_path = 'data/training'
        self.validation_trajectories_path = 'data/validation'
        self.batch_size = 512
        self.num_workers = 16
        self.lr = 0.001
        self.early_stop_patience = 20
        self.epochs = 10000
        self.device = torch.device('cuda:0')
        self.seed = 29
        self.weight_decay = 0.000005
        self.lr_reduce_factor = 0.1
        self.lr_reduce_patience = 8

        self.num_layers = 2
        self.embed_dims = 128
        self.num_heads = 2
        self.feedforward_channels = 256
        self.drop_rate = 0.3
        self.attn_drop_rate = 0.3
        self.drop_path_rate = 0.3
args = Args()


def train(model):
    # robotic_arm = ikpy.chain.Chain.from_urdf_file('/home/t41/BerenChouProjects/RAIKPR/assets/UR5/urdf/ur5_robot.urdf')
    logging.info(f'\nArguments:\n'
                 f'  training_trajectories_path: \'{args.training_trajectories_path}\'\n'
                 f'  validation_trajectories_path: \'{args.validation_trajectories_path}\'\n'
                 f'  batch_size: {args.batch_size}\n'
                 f'  num_workers: {args.num_workers}\n'
                 f'  lr: {args.lr}\n'
                 f'  early_stop_patience: {args.early_stop_patience}\n'
                 f'  epochs: {args.epochs}\n'
                 f'  device: {args.device}\n'
                 f'  seed: {args.seed}\n'
                 f'  weight_decay: {args.weight_decay}\n'
                 f'  lr_reduce_factor: {args.lr_reduce_factor}\n'
                 f'  lr_reduce_patience: {args.lr_reduce_patience}\n'
                 f'  model.num_layers: {args.num_layers}\n'
                 f'  model.embed_dims: {args.embed_dims}\n'
                 f'  model.num_heads: {args.num_heads}\n'
                 f'  model.feedforward_channels: {args.feedforward_channels}\n'
                 f'  model.drop_rate: {args.drop_rate}\n'
                 f'  model.attn_drop_rate: {args.attn_drop_rate}\n'
                 f'  model.drop_path_rate: {args.drop_path_rate}\n')

    logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")

    training_tra_dataset = TraDataset(args.training_trajectories_path)
    logging.info(f'Num of training trajectories: {len(training_tra_dataset) // 1000}K.')
    training_tra_dataloader = DataLoader(training_tra_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.num_workers)

    validation_tra_dataset = TraDataset(args.validation_trajectories_path)
    logging.info(f'Num of validation trajectories: {len(validation_tra_dataset) // 1000}K.\n')
    validation_tra_dataloader = DataLoader(validation_tra_dataset, batch_size=args.batch_size,
                                           shuffle=False, num_workers=args.num_workers)

    l1_loss = nn.L1Loss()
    neighborhood_loss = NeighborhoodLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduce_factor,
                                               patience=args.lr_reduce_patience)

    training_all_epochs_loss = []
    val_all_epochs_loss = []
    best_val_loss_per_batch = 5e5
    for epoch in range(args.epochs):
        # ========================================train======================================
        model.train()
        training_one_epoch_loss = []  # 包含iteration个per batch loss
        for _, (eatv_batch, ja_batch) in enumerate(training_tra_dataloader):
            eatv_batch = eatv_batch.to(args.device)
            ja_batch = ja_batch.to(args.device)

            pred_ja_batch = model(eatv_batch)
            primary_loss = l1_loss(pred_ja_batch, ja_batch)
            secondary_loss = neighborhood_loss(pred_ja_batch)
            total_loss = primary_loss + 0.3 * secondary_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            training_one_epoch_loss.append(total_loss.item())
        training_loss_per_batch = np.average(training_one_epoch_loss)
        logging.info("Epoch {}: training loss per batch = {:3f}, lr = {:3f}.".format(epoch, training_loss_per_batch,
                                                                                     optimizer.param_groups[0]['lr']))
        training_all_epochs_loss.append(training_loss_per_batch)

        # ========================================val========================================
        if epoch % 5 == 0 and epoch > 0:
            with torch.no_grad():
                model.eval()
                validation_one_epoch_loss = []  # 包含iteration个per batch loss
                # all_gt_val_eatv_list = []
                # all_pred_val_ja_list = []
                logging.info('Start validation...')
                for _, (eatv_batch, ja_batch) in enumerate(validation_tra_dataloader):
                    # all_gt_val_eatv_list.append(eatv_batch)

                    eatv_batch = eatv_batch.to(args.device)
                    ja_batch = ja_batch.to(args.device)

                    pred_ja_batch = model(eatv_batch)
                    # all_pred_val_ja_list.append(pred_ja_batch.cpu())
                    primary_loss = l1_loss(pred_ja_batch, ja_batch)

                    validation_one_epoch_loss.append(primary_loss.item())
                val_loss_per_batch = np.average(validation_one_epoch_loss)
                logging.info("Epoch {}: validation loss (without NeighborhoodLoss) per batch = {:3f}."
                             .format(epoch, val_loss_per_batch))
                scheduler.step(val_loss_per_batch)

                if val_loss_per_batch < best_val_loss_per_batch:
                    logging.info('New minimal val loss per batch: {:3f}!'.format(val_loss_per_batch))
                    best_val_loss_per_batch = val_loss_per_batch
                    # ckpt_name = 'best_model_loss{:3f}_epoch{}.pth'.format(best_val_loss_per_batch, epoch)
                    # torch.save(model, 'saved_ckpt/' + ckpt_name)
                    args.early_stop_patience = 20

                    # all_gt_eatv = torch.cat(all_gt_val_eatv_list, dim=0)
                    # all_pred_ja = torch.cat(all_pred_val_ja_list, dim=0)
                    # all_gt_eatv = torch.reshape(all_gt_eatv, (-1, 6)).numpy()  # torch.Size([300000, 6])
                    # all_pred_ja = torch.reshape(all_pred_ja, (-1, 6))  # torch.Size([300000, 6])
                    # total_translation_vectors_eud = 0
                    # for i in range(all_pred_ja.shape[0]):
                    #     transformation_matrix = robotic_arm.forward_kinematics(
                    #         [0] + [all_pred_ja[i][j] for j in range(all_pred_ja.shape[1])] + [0]
                    #     )
                    #     euler_angles = rotationMatrixToEulerAngles(transformation_matrix[:3, :3])
                    #     translation_vector = transformation_matrix[:3, 3]
                    #     single_pred_eatv = np.concatenate((euler_angles, translation_vector))
                    #     total_translation_vectors_eud += np.linalg.norm(single_pred_eatv[:3] - all_gt_eatv[i][:3], ord=2)
                    # logging.info('**********The average error (Euclidean distance) of each translation vector '
                    #       'in the validation set is {}.**********'.format(total_translation_vectors_eud / all_gt_eatv.shape[0]))

                val_all_epochs_loss.append(val_loss_per_batch)
                args.early_stop_patience -= 1
                logging.info('Validation finished...\n')
                if args.early_stop_patience == 0:
                    logging.info('Run out of patience!')
                    break

    # =========================================plot==========================================
    # plt.figure(figsize=(12, 4))
    #
    # plt.subplot(121)
    # plt.plot(training_all_epochs_loss[:])
    # plt.title("train_loss")
    #
    # plt.subplot(122)
    # plt.plot(val_all_epochs_loss)
    # plt.title("val_loss")
    #
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    from model import IKTransformer

    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logging.basicConfig(filename='training_' + str(datetime.now())[:16] + '.log',
                        level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M')

    ik_transformer = IKTransformer(num_layers=args.num_layers, embed_dims=args.embed_dims, num_heads=args.num_heads,
                                   feedforward_channels=args.feedforward_channels, drop_rate=args.drop_rate,
                                   attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path_rate).to(args.device)

    train(ik_transformer)
