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
from contrastive_like_loss import ContrastiveLikeLoss
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import matplotlib.pyplot as plt
# import ikpy.chain
# from utils.rotation_matrix2euler_angles import rotationMatrixToEulerAngles


class Args:
    def __init__(self):
        # major hyperparameters
        self.lr = 0.001
        self.batch_size = 512
        self.weight_decay = 0.00005

        # device
        self.device = torch.device('cuda:1')

        # loss
        self.contrastive_like_loss_weight = 0.0
        self.contrastive_like_loss_temperature = 2.0
        self.contrastive_like_loss_keep_num = 50

        self.enable_neighborhood_loss = False
        self.sub_trajectory_length_in_neighborhood_loss = 20
        self.inherent_neighborhood_in_neighborhood_loss = 0.0492

        # model
        self.num_layers = 8
        self.embed_dims = 256
        self.num_heads = 4
        self.feedforward_channels = 256
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.0

        # unchanging hyperparameters
        self.num_workers = 4
        self.seed = 29
        self.lr_reduce_factor = 0.5
        self.lr_reduce_patience = 4
        self.early_stop_patience = 15
        self.epochs = 10000
args = Args()


def train(model):
    logging.info(
        f'\nArguments:\n'

        f'  lr: {args.lr}\n'
        f'  batch_size: {args.batch_size}\n'
        f'  weight_decay: {args.weight_decay}\n\n'

        f'  device: {args.device}\n\n'

        f'  contrastive_like_loss_weight: {args.contrastive_like_loss_weight}\n'
        f'  contrastive_like_loss_temperature: {args.contrastive_like_loss_temperature}\n'
        f'  contrastive_like_loss_keep_num: {args.contrastive_like_loss_keep_num}\n\n'

        f'  enable_neighborhood_loss: {args.enable_neighborhood_loss}\n'
        f'  sub_trajectory_length_in_neighborhood_loss: {args.sub_trajectory_length_in_neighborhood_loss}\n'
        f'  inherent_neighborhood_in_neighborhood_loss: {args.inherent_neighborhood_in_neighborhood_loss}\n\n'

        f'  model.num_layers: {args.num_layers}\n'
        f'  model.embed_dims: {args.embed_dims}\n'
        f'  model.num_heads: {args.num_heads}\n'
        f'  model.feedforward_channels: {args.feedforward_channels}\n'
        f'  model.drop_rate: {args.drop_rate}\n'
        f'  model.attn_drop_rate: {args.attn_drop_rate}\n'
        f'  model.drop_path_rate: {args.drop_path_rate}\n\n'

        f'  num_workers: {args.num_workers}\n'
        f'  seed: {args.seed}\n'
        f'  lr_reduce_factor: {args.lr_reduce_factor}\n'
        f'  lr_reduce_patience: {args.lr_reduce_patience}\n'
        f'  early_stop_patience: {args.early_stop_patience}\n'
        f'  epochs: {args.epochs}\n\n'

        f'  Number of available GPUs: {torch.cuda.device_count()}\n\n')

    training_dataset = TraDataset('data_without_mutation/training/eatvs_npy.npy', 'data_without_mutation/training/jas_npy.npy')
    logging.info(f'Num of training trajectories: {len(training_dataset) // 1000}K.')
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)

    validation_dataset = TraDataset('data_without_mutation/validation/eatvs_npy.npy', 'data_without_mutation/validation/jas_npy.npy')
    logging.info(f'Num of validation trajectories: {len(validation_dataset) // 1000}K.\n')
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers, drop_last=True)

    l1_loss_f = nn.L1Loss()
    neighborhood_loss_f = NeighborhoodLoss(args.sub_trajectory_length_in_neighborhood_loss, args.inherent_neighborhood_in_neighborhood_loss)
    contrastive_like_loss_f = ContrastiveLikeLoss(batch_size=args.batch_size, temperature=args.contrastive_like_loss_temperature,
                                                  tra_length=100, keep_num=args.contrastive_like_loss_keep_num, device=args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduce_factor, patience=args.lr_reduce_patience)
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

    # training_all_epochs_loss = []
    # val_all_epochs_loss = []
    best_val_l1_loss_per_joint = 5e5
    for epoch in range(args.epochs):
        # ========================================train======================================
        model.train()

        training_one_epoch_total_loss = []
        training_one_epoch_l1_loss = []
        training_one_epoch_contrastive_like_loss = []
        training_one_epoch_neighborhood_loss = []
        # 以上四个list的len均为60000//512

        for _, (eatv_batch, ja_batch) in enumerate(training_dataloader):
            eatv_batch = eatv_batch.to(args.device)
            ja_batch = ja_batch.to(args.device)

            model_output = model(eatv_batch)
            pred_ja_batch = model_output[0]
            shallow_embedding_sequence = model_output[1]
            deep_embedding_sequence = model_output[2]

            l1_loss = l1_loss_f(pred_ja_batch, ja_batch)
            # L1Loss在每一个值上都求了平均(除以batchSize*100*6), 即mean(sum(abs(pred_ja_batch - ja_batch)))
            if args.enable_neighborhood_loss and epoch > 50:
                # start_time = time.time()
                neighborhood_loss = neighborhood_loss_f(pred_ja_batch)
                # end_time = time.time()
                # print(f"运行时间：{end_time - start_time}秒.")
                neighborhood_loss_dynamic_weight = epoch / 300 if epoch / 300 < 1.0 else 1.0
            else:
                neighborhood_loss = 0.0
                neighborhood_loss_dynamic_weight = 0.0
            if args.contrastive_like_loss_weight != 0.0:
                contrastive_like_loss = contrastive_like_loss_f(shallow_embedding_sequence, deep_embedding_sequence)
            else:
                contrastive_like_loss = 0.0

            total_loss = l1_loss + neighborhood_loss_dynamic_weight * neighborhood_loss +\
                         args.contrastive_like_loss_weight * contrastive_like_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            training_one_epoch_total_loss.append(total_loss.item())
            training_one_epoch_l1_loss.append(l1_loss.item())
            training_one_epoch_contrastive_like_loss.append(contrastive_like_loss)
            training_one_epoch_neighborhood_loss.append(neighborhood_loss)

        training_total_loss = sum(training_one_epoch_total_loss) / len(training_one_epoch_total_loss)
        training_l1_loss_per_joint = sum(training_one_epoch_l1_loss) / len(training_one_epoch_l1_loss)
        training_contrastive_like_loss_per_trajectory = sum(training_one_epoch_contrastive_like_loss) / len(training_one_epoch_contrastive_like_loss)
        training_neighborhood_loss_per_trajectory = sum(training_one_epoch_neighborhood_loss) / len(training_one_epoch_neighborhood_loss)

        logging.info(
            "Epoch {}: total_loss = {:3f}, l1_loss per joint = {:3f}, contrastive_like_loss_per_trajectory = {:3f}, "
            "neighborhood_loss_per_trajectory(dynamic_weight: {:3f}) = {:3f}, lr = {:3f}.".format(
                epoch, training_total_loss, training_l1_loss_per_joint, training_contrastive_like_loss_per_trajectory,
                neighborhood_loss_dynamic_weight, training_neighborhood_loss_per_trajectory, optimizer.param_groups[0]['lr']
            )
        )
        # training_all_epochs_loss.append(training_total_loss)

        # ========================================val========================================
        if epoch % 5 == 0 and epoch > 0:
            with torch.no_grad():
                model.eval()
                validation_one_epoch_l1_loss = []  # len为20000//512
                # all_gt_val_eatv_list = []
                # all_pred_val_ja_list = []
                logging.info('Start validation...')
                for _, (eatv_batch, ja_batch) in enumerate(validation_dataloader):
                    # all_gt_val_eatv_list.append(eatv_batch)

                    eatv_batch = eatv_batch.to(args.device)
                    ja_batch = ja_batch.to(args.device)

                    pred_ja_batch = model(eatv_batch)[0]
                    # all_pred_val_ja_list.append(pred_ja_batch.cpu())
                    l1_loss = l1_loss_f(pred_ja_batch, ja_batch)

                    validation_one_epoch_l1_loss.append(l1_loss.item())
                val_l1_loss_per_joint = sum(validation_one_epoch_l1_loss) / len(validation_one_epoch_l1_loss)
                logging.info(
                    "Epoch {}: validation l1_loss per joint = {:3f}, best prior = {:3f}.".format(
                        epoch, val_l1_loss_per_joint, best_val_l1_loss_per_joint)
                )
                scheduler.step(val_l1_loss_per_joint)

                if val_l1_loss_per_joint < best_val_l1_loss_per_joint:
                    logging.info('New minimal validation l1_loss per joint: {:3f}!'.format(val_l1_loss_per_joint))
                    best_val_l1_loss_per_joint = val_l1_loss_per_joint
                    # ckpt_name = 'best_model_loss{:3f}_epoch{}.pth'.format(best_val_l1_loss_per_joint, epoch)
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

                # val_all_epochs_loss.append(val_l1_loss_per_joint)
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

    ik_transformer = IKTransformer(tra_length=100, num_layers=args.num_layers,
                                   embed_dims=args.embed_dims, num_heads=args.num_heads,
                                   feedforward_channels=args.feedforward_channels,
                                   drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                                   drop_path_rate=args.drop_path_rate).to(args.device)

    train(ik_transformer)
