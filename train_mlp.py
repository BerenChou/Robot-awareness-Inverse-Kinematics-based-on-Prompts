import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MLP_Dataset
from torch.backends import cudnn
import logging
from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler
from fk_loss import FKLoss


class Args:
    def __init__(self):
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.0

        self.batch_size = 512
        self.loss_f = 'L1Loss'  # or 'MSELoss'
        self.loss_f_in_fk = 'None'  # 'L1Loss' or 'MSELoss' or 'None' which means disable fk_loss
        self.normalization_value_for_radian = 1.0  # or torch.pi

        self.input_dim = 6
        self.hidden_size = 256
        self.output_dim = 6

        self.device = torch.device('cuda:1')

        self.num_workers = 4
        self.seed = 400
        self.lr_reduce_factor = 0.5
        self.lr_reduce_patience = 6
        self.epochs = 10000
args = Args()


def train(model):
    logging.info(
        f'\nArguments:\n\n'

        f'  lr: {args.lr}\n'
        f'  momentum: {args.momentum}\n'
        f'  weight_decay: {args.weight_decay}\n\n'

        f'  batch_size: {args.batch_size}\n'
        f'  loss_f: {args.loss_f}\n'
        f'  loss_f_in_fk: {args.loss_f_in_fk}\n'
        f'  normalization_value_for_radian: {args.normalization_value_for_radian}\n\n'

        f'  mlp.input_dim: {args.input_dim}\n'
        f'  mlp.hidden_size: {args.hidden_size}\n'
        f'  mlp.output_dim: {args.output_dim}\n\n'

        f'  device: {args.device} / {torch.cuda.device_count()}\n\n'

        f'  num_workers: {args.num_workers}\n'
        f'  seed: {args.seed}\n'
        f'  lr_reduce_factor: {args.lr_reduce_factor}\n'
        f'  lr_reduce_patience: {args.lr_reduce_patience}\n'
        f'  epochs: {args.epochs}\n\n')

    training_dataset = MLP_Dataset('data_without_mutation/training/eatvs_npy_samples.npy', 'data_without_mutation/training/jas_npy_samples.npy', args.normalization_value_for_radian)
    validation_dataset = MLP_Dataset('data_without_mutation/validation/eatvs_npy_samples.npy', 'data_without_mutation/validation/jas_npy_samples.npy', args.normalization_value_for_radian)
    logging.info(f'Num of training samples: {len(training_dataset) // 10000}W.')
    logging.info(f'Num of validation samples: {len(validation_dataset) // 10000}W.\n')
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)

    if args.loss_f == 'L1Loss':
        loss_f = nn.L1Loss()
    elif args.loss_f == 'MSELoss':
        loss_f = nn.MSELoss()
    else:
        raise Exception('Please specify the loss function for eatv2ja!')

    fk_loss_f = FKLoss(args.device, loss_name=args.loss_f_in_fk)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_reduce_factor, patience=args.lr_reduce_patience)

    best_validation_eatv2ja_loss = 5e5

    for epoch in range(args.epochs):
        # ========================================train======================================
        model.train()

        training_one_epoch_total_loss = []
        training_one_epoch_eatv2ja_loss = []
        training_one_epoch_pred_eatv2gt_eatv_loss = []

        for iteration, (eatv_batch, ja_batch) in enumerate(training_dataloader):
            eatv_batch = eatv_batch.to(args.device)
            ja_batch = ja_batch.to(args.device)

            pred_ja_batch = model(eatv_batch)
            eatv2ja_loss = loss_f(pred_ja_batch, ja_batch)

            if args.loss_f_in_fk != 'None':
                pred_eatv2gt_eatv_loss = fk_loss_f(pred_ja_batch, eatv_batch)
                total_loss = eatv2ja_loss + pred_eatv2gt_eatv_loss
            else:
                total_loss = eatv2ja_loss

            optimizer.zero_grad()
            total_loss.backward()
            if iteration == len(training_dataset) // args.batch_size - 1:  # 打印当前epoch最后一个iteration上的模型梯度
                for i, (name, param) in enumerate(model.named_parameters()):
                    if i == 0:
                        first_layer_grad_mean = torch.mean(abs(param.grad))
                        # first_layer_grad_var = torch.var(param.grad)
                    if i == sum(1 for _ in model.named_parameters()) - 2:  # 最后一组参数为bias, 所以取倒数第二组参数
                        last_layer_grad_mean = torch.mean(abs(param.grad))
                        # last_layer_grad_var = torch.var(param.grad)
            optimizer.step()

            training_one_epoch_total_loss.append(total_loss.item())
            training_one_epoch_eatv2ja_loss.append(eatv2ja_loss.item())
            training_one_epoch_pred_eatv2gt_eatv_loss.append(pred_eatv2gt_eatv_loss.item() if args.loss_f_in_fk != 'None' else 0.0)

        training_total_loss = sum(training_one_epoch_total_loss) / len(training_one_epoch_total_loss)
        training_eatv2ja_loss = sum(training_one_epoch_eatv2ja_loss) / len(training_one_epoch_eatv2ja_loss)
        training_pred_eatv2gt_eatv_loss = sum(training_one_epoch_pred_eatv2gt_eatv_loss) / len(training_one_epoch_pred_eatv2gt_eatv_loss)

        logging.info(
            "Epoch {}: total_loss={:.3f}, eatv2ja_loss={:.3f}, pred_eatv2gt_eatv_loss={:.3f}, lr={:.3f}, 1st_layer_grad_mean={:.5f}, last_layer_grad_mean={:.5f}.".format(
                epoch, training_total_loss, training_eatv2ja_loss, training_pred_eatv2gt_eatv_loss, optimizer.param_groups[0]['lr'], first_layer_grad_mean, last_layer_grad_mean,
            )
        )

        # ========================================val========================================
        if epoch % 5 == 0 and epoch > 0:
            with torch.no_grad():
                model.eval()

                validation_one_epoch_eatv2ja_loss = []
                validation_one_epoch_eatv2ja_MAE = torch.zeros(6).to(args.device)

                validation_num_iteration = 0
                logging.info('Start validation...')
                for _, (eatv_batch, ja_batch) in enumerate(validation_dataloader):
                    eatv_batch = eatv_batch.to(args.device)
                    ja_batch = ja_batch.to(args.device)

                    pred_ja_batch = model(eatv_batch)
                    eatv2ja_loss = loss_f(pred_ja_batch, ja_batch)

                    validation_one_epoch_eatv2ja_loss.append(eatv2ja_loss.item())
                    validation_one_epoch_eatv2ja_MAE += torch.mean(torch.abs(pred_ja_batch - ja_batch), dim=0)
                    validation_num_iteration += 1

                validation_eatv2ja_loss = sum(validation_one_epoch_eatv2ja_loss) / len(validation_one_epoch_eatv2ja_loss)

                logging.info(
                    "Epoch {}: eatv2ja_loss={:.3f}, best_prior_eatv2ja_loss={:.3f}, eatv2ja_MAE={}.".format(
                        epoch, validation_eatv2ja_loss, best_validation_eatv2ja_loss,
                        (validation_one_epoch_eatv2ja_MAE / validation_num_iteration * args.normalization_value_for_radian * 57.3).data
                    )
                )

                scheduler.step(validation_eatv2ja_loss)
                logging.info('Validation finished...')

                if validation_eatv2ja_loss < best_validation_eatv2ja_loss:
                    logging.info('New minimal validation eatv2ja_loss: {:.3f}!'.format(validation_eatv2ja_loss))
                    best_validation_eatv2ja_loss = validation_eatv2ja_loss
                    if best_validation_eatv2ja_loss < 0.20:
                        ckpt_name = '{}_eatv2jaloss{:.3f}_epoch{}.pth'.format(time, best_validation_eatv2ja_loss, epoch).replace(" ", "-")
                        torch.save(model, 'saved_ckpt/' + ckpt_name)
                        logging.info('Model ({}) has been saved.'.format(ckpt_name))

                logging.info('Continuing...\n')


if __name__ == '__main__':
    from model import MLP

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

    time = str(datetime.now())[:19]

    logging.basicConfig(filename=time + '.log', level=logging.INFO,
                        format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M')

    ik_mlp = MLP(input_dim=args.input_dim, hidden_size=args.hidden_size, output_dim=args.output_dim).to(args.device)

    train(ik_mlp)
