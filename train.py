#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 09:53:41 2021
@author: Boxiang Yun   School:ECNU&HFUT   Email:971950297@qq.com
"""
import os
import sys
import json

import timm
import torch
import signal
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from argument import Transform
from torch.utils.data import DataLoader
from timm import create_model

from local_utils.tools import save_dict
from local_utils.misc import AverageMeter
from local_utils.metrics import NPV
from local_utils.tools import EarlyStopping
from local_utils.dice_bce_loss import Dice_BCE_Loss
from local_utils.seed_everything import seed_reproducer

from torch.optim.lr_scheduler import CosineAnnealingLR
from Data_Generate import Data_Generate_Base, mask2label
from networks.cenet import UNet
from networks.Att_Res2_UNet import CE_Net_
# from lib.networks.snake.ct_snake import get_network

from medpy.metric import dc, jc, assd, sensitivity, specificity, precision
from sklearn.metrics import accuracy_score
print(f"start train")

def main(args):
    seed_reproducer(42)
    root_path = args.root_path
    json_file = args.json_file
    batch = args.batch
    lr = args.lr
    wd = args.wd
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    net_type = args.net
    worker = args.worker
    device = args.device
    classes = args.classes

    device = torch.device(device)
    with open(os.path.join(root_path, json_file), 'r') as f:
        file_names = json.load(f)

    train_img_paths, val_img_paths, test_img_paths = [], [], []
    train_mask_paths, val_mask_paths, test_mask_paths = [], [], []
    for c in ['NIP']: #['NP', 'NIP', 'FS', 'TM']
        train_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['train']]
        train_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['train']]
        val_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['val']]
        val_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['val']]
        test_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['test']]
        test_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['test']]

    np.random.seed(42)
    np.random.shuffle(train_img_paths)
    np.random.seed(42)
    np.random.shuffle(train_mask_paths)
    np.random.seed(42)
    np.random.shuffle(val_img_paths)
    np.random.seed(42)
    np.random.shuffle(val_mask_paths)
    np.random.seed(42)
    np.random.shuffle(test_img_paths)
    np.random.seed(42)
    np.random.shuffle(test_mask_paths)

    train_transform = Transform(target_size=256,
                          nature_aug_ration={"fog": 0., "rain": 0., "shadow": 0., 'snow': 0., 'sun': 0.},
                          general_aug_ration={"flip": 0.2, "blur": 0.1, "gauss": 0.1, 'jitter': 0., 'bright': 0.2})
    val_transform = Transform(target_size=256)
    test_transform = Transform(target_size=256)

    train_db = Data_Generate_Base(train_img_paths, train_mask_paths, transform=train_transform, cut=1024)
    train_loader = DataLoader(train_db, batch_size=batch, shuffle=True, num_workers=worker, drop_last=False)
    val_db = Data_Generate_Base(val_img_paths, val_mask_paths, transform=val_transform, cut=1024)
    val_loader = DataLoader(val_db, batch_size=batch, shuffle=False, num_workers=worker, drop_last=False)
    test_db = Data_Generate_Base(test_img_paths, test_mask_paths, transform=test_transform, cut=1024)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=worker, drop_last=False)

    if os.path.exists(f'{output_path}/{experiment_name}') == False:
        os.mkdir(f'{output_path}/{experiment_name}')
    save_dict(os.path.join(f'{output_path}/{experiment_name}', 'args.csv'), args.__dict__)

    early_stopping_val = EarlyStopping(patience=10, verbose=True,
                                       path=os.path.join(f'{output_path}/{experiment_name}',
                                             f'best_{experiment_name}.pth'))
    if net_type=='unet':
        model = UNet(3, classes)
    elif net_type=='attres2unet':
        model = CE_Net_(num_channels=3, num_classes=classes)

    # If you want to repreduce the result of Snake , must install DCNv2!
    # elif net_type=='snake':
        # model = UNet(3, classes)
        # model = get_network(num_layers=34, heads={'ct_hm': 7, 'wh': 2}, head_conv=256, down_ratio=4)
    else:
        raise ValueError("Oops! That was no valid model.Try again...")
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    criterion = Dice_BCE_Loss()

    history = {'epoch': [], 'LR': [], 'train_loss': [], 'val_dice':[],
               'test_dice':[], 'test_acc':[], 'test_recall':[], 'test_spe' :[], 'test_pre':[], 'test_npv':[]}

    # history['test_acc'].append(accuracy_score(labels, outs))
    # history['test_recall'].append(sensitivity(labels, outs))
    # history['test_spe'].append(specificity(labels, outs))
    # history['test_pre'].append(precision(labels, outs))
    # history['test_npv'].append(NPV(labels, outs))

    ############## shut down train with save model and safely exit
    stop_training = False
    def sigint_handler(signal, frame):
        print("Ctrl+c caught, stopping the training and saving the model...")
        nonlocal stop_training
        stop_training = True
        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', 'log.csv'), index=False)

    signal.signal(signal.SIGINT, sigint_handler)
    for epoch in range(epochs):
        train_losses = AverageMeter()
        print('now start train ..')
        print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        train_losses.reset()
        model.train()
        labels, outs = [], []
        try:
            for idx, sample in enumerate(tqdm(train_loader)):
                if stop_training:
                    break
                x, label = sample
                x, label = x.to(device), label.to(device)
                out = model(x)

                loss = criterion(out, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.update(loss.item())

                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)
                labels.extend(label)
            # outs, labels = np.array(outs), np.array(labels)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, please reduce batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return
            else:
                raise e
        history['train_loss'].append(train_losses.avg)

        print('now start evaluate ...')
        model.eval()
        val_losses = AverageMeter()
        val_losses.reset()
        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(val_loader)):
                if stop_training:
                    break

                x, label = sample
                x, label = x.to(device), label.to(device)

                out = model(x)
                loss = criterion(out, label)
                val_losses.update(loss.item())
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)
                labels.extend(label)
        outs = np.where(np.array(outs) > 0.5, 1, 0).astype(np.int)
        labels = np.array(labels).astype(np.int)

        print('epoch {}/{}\t LR:{}\t train loss:{}\t val loss:{}' \
              .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, val_losses.avg))

        history['val_dice'].append(dc(labels, outs))
        # labels, outs = mask2label(labels), mask2label(outs)
        # history['val_acc'].append([accuracy_score(labels[:,i], outs[:,i]) for i in range(1)])
        # history['val_recall'].append([sensitivity(labels[:,i], outs[:,i]) for i in range(1)])
        # history['val_spe'].append([specificity(labels[:,i], outs[:,i]) for i in range(1)])
        # history['val_pre'].append([specificity(labels[:,i], outs[:,i]) for i in range(1)])
        # history['val_npv'].append([NPV(labels[:,i], outs[:,i]) for i in range(1)])
        # history['val_acc'].append(accuracy_score(labels, outs))
        # history['val_recall'].append(sensitivity(labels, outs))
        # history['val_spe'].append(specificity(labels, outs))
        # history['val_pre'].append(c(labels, outs))
        # history['val_npv'].append(NPV(labels, outs))

        print('now start test ...')
        model.eval()
        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_loader)):
                if stop_training:
                    break
                x, label = sample
                x, label = x.to(device), label.to(device)
                out = model(x)
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)
                labels.extend(label)
        outs = np.where(np.array(outs)>0.5, 1, 0).astype(np.int)
        labels = np.array(labels).astype(np.int)
        history['test_dice'].append(dc(labels, outs))

        # labels, outs = mask2label(labels), mask2label(outs)

        history['test_acc'].append([accuracy_score(labels[:, i].reshape(-1), outs[:, i].reshape(-1)) for i in range(classes)])
        history['test_recall'].append([sensitivity(labels[:, i].reshape(-1), outs[:, i].reshape(-1)) for i in range(classes)])
        history['test_spe'].append([specificity(labels[:, i].reshape(-1), outs[:, i].reshape(-1)) for i in range(classes)])
        history['test_pre'].append([precision(labels[:, i].reshape(-1), outs[:, i].reshape(-1)) for i in range(classes)])
        history['test_npv'].append([NPV(labels[:, i].reshape(-1), outs[:, i].reshape(-1)) for i in range(classes)])
        # history['test_acc'].append(accuracy_score(labels, outs))
        # history['test_recall'].append(sensitivity(labels, outs))
        # history['test_spe'].append(specificity(labels, outs))
        # history['test_pre'].append(precision(labels, outs))
        # history['test_npv'].append(NPV(labels, outs))
        history['epoch'].append(epoch + 1)
        history['LR'].append(optimizer.param_groups[0]['lr'])
        scheduler.step(val_losses.avg)

        early_stopping_val(val_losses.avg, model)
        if early_stopping_val.early_stop:
            print("Early stopping")
            break

        if stop_training:
            torch.save(model.state_dict(),
                       os.path.join(f'{output_path}/{experiment_name}', 'final_{}.pth'.format(epoch)))
            break

        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log.csv'), index=False)
    history_pd = pd.DataFrame(history)
    history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', '-r', type=str, default='/home/ubuntu/T/Nose/all_data/image')
    parser.add_argument('--json_file', '-js', type=str, default='zyn.json')

    parser.add_argument('--device', '-dev', type=str, default='cuda:0')
    parser.add_argument('--worker', '-nw', type=int, default=4)
    parser.add_argument('--batch', '-b', type=int, default=8)
    parser.add_argument('--lr', '-l', default=0.001, type=float)
    parser.add_argument('--wd', '-w', default=5e-4, type=float)
    parser.add_argument('--output', '-o', type=str, default='./checkpoint')

    parser.add_argument('--experiment_name', '-name', type=str, default='resnet34_dual_stream')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--classes', '-c', type=int, default=4)
    parser.add_argument('--net', '-n', default='attres2unet', type=str)

    args = parser.parse_args()

    main(args)
