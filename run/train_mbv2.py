import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
from checkpoint_mgr.metrics import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
n_cuda_device = torch.cuda.device_count()
print('n_cuda_device: ', n_cuda_device)


if __name__ == '__main__':

    cudnn.enabled = True

    from model_design.hopenet_mbv2 import HopenetMBV2
    model = HopenetMBV2(num_bins=60)

    save_dir = '/data/output/head_pose_estimate_hopenet_mbv2_biwi_v1'
    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model,
                                  warm_load=True,
                                  map_location='cpu')

    if n_cuda_device > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    reg_criterion = nn.MSELoss().cuda()
    # Regression loss coefficient
    alpha = 0.001  # for mse loss
    beta = 1.0  # for cls loss

    idx_tensor = [idx for idx in range(60)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda().requires_grad_(False)

    ################### OPTIMIZER ##################
    # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # for k, v in model.named_parameters():
    #     if k.endswith('bias'):
    #         pg2.append(v)  # biases
    #     elif '.bn' in k and k.endswith('weight'):
    #         pg0.append(v)  # bn_weights no_decay
    #     elif k.endswith('weight') and len(v.shape) == 1:
    #         pg0.append(v)  # weights no_decay
    #     elif k.endswith('weight'):
    #         pg1.append(v)  # apply decay
    #     else:
    #         pg0.append(v)  # other_weights no_decay
    # optimizer = torch.optim.SGD(pg0, lr=1e-3, momentum=0.9, nesterov=True)
    # optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    # del pg0, pg1, pg2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    from datasets.biwi_dataset import BIWIData
    dataset = BIWIData()
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=128*n_cuda_device,
                                               shuffle=True,
                                               num_workers=4*n_cuda_device)
    num_epochs = 10000
    print('Ready to train network.')
    m_yaw_loss = AverageMeter()
    m_pitch_loss = AverageMeter()
    m_roll_loss = AverageMeter()
    m_yaw_err = AverageMeter()
    m_pitch_err = AverageMeter()
    m_roll_err = AverageMeter()
    for epoch in range(num_epochs):
        for i, (images, angle_labels, bin_labels) in enumerate(train_loader):
            images = images.cuda()
            # Binned labels
            label_yaw = angle_labels[:, 0].cuda()
            label_pitch = angle_labels[:, 1].cuda()
            label_roll = angle_labels[:, 2].cuda()

            # Continuous labels
            label_yaw_bin = bin_labels[:, 0].cuda()
            label_pitch_bin = bin_labels[:, 1].cuda()
            label_roll_bin = bin_labels[:, 2].cuda()

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            # TODO: make a distance_sense label smooth on cls_label
            loss_cls_yaw = criterion(yaw, label_yaw_bin)
            loss_cls_pitch = criterion(pitch, label_pitch_bin)
            loss_cls_roll = criterion(roll, label_roll_bin)

            # MSE loss
            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 90
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 90
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 90

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll)

            # Total loss
            loss_yaw = alpha * loss_reg_yaw + beta * loss_cls_yaw
            loss_pitch = alpha * loss_reg_pitch + beta * loss_cls_pitch
            loss_roll = alpha * loss_reg_roll + beta * loss_cls_roll

            bs = images.shape[0]
            m_yaw_loss.update(loss_yaw.item(), bs)
            m_pitch_loss.update(loss_pitch.item(), bs)
            m_roll_loss.update(loss_roll.item(), bs)

            # backward
            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            # print('loss_seq:', loss_seq)
            # grad_seq = [torch.ones(1).cuda() for _ in range(len(loss_seq))]
            grad_seq = [torch.tensor(1.0, dtype=torch.float32).cuda()] * len(loss_seq)
            # print('grad_seq:', grad_seq)
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            step = (i + 1 + epoch * len(train_loader))
            lr = optimizer.param_groups[0]['lr']

            # Mean absolute error
            m_yaw_err.update(torch.mean(torch.abs(yaw_predicted - label_yaw)).item(), bs)
            m_pitch_err.update(torch.mean(torch.abs(pitch_predicted - label_pitch)).item(), bs)
            m_roll_err.update(torch.mean(torch.abs(roll_predicted - label_roll)).item(), bs)

            if (step) % 10 == 0:
                print('epoch[{}][{}/{}] lr:{}, total_loss:{:.3f}, yaw_loss:{:.3f}, pitch_loss:{:.3f}, roll_loss:{:.3f}, '
                      'yaw_err:{:.3f}, pitch_err:{:.3f}, roll_err:{:.3f}'
                      .format(epoch + 1,
                              i + 1,
                              len(train_loader),
                              lr,
                              m_yaw_loss.avg + m_pitch_loss.avg + m_roll_loss.avg,
                              m_yaw_loss.avg,
                              m_pitch_loss.avg,
                              m_roll_loss.avg,
                              m_yaw_err.avg,
                              m_pitch_err.avg,
                              m_roll_err.avg,
                              ))
                m_yaw_loss.reset()
                m_pitch_loss.reset()
                m_roll_loss.reset()

            if epoch >= 0 and (step) % 100 == 0:
                checkpoint_op.save_checkpoint(model=model.module if n_cuda_device > 1 else model)
