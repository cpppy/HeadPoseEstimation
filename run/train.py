import sys, os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from checkpoint_mgr.checkpoint_mgr import CheckpointMgr
from checkpoint_mgr.metrics import AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
n_cuda_device = torch.cuda.device_count()
print('n_cuda_device: ', n_cuda_device)


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    cudnn.enabled = True

    # ResNet50 structure
    from model_design import hopenet
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))

    save_dir = '/data/output/head_pose_estimate_hopenet_biwi_v1'
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
    alpha = 1.0

    softmax = nn.Softmax(dim=1).cuda()
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda().requires_grad_(False)

    # optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
    #                               {'params': get_non_ignored_params(model), 'lr': 1e-2},
    #                               {'params': get_fc_params(model), 'lr': 1e-2 * 5}],
    #                                lr =1e-2)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # build new optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if k.endswith('bias'):
            pg2.append(v)  # biases
        elif '.bn' in k and k.endswith('weight'):
            pg0.append(v)  # bn_weights no_decay
        elif k.endswith('weight') and len(v.shape) == 1:
            pg0.append(v)  # weights no_decay
        elif k.endswith('weight'):
            pg1.append(v)  # apply decay
        else:
            pg0.append(v)  # other_weights no_decay
    optimizer = torch.optim.SGD(pg0, lr=1e-4, momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    del pg0, pg1, pg2

    from datasets.biwi_dataset import BIWIData

    dataset = BIWIData()
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=64*n_cuda_device,
                                               shuffle=True,
                                               num_workers=2*n_cuda_device)
    num_epochs = 10000
    print('Ready to train network.')
    m_yaw_loss = AverageMeter()
    m_pitch_loss = AverageMeter()
    m_roll_loss = AverageMeter()
    m_yaw_err = AverageMeter()
    m_pitch_err = AverageMeter()
    m_roll_err = AverageMeter()
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels) in enumerate(train_loader):
            images = images.cuda()
            # Binned labels
            label_yaw = labels[:, 0].cuda()
            label_pitch = labels[:, 1].cuda()
            label_roll = labels[:, 2].cuda()

            # Continuous labels
            label_yaw_cont = cont_labels[:, 0].cuda()
            label_pitch_cont = cont_labels[:, 1].cuda()
            label_roll_cont = cont_labels[:, 2].cuda()

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            # TODO: make a distance_sense label smooth on cls_label
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

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
            m_yaw_err.update(torch.sum(torch.abs(yaw_predicted - label_yaw)/bs).item(), bs)
            m_pitch_err.update(torch.sum(torch.abs(pitch_predicted - label_pitch)/bs).item(), bs)
            m_roll_err.update(torch.sum(torch.abs(roll_predicted - label_roll)/bs).item(), bs)

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
