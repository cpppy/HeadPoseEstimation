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
from PIL import Image
from face_detect_mbv2_api.detect_inference import FaceDetAPI


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
n_cuda_device = torch.cuda.device_count()
print('n_cuda_device: ', n_cuda_device)

face_det_api = FaceDetAPI()

def img_preprocess(img_cv2, use_det=False):
    if use_det:
        bbox = face_det_api(img_cv2)
        if len(bbox) == 0:
            print('no face detected.')
            exit(0)
        x0, y0, x1, y1 = bbox
        img_cv2 = img_cv2[y0:y1, x0:x1]
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    transformations = transforms.Compose([transforms.Resize(240),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imgs_t = transformations(img_pil).unsqueeze(dim=0)
    return imgs_t, img_cv2


if __name__ == '__main__':

    cudnn.enabled = True

    # from model_design import hopenet
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # save_dir = '/data/output/head_pose_estimate_hopenet_biwi_v1'

    from model_design.hopenet_mbv2 import HopenetMBV2
    model = HopenetMBV2(num_bins=180)
    save_dir = '/data/output/head_pose_estimate_hopenet_mbv2_biwi_v2'

    checkpoint_op = CheckpointMgr(ckpt_dir=save_dir)
    checkpoint_op.load_checkpoint(model,
                                  warm_load=False,
                                  map_location='cpu')

    model.cuda()
    model.eval()

    # softmax = nn.Softmax(dim=1).cuda()
    idx_tensor = [idx for idx in range(180)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda().requires_grad_(False)

    img_fpath = '../datasets/test.jpg'
    # img_fpath = '/data/data/BIWI/hpdb/12/frame_00567_rgb.png'
    # img_fpath = '/data/data/BIWI/hpdb/08/frame_00076_rgb.png'
    img_cv2 = cv2.imread(img_fpath)
    images, face_img = img_preprocess(img_cv2, use_det=True)
    images = images.cuda()
    print(images.shape)

    # Forward pass
    angles = model(images)
    print(angles.shape)
    yaw, pitch, roll = torch.split(angles, split_size_or_sections=(1, 1, 1), dim=0)
    # yaw_predicted = F.softmax(yaw, dim=1)
    # pitch_predicted = F.softmax(pitch, dim=1)
    # roll_predicted = F.softmax(roll, dim=1)
    yaw_predicted = yaw
    pitch_predicted = pitch
    roll_predicted = roll

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 1 - 90
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 1 - 90
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 1 - 90

    yaw = yaw_predicted[0].item()
    pitch = pitch_predicted[0].item()
    roll = roll_predicted[0].item()

    print('yaw:{}, pitch:{}, roll:{}'.format(yaw, pitch, roll))

    from utils import visual
    # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
    draw_img = visual.draw_axis(face_img, yaw, pitch, roll)
    cv2.imwrite('result9.jpg', draw_img)




