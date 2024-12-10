import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import glob
import logging

from face_detect_mbv2_api.detect_inference import FaceDetAPI

# import utils


class BIWIData(Dataset):
    def __init__(self,
                 data_dir='/data/data/BIWI/hpdb',
                 # filename_path,
                 aug_mode=None,
                 img_ext='.png',
                 annot_ext='.txt',
                 image_mode='RGB',
                 bbox_shrink=0.0,
                 bin_step=3):
        self.data_dir = data_dir
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.bbox_shrink = bbox_shrink
        self.bin_step = bin_step
        if aug_mode is None:
            self.pipeline = lambda x:x
        elif aug_mode is 'pipeline_v1':
            from dataset.pipeline_v1 import Pipeline
            self.pipeline = Pipeline()
        else:
            print('ERROR: INVALID PIPELINE TYPE.')
            exit(-1)

        self.det_api = FaceDetAPI()

        img_fpathes = glob.glob(os.path.join(data_dir, '*/*.png'))
        print(img_fpathes[0:2])
        self.img_fpathes = self._check_bbox(img_fpathes)
        self.n = len(self.img_fpathes)
        self.image_mode = image_mode

    def _check_bbox(self, img_fpathes):
        # check bbox
        valid_img_fpathes = []
        from tqdm import tqdm
        pbar = tqdm(desc='load', total=len(img_fpathes))
        for img_fpath in tqdm(img_fpathes):
            pbar.update(1)
            bbox_path = img_fpath.replace('_rgb.png', '_bbox' + self.annot_ext)
            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as f:
                    line = f.readlines()[0]
                    bbox = line.strip('\n').split(' ')
                    bbox = [int(val) for val in bbox[0:4]]
                    if len(bbox) == 4:
                        valid_img_fpathes.append(img_fpath)
                        continue
            # logging.debug('gen_bbox_label:{}'.format(bbox_path))
            # img_cv2 = cv2.imread(img_fpath)
            # bbox = self.det_api(img_cv2)
            # if len(bbox) == 4:
            #     with open(bbox_path, 'w') as f:
            #         line = ' '.join([str(val) for val in bbox])
            #         # print(line)
            #         f.write(line)
            #     valid_img_fpathes.append(img_fpath)
            # else:
            #     logging.info('### no face detected ! n_valid_sample:{}'.format(len(valid_img_fpathes)))
        pbar.close()
        print('n_valid_sample:', len(valid_img_fpathes))
        return valid_img_fpathes

    def __getitem__(self, index):
        img_fpath = self.img_fpathes[index]
        img_cv2 = cv2.imread(img_fpath)

        ###################### pose #####################
        pose_path = img_fpath.replace('_rgb.png', '_pose' + self.annot_ext)
        # print('label_fpath:', pose_path)
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)
        # print(R)
        R = np.array(R)
        T = R[3, :]
        R = R[:3, :]
        pose_annot.close()

        R = np.transpose(R)
        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        ######################### load bbox ########################
        bbox_path = img_fpath.replace('_rgb.png', '_bbox' + self.annot_ext)
        with open(bbox_path, 'r') as f:
            line = f.readlines()[0]
            bbox = line.strip('\n').split(' ')
            bbox = [int(val) for val in bbox[0:4]]
        x_min, y_min, x_max, y_max = [int(val) for val in bbox][:]

        shrink_ratio = self.bbox_shrink
        if shrink_ratio > 0:
            size = x_max - x_min
            x_min += int(size*shrink_ratio)
            x_max -= int(size*shrink_ratio)
            y_min += int(size*shrink_ratio)
            y_max -= int(size*shrink_ratio)

        img_cv2 = img_cv2[y_min: y_max, x_min: x_max]
        # img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # draw angles
        # from utils import visual
        # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
        # draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
        # cv2.imwrite('visual.jpg', draw_img)
        # exit(0)

        # Bin values
        bins = np.array(range(-90, 90, self.bin_step))
        binned_pose = np.digitize([round(yaw), round(pitch), round(roll)], bins, right=False) - 1

        bin_label = torch.LongTensor(binned_pose)
        angle_label = torch.FloatTensor([yaw, pitch, roll])

        img_cv2 = cv2.resize(img_cv2, (128, 128))
        img_cv2 = self.pipeline(img_cv2)

        # preprocess img
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_cv2 = cv2.resize(img_cv2, (112, 112))
        img = np.array(img_cv2, dtype=np.float32)
        rgb_mean = np.array([123.675, 116.28, 103.53], np.float32)
        rgb_std = np.array([58.395, 57.12, 57.375], np.float32)
        img -= rgb_mean
        img /= rgb_std
        img = np.transpose(img, axes=(2, 0, 1))
        return img, angle_label, bin_label#, None  # elf.X_train[index]

    def __len__(self):
        # 15,667
        return self.n


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(filename)s:line%(lineno)d][func:%(funcName)s]%(message)s")

    dataset = BIWIData(aug_mode='pipeline_v1', bbox_shrink=0.1, bin_step=3)
    print('n_sample: ', len(dataset))

    for sample in dataset:
        img, angle_label, bin_label = sample
        print(img.shape, angle_label, bin_label)
        break


    # train_loader = torch.utils.data.DataLoader(dataset=dataset,
    #                                            batch_size=8,
    #                                            shuffle=True,
    #                                            num_workers=2)
    # for batch in train_loader:
    #     img, labels, cont_labels = batch
    #     print(img.shape, labels, cont_labels)
    #     break
