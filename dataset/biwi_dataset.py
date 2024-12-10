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


transformations = transforms.Compose(
    [
        transforms.Resize(240),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class BIWIData(Dataset):
    def __init__(self,
                 data_dir='/data/data/BIWI/hpdb',
                 # filename_path,
                 transform=transformations,
                 img_ext='.png',
                 annot_ext='.txt',
                 image_mode='RGB',
                 bin_step=3):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.bin_step = bin_step

        self.det_api = FaceDetAPI()

        # filename_list = get_list_from_filenames(filename_path)
        img_fpathes = glob.glob(os.path.join(data_dir, '*/*.png'))
        print(img_fpathes[0:2])
        self.img_fpathes = self._check_bbox(img_fpathes)
        self.n = len(self.img_fpathes)

        # self.X_train = filename_list
        # self.y_train = filename_list
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
        # print('img_fpath:', img_fpath)
        # img = Image.open(img_fpath)
        # img = img.convert(self.image_mode)
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

        shrink_ratio = 0.0
        if shrink_ratio > 0:
            size = x_max - x_min
            x_min += int(size*shrink_ratio)
            x_max -= int(size*shrink_ratio)
            y_min += int(size*shrink_ratio)
            y_max -= int(size*shrink_ratio)

        img_cv2 = img_cv2[y_min: y_max, x_min: x_max]
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

        # # draw angles
        # from utils import visual
        # # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
        # draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
        # cv2.imwrite('visual.jpg', draw_img)

        # Bin values
        bins = np.array(range(-90, 90, self.bin_step))
        binned_pose = np.digitize([round(yaw), round(pitch), round(roll)], bins, right=False) - 1

        bin_label = torch.LongTensor(binned_pose)
        angle_label = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            transform = transforms.Compose(
                [
                    transforms.Scale(224),
                    # transforms.RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            img = transform(img_pil)

        # preprocess img
        # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # img_cv2 = cv2.resize(img_cv2, (224, 224))
        # img = np.array(img_cv2, dtype=np.float32)
        # img = (img - 127.5) / 127.5
        # img = np.transpose(img, axes=(2, 0, 1))
        return img, angle_label, bin_label#, None  # elf.X_train[index]

    def __len__(self):
        # 15,667
        return self.n


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(filename)s:line%(lineno)d][func:%(funcName)s]%(message)s")

    dataset = BIWIData(bin_step=1)
    print('n_sample: ', len(dataset))


    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               num_workers=2)
    for batch in train_loader:
        img, labels, cont_labels = batch
        print(img.shape, labels, cont_labels)
        break
