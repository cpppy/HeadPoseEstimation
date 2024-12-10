import cv2
import numpy as np
import os

from dataset.aug_utils.flip import RandomFlip
from dataset.aug_utils.shrink import RandomCrop
from dataset.aug_utils.noise import AddNoise
from dataset.aug_utils.effect import RandomEffect
from dataset.aug_utils.effect2 import RandomEffect2
from dataset.aug_utils.rotate import Rotate

from dataset.aug_utils.mixup import MixUp
from dataset.aug_utils.expand import Expand
from dataset.aug_utils.mosaic import Mosaic
from dataset.aug_utils.sharpness import Sharpness


class Pipeline(object):
    def __init__(self):
        super(Pipeline, self).__init__()
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_crop = RandomCrop(crop_ratio=[0.6, 0.9])
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()
        self.random_effect2 = RandomEffect2()
        self.rotate_func = Rotate()

        # self.mixup_func = MixUp()
        expand_params = dict(expand_ratio=[1.01, 1.2])
        mosaic_params = dict(mosaic_weight=0.1, base=224)
        self.expand_func = Expand(**expand_params)
        self.mosaic_func = Mosaic(**mosaic_params)
        self.sharpen_func = Sharpness()

    def __call__(self, img_cv2):
        # img_cv2 = self.random_flip(img_cv2)
        if np.random.rand() < 0.5:
            img_cv2 = self.random_crop(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self.add_noise(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect2(img_cv2)
        if np.random.rand() < 0.3:
            img_cv2 = self.random_effect(img_cv2)
        # if np.random.rand() < 0.2:
        #     img_cv2 = self.rotate_func(img_cv2)
        if np.random.rand() < 0.1:
            img_cv2 = self.expand_func(img_cv2)
        # if np.random.rand() < 0.1:
        #     img_cv2 = self.mixup_func(img_cv2)
        if np.random.rand() < 0.2:
            if np.random.rand() < 0.5:
                img_cv2 = self.mosaic_func(img_cv2)
            else:
                img_cv2 = self.sharpen_func(img_cv2)
        return img_cv2



if __name__ == "__main__":
    func = Pipeline()
    img_dir = '/data/FaceRecog/tupu_data/image_cache_0602_mtcnn_crop'
    for img_fn in os.listdir(img_dir):
        img_fpath = os.path.join(img_dir, img_fn)
        img_cv2 = cv2.imread(img_fpath)
        img_cv2 = func(img_cv2)
        cv2.imwrite('/data/FaceRecog/results/aug_test/' + img_fn, img_cv2)