import cv2


import cv2
import numpy as np
import os

from dataset.aug_utils.flip import RandomFlip
from dataset.aug_utils.shrink import RandomCrop
from dataset.aug_utils.noise import AddNoise
from dataset.aug_utils.effect import RandomEffect
from dataset.aug_utils.effect2 import RandomEffect2
from dataset.aug_utils.mixup import MixUp


class Pipeline2(object):
    def __init__(self):
        super(Pipeline2, self).__init__()
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_crop = RandomCrop(crop_ratio=[0.75, 0.80])
        self.add_noise = AddNoise()
        self.random_effect = RandomEffect()
        self.random_effect2 = RandomEffect2()
        self.mixup = MixUp()

    def __call__(self, img_cv2):
        # img_cv2 = self.random_flip(img_cv2)

        # img_cv2 = self.random_crop(img_cv2)
        # if np.random.rand() < 0.1:
        img_cv2 = self.add_noise(img_cv2)
        # if np.random.rand() < 0.3:
        #     img_cv2 = self.random_effect2(img_cv2)
        # if np.random.rand() < 0.3:
        #     img_cv2 = self.random_effect(img_cv2)
        return img_cv2


if __name__ == "__main__":
    func = Pipeline2()
    img_fpath = '/data/FaceRecog/datasets/aug_utils/7.png'
    img_cv2 = cv2.imread(img_fpath)
    img_cv2 = cv2.resize(img_cv2, (256, 256))
    effect = MixUp()
    img_cv2 = effect(img_cv2)
    # img_cv2 = cv2.putText(img_cv2,
    #             'RandomCrop',
    #             (10, 240),
    #             cv2.FONT_HERSHEY_PLAIN,
    #             fontScale=1.5,
    #             color=(255, 0, 0),
    #             thickness=2
    #             )
    cv2.imwrite('mixup.jpg', img_cv2)