import cv2
import numpy as np
import os
import time


class Mosaic(object):
    def __init__(self, mosaic_weight=0.5, base='origin'):
        super(Mosaic, self).__init__()
        self.mosaic_weight = mosaic_weight
        self.base = base
        assert mosaic_weight > 0.001 and mosaic_weight < 1.0
        assert base == 'origin' or isinstance(base, int)

    def __call__(self, origin_img):
        if self.base == 'origin':
            h, w = origin_img.shape[0:2]
        else:
            h, w = self.base, self.base
        new_h, new_w = int((1 - self.mosaic_weight) * h), int((1 - self.mosaic_weight) * w)
        mosaic_img = cv2.resize(origin_img, (new_w, new_h))
        return cv2.resize(mosaic_img, (w, h))