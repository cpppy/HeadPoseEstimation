import cv2
import numpy as np
import os


class Expand(object):

    def __init__(self, expand_ratio=[1.18, 3.0], expand_method='random'):
        super(Expand, self).__init__()
        self.expand_ratio = expand_ratio
        self.expand_method = expand_method
        if expand_ratio is not None:
            assert min(expand_ratio) > 1 and max(expand_ratio) <= 3
        assert expand_method in ['central', 'random']

    def __call__(self, img_cv2):
        h, w = img_cv2.shape[0:2]
        ratio = np.random.uniform(low=min(self.expand_ratio), high=max(self.expand_ratio))
        new_h, new_w = int(h * ratio), int(w * ratio)
        if self.expand_method == 'random':
            x0 = np.random.randint(low=0, high=max(1, new_w - w))
            y0 = np.random.randint(low=0, high=max(1, new_h - h))
        else:
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
        new_img = np.zeros((new_h, new_w, 3), dtype=img_cv2.dtype)
        new_img[y0: y0 + h, x0:x0 + w] = img_cv2
        img_cv2 = cv2.resize(new_img, (w, h))
        return img_cv2

    def __repr__(self):
        return self.__class__.__name__ + '(expand_ratio={})'.format(
            self.expand_ratio)

if __name__=='__main__':
    x0 = np.random.randint(low=0, high=1)
    print(x0)


