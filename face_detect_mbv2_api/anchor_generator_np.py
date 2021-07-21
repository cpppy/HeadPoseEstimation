
from itertools import product as product
import numpy as np
from math import ceil


class AnchorGenerator(object):

    def __init__(self):
        super(AnchorGenerator, self).__init__()
        self.anchor_size_for_fms = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False

    def __call__(self, img_size):
        anchors = []
        self.feature_maps = [
            [ceil(img_size[0] / step), ceil(img_size[1] / step)]
            for step in self.steps
        ]
        # print('feature_maps: {}'.format(self.feature_maps))
        for fm_i, fm_size in enumerate(self.feature_maps):
            anchor_sizes = self.anchor_size_for_fms[fm_i]
            for i, j in product(range(fm_size[0]), range(fm_size[1])):
                # print('cell_coord: {}'.format([i, j]))
                # for anchor_size in anchor_sizes:
                #     w_ratio = anchor_size / img_size[1]
                #     h_ratio = anchor_size / img_size[0]
                #     cx_ratio = (j + 0.5) * self.steps[fm_i] / img_size[1]
                #     cy_ratio = (i + 0.5) * self.steps[fm_i] / img_size[0]
                #     anchors.append([cx_ratio, cy_ratio, w_ratio, h_ratio])
                for anchor_size in anchor_sizes:
                    w, h = anchor_size, anchor_size
                    cx = (j + 0.5) * self.steps[fm_i]
                    cy = (i + 0.5) * self.steps[fm_i]
                    anchors.append([cx, cy, w, h])

        anchors = np.array(anchors, dtype=np.float32)
        anchors = np.reshape(anchors, newshape=(-1, 4))
        return anchors


if __name__ == '__main__':
    output = AnchorGenerator()(img_size=(640, 480))
    print(output.shape)
