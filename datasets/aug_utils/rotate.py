import cv2
import random

class Rotate(object):

    def __init__(self, angle_range=[-10, 10]):
        self.angle_range = angle_range

    def _rotate(self, img_cv2, angle):
        h, w = img_cv2.shape[0:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
        img_res = cv2.warpAffine(img_cv2, M, (w, h))
        return img_res

    def __call__(self, img_cv2):
        angle = random.uniform(a=min(self.angle_range), b=max(self.angle_range))
        return self._rotate(img_cv2, 10)



if __name__=='__main__':

    img_cv2 = cv2.imread('./7.png')
    print(img_cv2.shape)
    img_cv2 = cv2.resize(img_cv2, (256, 256))
    img_res = Rotate()(img_cv2)
    cv2.imshow('result', img_res)
    cv2.waitKey(0)





