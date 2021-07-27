import cv2
import numpy as np



class Sharpness(object):

    def __init__(self):
        super(Sharpness, self).__init__()

    def __call__(self, img_cv2):
        sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpen_img = cv2.filter2D(src=img_cv2, ddepth=cv2.CV_32F, kernel=sharpen_op)
        sharpen_img = cv2.convertScaleAbs(sharpen_img)
        return sharpen_img



if __name__=='__main__':

    img_cv2 = cv2.imread('./7.png')
    img_cv2 = cv2.resize(img_cv2, (256, 256))
    sharpen_img = Sharpness()(img_cv2)
    print(sharpen_img.shape)
    comp = cv2.hconcat([img_cv2, sharpen_img])
    cv2.imshow('sharpness', comp)
    cv2.waitKey(0)



