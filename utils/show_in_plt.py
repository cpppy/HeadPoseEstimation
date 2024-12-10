import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('')

import cv2


def show_image_in_plt(img_cv2):
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()





if __name__=='__main__':

    img_cv2 = cv2.imread('../dataset/aug_utils/7.png')
    show_image_in_plt(img_cv2)


