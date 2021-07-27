import os
import numpy as np
import cv2

para = {"sigma_list": [80, 200], "G": 10.0, "b": 25.0, "alpha": 125.0, "beta": 46.0, "low_clip": 0.01,
        "high_clip": 0.99}


def singleScaleRetinex(img, sigma):
    # min_nonzero = min(img[np.nonzero(img)])
    # img[img == 0] = min_nonzero
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    retinex = retinex / len(sigma_list)

    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        # print(unique, counts)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    h, w, _ = img.shape
    central = [int(h / 3), int(h * 0.66), int(w / 3), int(w * 0.66)]
    img = np.float64(img) + 1
    # singleScaleRetinex(img, sigma_list[2])#
    img_retinex = multiScaleRetinex(img, sigma_list)

    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    # '''
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[central[0]:central[1], central[2]:central[3], i])) / \
                             (np.max(img_msrcr[central[0]:central[1], central[2]:central[3], i]) - np.min(
                                 img_msrcr[central[0]:central[1], central[2]:central[3], i])) * \
                             255
    # '''
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def main(pic, save_path):
    img = cv2.imread(pic)
    msrcr_img = MSRCR(img, para['sigma_list'], para['G'], para['b'], para['alpha'], para['beta'], para['low_clip'],
                      para['high_clip'])
    name = os.path.basename(pic)
    cv2.imwrite(os.path.join(save_path, name), msrcr_img)


def getfilelist(path):
    pics = []
    dirs = os.listdir(path)
    for d in dirs:
        dirpath = os.path.join(path, d)
        files = os.listdir(dirpath)
        for f in files:
            if f[-3:] == 'jpg':
                pics.append(os.path.join(dirpath, f))
            else:
                test_file = os.listdir(os.path.join(dirpath, f))
                if len(test_file) > 0:
                    pics.extend([os.path.join(dirpath, f, test_f) for test_f in test_file])
    print('num of pics is ', len(pics))
    return pics


if __name__ == "__main__":
    picpath = '../tupu_data/guiyangzhongtian_collections_update0624/'
    save_path = 'preprocess_MSRCR_guiyang'
    pics = getfilelist(picpath)
    for p in pics:
        main(p, save_path)
