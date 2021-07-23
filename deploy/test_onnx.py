import onnx
import onnxruntime
import cv2
import numpy as np
import time
import os


def onnx_inference():
    model_path = 'head_pose_estimation_hopenet_biwi_mbv2_v2.onnx'
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    model = onnxruntime.InferenceSession(model_path, sess_options=opts)

    img_cv2 = cv2.imread('../datasets/test.png', cv2.IMREAD_COLOR)
    print('org_img:', img_cv2.shape)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_cv2 = cv2.resize(img_cv2, (224, 224))
    img = np.asarray(img_cv2, np.float32)
    rgb_mean = np.array([123.675, 116.28, 103.53], np.float32)
    rgb_std = np.array([58.395, 57.12, 57.375], np.float32)
    img -= rgb_mean
    img /= rgb_std
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    print(img.shape)

    output = model.run(None, {'input': img})[0]
    print('output: ', output.shape)

    # decode
    idx_t = np.arange(180)[np.newaxis, :]
    angles = np.sum(output * idx_t, axis=1) - 90
    print(angles)
    yaw, pitch, roll = list(angles)

    from utils import visual
    # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
    draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
    cv2.imwrite('result3.jpg', draw_img)



if __name__ == '__main__':
    onnx_inference()
