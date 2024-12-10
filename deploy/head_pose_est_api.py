import onnxruntime
import cv2
import numpy as np



class HeadPoseEstAPI(object):

    def __init__(self,
                 model_path='/data/HeadPoseEstimation/deploy/head_pose_estimation_hopenet_biwi_mbv2_v3.onnx'):
        super(HeadPoseEstAPI, self).__init__()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        # opts = onnxruntime.SessionOptions()
        # opts.intra_op_num_threads = 2
        # opts.inter_op_num_threads = 2
        # opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        # model = onnxruntime.InferenceSession(model_path, sess_options=opts)
        model = onnxruntime.InferenceSession(model_path, None, providers=["CUDAExecutionProvider"])
        return model

    def _img_preprocess(self, img_cv2):
        img_cv2 = cv2.resize(img_cv2, (112, 112))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img = np.asarray(img_cv2, np.float32)
        rgb_mean = np.array([123.675, 116.28, 103.53], np.float32)
        rgb_std = np.array([58.395, 57.12, 57.375], np.float32)
        img -= rgb_mean
        img /= rgb_std
        img = np.transpose(img, axes=(2, 0, 1))
        img = np.expand_dims(img, axis=0)
        # print(img.shape)
        return img

    def __call__(self, img_cv2):
        img_t = self._img_preprocess(img_cv2)
        output = self.model.run(None, {'input': img_t})[0][0]
        # print('output: ', output.shape)

        # decode
        idx_t = np.arange(60)[np.newaxis, :]
        angles = np.sum(output * idx_t, axis=1)*3 - 90
        yaw, pitch, roll = list(angles)
        return (yaw, pitch, roll)



def onnx_inference():

    hpe_api = HeadPoseEstAPI()

    img_cv2 = cv2.imread('../dataset/test.jpg', cv2.IMREAD_COLOR)
    print('org_img:', img_cv2.shape)
    yaw, pitch, roll = hpe_api(img_cv2)[:]

    # draw result
    from utils import visual
    # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
    draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
    cv2.imwrite('result9.jpg', draw_img)




if __name__ == '__main__':
    onnx_inference()
