import onnxruntime
import cv2
import numpy as np



class HeadPoseEstAPI(object):

    def __init__(self,
                 # model_path='./head_pose_estimation_hopenet_biwi_mbv2_v3.onnx',
                 model_path='/data/FaceRecog/head_pose/head_pose_estimation_hopenet_biwi_mbv2_20211223.onnx'
                 ):
        super(HeadPoseEstAPI, self).__init__()
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        model = onnxruntime.InferenceSession(model_path, sess_options=opts)
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

        # decode
        idx_t = np.arange(60)[np.newaxis, :]
        angles = np.sum(output * idx_t, axis=1)*3 - 90
        yaw, pitch, roll = list(angles)

        # if yaw >= -15 and yaw <= 25 and pitch >= -32 and pitch <= 22 and roll >= -23 and roll <= 23:
        if yaw >= -18 and yaw <= 25 and pitch >= -27 and pitch <= 22 and roll >= -23 and roll <= 23:
            is_valid = True
        else:
            is_valid = False
        return dict(is_valid=is_valid, metrics=dict(yaw=yaw, pitch=pitch, roll=roll))


def test():

    hpe_api = HeadPoseEstAPI()

    img_cv2 = cv2.imread('../dataset/test3.png', cv2.IMREAD_COLOR)
    # img_cv2 = cv2.imread('/data/FaceRecog/deploy/0.jpg')
    print('org_img:', img_cv2.shape)
    result = hpe_api(img_cv2)
    print(result)
    yaw, pitch, roll = result['metrics']['yaw'], result['metrics']['pitch'], result['metrics']['roll']
    from utils import visual
    # draw_img = visual.plot_pose_cube(img_cv2.copy(), yaw, pitch, roll)
    draw_img = visual.draw_axis(img_cv2.copy(), yaw, pitch, roll)
    cv2.imwrite('result3_crop.jpg', draw_img)



if __name__ == '__main__':

    test()
