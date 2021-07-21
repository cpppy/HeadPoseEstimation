import onnx
import onnxruntime
import cv2
import numpy as np
import time
import os
import logging

from face_detect_mbv2_api.py_cpu_nms import py_cpu_nms
from face_detect_mbv2_api import face as face_visual
from face_detect_mbv2_api.anchor_generator_np import AnchorGenerator


class FaceDetAPI(object):

    def __init__(self,
                 onnx_fpath='/data/HeadPoseEstimation/face_detect_mbv2_api/retinaface_mbv2_s840_fixfpn_relu_no_postproc_20210630.onnx',
                 draw_save_dir='./results'):
        super(FaceDetAPI, self).__init__()
        self.model = self._load_model(onnx_fpath)
        self.draw_save_dir = draw_save_dir
        if not os.path.exists(draw_save_dir):
            os.mkdir(draw_save_dir)

    def _load_model(self, model_path):
        logging.info('load_face_det_model:{}'.format(model_path))
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        model = onnxruntime.InferenceSession(model_path, sess_options=opts)
        return model


    def img_preprocess(self, img_cv2, use_origin_size=False, size_divisor=32):
        img = np.float32(img_cv2)

        # testing scale
        target_size = 840
        max_size = 1280
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if use_origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            resized_img_cv2 = cv2.resize(img_cv2, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        else:
            resized_img_cv2 = img_cv2
        # divisor
        pad_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        pad_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        shape = (pad_h, pad_w, img.shape[-1])
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = 0
        pad[:img.shape[0], :img.shape[1], ...] = img
        img = pad
        im_height, im_width, _ = img.shape
        rgb_mean = np.array([123.675, 116.28, 103.53], np.float32)
        rgb_std = np.array([58.395, 57.12, 57.375], np.float32)
        img -= rgb_mean
        img /= rgb_std
        img = np.transpose(img, axes=(2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return resized_img_cv2, img, resize

    @staticmethod
    def decode_bboxes(loc,
                      priors,
                      variances=[0.1, 0.2]
                      ):
        bboxes = np.concatenate(
            [
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
            ],
            axis=1
        )
        bboxes_x0y0 = bboxes[:, :2] - bboxes[:, 2:] / 2
        bboxes_x1y1 = bboxes_x0y0 + bboxes[:, 2:]
        return np.concatenate([bboxes_x0y0, bboxes_x1y1], axis=1)

    @staticmethod
    def decode_landmarks(pre,
                         priors,
                         variances=[0.1, 0.2]
                         ):
        landms = np.concatenate(
            [
                priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
            ],
            axis=1
        )
        return landms

    @staticmethod
    def select_max_face(bboxes, img_h, img_w, expand_ratio=0.0):
        logging.debug('bboxes:{}'.format(bboxes))
        if len(bboxes) == 0:
            return ()
        bboxes = np.asarray(bboxes)[:, 0:4]
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        max_area_idx = np.argmax(areas)
        max_area_bbox = bboxes[max_area_idx]
        x0, y0, x1, y1 = max_area_bbox
        w = x1 - x0
        h = y1 - y0
        xc = int((x0 + x1) / 2)
        yc = int((y0 + y1) / 2)
        size = max(w, h) + int(min(w, h) * expand_ratio)
        if xc + int(size / 2) > img_w:
            size = (img_w - xc) * 2
        if xc - int(size / 2) < 0:
            size = xc * 2
        if yc + int(size / 2) > img_h:
            size = (img_h - yc) * 2
        if yc - int(size / 2) < 0:
            size = yc * 2
        x0, y0 = xc - int(size / 2), yc - int(size / 2)
        x1, y1 = xc + int(size / 2), yc + int(size / 2)
        return (x0, y0, x1, y1)


    def detect(self,
               img_cv2,
               score_thresh=0.8,
               topK_before_NMS=100,
               topK_after_NMS=50,
               multi_face=False,
               draw_result=False):
        logging.debug('org_img_shape: {}'.format(img_cv2.shape))
        resized_img_cv2, img_t, resize = self.img_preprocess(img_cv2)
        logging.debug('resize: {}'.format(resize))
        logging.debug('img_t_size: {}'.format(img_t.shape))

        ################## model inference #################
        output = self.model.run(None, {'input': img_t})[0]
        conf, loc, landms = output[:, :, :2], output[:, :, 2:6], output[:, :, 6:]
        logging.debug('loc:{}'.format(loc.shape))
        logging.debug('conf:{}'.format(conf.shape))
        logging.debug('landms:{}'.format(landms.shape))

        ###################### decode box and landms ###################
        anchors = AnchorGenerator()(img_size=tuple(img_t.shape[2:4]))  # img_size=(h, w)

        boxes = self.decode_bboxes(loc[0], anchors)
        from scipy.special import softmax
        conf = softmax(conf, axis=2)
        scores = conf[0, :, 1]
        landms = self.decode_landmarks(landms[0], anchors)

        # ignore low scores
        inds = np.where(scores > score_thresh)[0]
        boxes = boxes[inds]
        landms = landms[inds]

        scores = scores[inds]

        # keep top-K before NMS
        # order = scores.argsort()[::-1]
        order = scores.argsort()[::-1][:topK_before_NMS]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep, :]

        # keep top-K after NMS
        dets = dets[:topK_after_NMS, :]
        landms = landms[:topK_after_NMS, :]

        dets = np.concatenate((dets, landms), axis=1)

        # ----------------------- enlarge to original scale -----------------------
        bboxes_xywh = []
        bboxes_scores = []
        bboxes_landmarks = []
        bboxes_xyxy = []
        for det, land in zip(dets, landms):
            box = det[0:4] / resize
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            x = int(box[0] + w/2)
            y = int(box[1] + h/2)
            bboxes_xywh.append([x, y, w, h])
            bboxes_xyxy.append([int(val) for val in box])
            bboxes_scores.append(round(float(det[4]), 5))
            land = (land / resize).astype(np.int)
            bboxes_landmarks.append(land.tolist())
        dets = np.concatenate([dets[:, 0:4] / resize, dets[:, 4:5], dets[:, 5:] / resize], axis=1)

        detect_result = dict(bboxes_xywh=bboxes_xywh,
                             bboxes_xyxy=bboxes_xyxy,
                             scores=bboxes_scores,
                             landmarks=bboxes_landmarks)
        logging.debug('detect_result: {}'.format(detect_result))
        if draw_result:
            img_draw = face_visual.draw_face_detect_result(img_cv2=img_cv2.copy(),
                                                           dets=dets,
                                                           visual_scale=840)
            save_fpath = os.path.join(self.draw_save_dir, 'draw_{}.jpg'.format(int(time.time() * 1000)))
            cv2.imwrite(save_fpath, img_draw)
            logging.debug('save draw_img to path: {}'.format(save_fpath))
        return detect_result


    def __call__(self, img_cv2):
        detect_result = self.detect(img_cv2)
        max_face_x0y0x1y1 = self.select_max_face(bboxes=detect_result['bboxes_xyxy'],
                                                 img_h=img_cv2.shape[0],
                                                 img_w=img_cv2.shape[1],
                                                 expand_ratio=0.6)
        logging.debug('max_face:{}'.format(max_face_x0y0x1y1))
        # # crop and save
        # x0, y0, x1, y1 = max_face_x0y0x1y1
        # max_face_img = img_cv2[y0:y1, x0:x1]
        # save_fpath = os.path.join(self.draw_save_dir,
        #                           'draw_{}.jpg'.format(int(time.time() * 1000)))
        # cv2.imwrite(save_fpath, max_face_img)
        # print('save draw_img to path: {}'.format(save_fpath))
        return max_face_x0y0x1y1




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s][%(levelname)s][%(filename)s:line%(lineno)d][func:%(funcName)s]%(message)s")

    det_api = FaceDetAPI()
    img_cv2 = cv2.imread('/data/data/BIWI/hpdb/12/frame_00567_rgb.png')
    det_api(img_cv2)

