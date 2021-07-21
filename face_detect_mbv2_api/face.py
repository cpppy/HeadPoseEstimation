import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


def draw_face_detect_result(img_cv2, dets, visual_thresh=0.5, visual_scale=600.0):
    thickness = round(img_cv2.shape[0] * 0.8 / visual_scale)
    print('thickness: {}'.format(thickness))
    font_size = img_cv2.shape[0] * 0.3 / visual_scale
    for b in dets:
        # if b[4] < visual_thresh:
        #     continue
        text = "{:.2f}".format(b[4])

        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=thickness)
        cx = b[0]
        cy = b[1] - int(2 * img_cv2.shape[0] / visual_scale)
        cv2.putText(img_cv2, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, font_size, (255, 255, 255))

        # landms
        cv2.circle(img_cv2, (b[5], b[6]), 1, (0, 0, 255), thickness=thickness * 2)
        cv2.circle(img_cv2, (b[7], b[8]), 1, (0, 255, 255), thickness=thickness * 2)
        cv2.circle(img_cv2, (b[9], b[10]), 1, (255, 0, 255), thickness=thickness * 2)
        cv2.circle(img_cv2, (b[11], b[12]), 1, (0, 255, 0), thickness=thickness * 2)
        cv2.circle(img_cv2, (b[13], b[14]), 1, (255, 0, 0), thickness=thickness * 2)
    h, w = img_cv2.shape[0:2]
    rescale_ratio = visual_scale / max(h, w)
    img_cv2 = cv2.resize(img_cv2, (round(w * rescale_ratio), round(h * rescale_ratio)))
    return img_cv2


def draw_eval_compare(img_cv2, pred_bboxes, gt_bboxes, visual_thresh=0.5, visual_scale=600.0):
    thickness = round(img_cv2.shape[0] * 2.0 / 600.0)
    for b in pred_bboxes:
        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=thickness)
        # cx = b[0]
        # cy = b[1] + 12
        # cv2.putText(img_cv2, '', (cx, cy),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    for b in gt_bboxes:
        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), thickness=thickness)
    if visual_scale is not None:
        h, w = img_cv2.shape[0:2]
        rescale_ratio = visual_scale / max(h, w)
        img_cv2 = cv2.resize(img_cv2, (round(w * rescale_ratio), round(h * rescale_ratio)))
    return img_cv2


def draw_bboxes(img_cv2, bboxes, visual_scale=600.0, rect_color=(0, 0, 255)):
    thickness = round(img_cv2.shape[0] * 2.0 / 600.0)
    for b in bboxes:
        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), rect_color, thickness=thickness)
    if visual_scale is not None:
        h, w = img_cv2.shape[0:2]
        rescale_ratio = visual_scale / max(h, w)
        img_cv2 = cv2.resize(img_cv2, (round(w * rescale_ratio), round(h * rescale_ratio)))
    return img_cv2


def draw_bboxes_and_landmark(img_cv2, bboxes, landmarks, visual_scale=600.0):
    thickness = round(img_cv2.shape[0] * 2.0 / 600.0)
    for b in bboxes:
        b = list(map(int, b))
        print(b)
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=thickness)
    print('landmarks: {}'.format(landmarks))
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 0, 0)]
    for landmark in landmarks:
        for point, color in zip(landmark, colors):
            print('point: {}'.format(point))
            cv2.circle(img_cv2, (point[0], point[1]), 1, color, thickness=thickness * 2)

    if visual_scale is not None:
        h, w = img_cv2.shape[0:2]
        rescale_ratio = visual_scale / max(h, w)
        img_cv2 = cv2.resize(img_cv2, (round(w * rescale_ratio), round(h * rescale_ratio)))
    return img_cv2
