import cv2
from face_detect_mbv2_api.detect_inference import FaceDetAPI
from deploy.head_pose_est_api import HeadPoseEstAPI
from utils import visual
import time

def realtime_test():

    face_det_api = FaceDetAPI()
    # hpe_api = HeadPoseEstAPI()

    ######################## load camera ######################
    clicked = False

    def onMouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            clicked = True

    cameraCapture = cv2.VideoCapture(0)
    cameraCapture.set(3, 720)
    cameraCapture.set(4, 1280)
    cameraCapture.set(cv2.CAP_PROP_FPS, 5)
    cv2.namedWindow('HeadPoseTest')
    cv2.setMouseCallback('HeadPoseTest', onMouse)
    print('Shiwing camera feed, Click window or press any key to stop')
    success, frame = cameraCapture.read()
    t0 = time.time()
    while success and cv2.waitKey(1) == -1 and not clicked:

        success, frame = cameraCapture.read()
        # print(frame.shape)
        # frame = rotate90(frame)
        # img_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cv2 = frame
        img_cv2 = cv2.flip(img_cv2, flipCode=1)
        face_bbox = face_det_api(img_cv2=img_cv2, expand_ratio=0.0)
        if len(face_bbox) == 4:
            b = list(map(int, face_bbox))
            line_color = (0, 0, 255)

            t1 = time.time()
            if t1 - t0 >= 1:
                face_img = img_cv2[b[1]:b[3], b[0]:b[2]]
                cv2.imwrite('./crop_faces/fake_{}.jpg'.format(t1*1e3), face_img)
                t0 = t1
                line_color = (255, 255, 0)
            # draw face bbox
            cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), line_color, thickness=2)

        frame = img_cv2

        # print(frame.shape)
        cv2.imshow('HeadPoseTest', frame)

    cv2.destroyAllWindows()
    cameraCapture.release()


if __name__ == '__main__':
    realtime_test()
    # batch_test()
