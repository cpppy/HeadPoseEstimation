import cv2
import sys
sys.path.append('..')
from face_detect_mbv2_api.detect_inference import FaceDetAPI
from deploy.head_pose_est_api import HeadPoseEstAPI
from utils import visual

def realtime_test():

    face_det_api = FaceDetAPI()
    hpe_api = HeadPoseEstAPI(model_path='../deploy/head_pose_estimation_hopenet_biwi_mbv2_20211223.onnx')

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
    print('camera feed, Click window or press any key to stop')
    success, frame = cameraCapture.read()
    while success and cv2.waitKey(1) == -1 and not clicked:

        success, frame = cameraCapture.read()
        # print(frame.shape)
        # frame = rotate90(frame)
        # img_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_cv2 = frame
        img_cv2 = cv2.flip(img_cv2, flipCode=1)
        face_bbox = face_det_api(img_cv2=img_cv2, expand_ratio=0.2)
        if len(face_bbox) == 4:
            # head pose est
            x0, y0, x1, y1 = face_bbox[:]
            face_img = img_cv2[y0:y1, x0:x1]
            yaw, pitch, roll = hpe_api(face_img)[:]
            cx, cy = int((x0+x1)/2), int((y0+y1)/2)
            visual.draw_axis(img_cv2, yaw, pitch, roll, tdx=cx, tdy=cy)

            # draw face bbox
            b = list(map(int, face_bbox))
            if yaw >= -22 and yaw <= 22 and pitch >= -27 and pitch <= 22 and roll >= -23 and roll <= 23:
                cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), thickness=2)
            else:
                cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=4)

        frame = img_cv2

        # print(frame.shape)
        cv2.imshow('HeadPoseTest', frame)

    cv2.destroyAllWindows()
    cameraCapture.release()


if __name__ == '__main__':
    realtime_test()
    # batch_test()
