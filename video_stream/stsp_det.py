import cv2
from face_detect_mbv2_api.detect_inference import FaceDetAPI
from deploy.head_pose_est_api import HeadPoseEstAPI
from utils import visual
import time


def infer(img_cv2):
    # img_cv2 = cv2.flip(img_cv2, flipCode=1)
    face_bbox = face_det_api(img_cv2=img_cv2)
    if len(face_bbox) == 4:
        # draw face bbox
        b = list(map(int, face_bbox))
        cv2.rectangle(img_cv2, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=1)

        # head pose est
        x0, y0, x1, y1 = face_bbox[:]
        face_img = img_cv2[y0:y1, x0:x1]
        yaw, pitch, roll = hpe_api(face_img)[:]
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        visual.draw_axis(img_cv2, yaw, pitch, roll, tdx=cx, tdy=cy)
    return img_cv2



def run():

    cap = cv2.VideoCapture(
        # 'rtsp://192.168.142.201/live1.sdp'
        'rtsp://admin:ucloud123@192.168.183.254/h264/ch1/main/av_stream'
        # 'rtsp://admin:8848chaffee@192.168.183.244/h264/ch1/main/av_stream'
    )
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        # frame = cv2.resize(frame, (640, 360))
        # frame = infer(frame)
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__=='__main__':

    face_det_api = FaceDetAPI()
    hpe_api = HeadPoseEstAPI()
    print('RUN !!!')
    run()