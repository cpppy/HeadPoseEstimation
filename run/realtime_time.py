import cv2


def realtime_test():
    # face_detect_api = FaceDetectAPI(img_size=(480, 640),
    #                                 device_type='cpu',
    #                                 draw_save_dir='/data/FaceDetect/results')

    # img_cv2 = cv2.imread('/data/FaceDetect/7.png')
    # img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    # face_detect_api.detect(img_cv2=img_cv2, draw_result=True)

    ######################## load camera ######################
    clicked = False

    def onMouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            clicked = True

    cameraCapture = cv2.VideoCapture(0)
    # cameraCapture.set(3, 720)
    # cameraCapture.set(4, 1280)
    cameraCapture.set(cv2.CAP_PROP_FPS, 60)
    cv2.namedWindow('MyWindow')
    cv2.setMouseCallback('MyWindow', onMouse)
    print('Shiwing camera feed, Click window or press any key to stop')
    success, frame = cameraCapture.read()
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('MyWindow', frame)
        success, frame = cameraCapture.read()
        # print(frame.shape)
        # frame = rotate90(frame)
        img_cv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = face_detect_api.detect(img_cv2=img_cv2, score_thresh=0.4,  draw_result=True)
        # print(frame.shape)
        frame = img_cv2

    cv2.destroyAllWindows()
    cameraCapture.release()


if __name__ == '__main__':
    realtime_test()
    # batch_test()
