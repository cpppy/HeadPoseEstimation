import ffmpeg
import rtsp
import cv2

if __name__ == '__main__':

    # client = rtsp.Client(rtsp_server_uri='rtsp://192.168.142.201/live1.sdp')
    # client.read().show()
    # client.close()0000000000000000000000000000000000000000000000000000

    cap = cv2.VideoCapture(
        # 'rtsp://192.168.142.201/live1.sdp'
        'rtsp://admin:ucloud123@192.168.183.254/h264/ch1/main/av_stream'
        # 'rtsp://admin:8848chaffee@192.168.183.244/h264/ch1/main/av_stream'
    )
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 360))
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
