
import cv2

class Camera:
    def __init__(self, cam_preset_num=10):
        self.cam_preset_num = cam_preset_num

    def get_cam_num(self):
        cnt = 0
        for device in range(0, self.cam_preset_num):
            stream = cv2.VideoCapture(device)

            grabbed = stream.grab()
            stream.release()
            if not grabbed:
                break
            print('found_live_cam, device_id:', device)
            cnt = cnt + 1

        return cnt

if __name__ == '__main__':
    cam = Camera()
    cam_num = cam.get_cam_num()
    print(cam_num)
