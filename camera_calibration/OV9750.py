import cv2


class OV9750:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280*2)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    def get_frame(self):
        ret, frame = self.cam.read()
        img_L = []
        img_R = []
        if ret:
            img_L = frame[:, :1280]
            img_R = frame[:, 1280:]
        return img_L, img_R

    def get_frame_stacked(self):
        ret, frame = self.cam.read()
        return frame

    def close(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np
    cam = OV9750()
    while cv2.waitKey(1):
        img_L, img_R = cam.get_frame()
        # stacked = cam.get_frame_stacked()
        print (np.shape(img_L))
        cv2.imshow("", img_L)
