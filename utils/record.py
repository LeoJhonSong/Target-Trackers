import time
import cv2


class Recorder(object):
    def __init__(self, cap, note) -> None:
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        log_file = note + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + '.log'
        vd_file = note + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + '.mp4'
        self.vw = cv2.VideoWriter(vd_file, codec, cap.get(cv2.CAP_PROP_FPS), shape, True)
        self.log = open(log_file, "w")

    def write(self, frame, text):
        self.vw.write(frame)
        if len(text) != 0:
            self.log.write(text + '\n')

    def release(self):
        self.vw.release()
        self.log.close()
