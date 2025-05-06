import cv2
from PyQt6.QtCore import QTimer


class Camera:
    def __init__(self, device, callback, interval):
        self.video_capture = None
        self.device = device
        self.callback = callback
        self.interval = interval
        self.err_cnt = 0
        self.err_threshold = 5

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start(self):
        self.video_capture = cv2.VideoCapture(self.device)
        if not self.video_capture.isOpened():
            raise IOError("Camera: Cannot open device!")
        self.timer.start(self.interval)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Return matrix for win and controller processing.
            if self.callback(frame) == 1:
                self.err_cnt += 1
            else:
                self.err_cnt = 0

            if self.err_cnt >= self.err_threshold:
                # TODO: Send signal to application to calibrate or shutdown.
                pass
        else:
            return 1

    def close(self):
        if self.video_capture is not None:
            self.video_capture.release()
        self.timer.stop()
