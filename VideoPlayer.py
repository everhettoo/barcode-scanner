import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QApplication, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QRadioButton


class VideoPlayer(QWidget):
    def __init__(self, video_source=0):  # 0 for default camera
        super().__init__()

        self.video_capture = cv2.VideoCapture(video_source)
        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.image_label)

        self.h_layout = QHBoxLayout()
        # self.h_layout.setContentsMargins(0,0,0,0)
        self.h_layout.setSpacing(50)


        self.auto_button = QRadioButton(self)
        self.auto_button.setChecked(True)
        # self.auto_button.setObjectName("Auto")
        self.auto_button.setText('Auto')

        self.capture_button = QPushButton(self)
        self.capture_button.setText('Capture')

        # self.button1.setGeometry(10,20,200,300)
        # self.button1.setFixedSize(200, self.button1.height())
        self.upload_button = QPushButton(self)
        self.upload_button.setText('Upload')

        self.h_layout.addWidget(self.auto_button)
        self.h_layout.addWidget(self.capture_button)
        self.h_layout.addWidget(self.upload_button)

        self.layout.addLayout(self.h_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms (adjust as needed)

        self.setWindowTitle("QR-Scanner")
        # self.setFixedSize(self.width(), self.height())
        self.resize(1920,1080)
        # self.setGeometry(100,200,100,400)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            image = QImage(frame.data, w, h, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
            # self.image_label.resize(680,480)

    def closeEvent(self, event):
        self.video_capture.release()
        event.accept()
