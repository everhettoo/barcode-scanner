import cv2
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QPushButton


class CamWin(QWidget):
    def __init__(self, video_source=0):  # 0 for default camera
        super().__init__()

        self.resize(1200, 900)
        # self.setWindowModality(Qt.WindowModality.WindowModal)

        # Setting the layouts.
        self.main_h_layout = QHBoxLayout(self)
        self.main_h_layout.setContentsMargins(10, 10, 10, 10)

        self.left_v_layout = QVBoxLayout(self)
        self.left_v_layout.setContentsMargins(10, 10, 10, 10)

        self.right_v_layout = QVBoxLayout(self)
        self.right_v_layout.setContentsMargins(10, 10, 10, 10)

        # self.main_h_layout.addLayout(self.left_v_layout)
        # self.main_h_layout.addLayout(self.right_v_layout)

        # self.__add_screen()
        # self.__add_controls()
        # self.__add_workspace()
        # self.__add_trace()
        #
        # self.video_capture = cv2.VideoCapture(video_source)
        # if not self.video_capture.isOpened():
        #     raise IOError("Cannot open webcam")
        #
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_frame)
        # self.timer.start(30)  # Update every 30 ms (adjust as needed)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            image = QImage(frame.data, w, h, QImage.Format.Format_RGB888)  # pyside6: QImage.Format_RGB888
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)
            # self.image_label.resize(680,480)

    def closeEvent(self, event):
        self.video_capture.release()
        event.accept()

    def __add_screen(self):
        # Add the screen label to left_v_layout.
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedWidth(750)
        self.image_label.setFixedHeight(800)
        self.image_label.setStyleSheet("border: 2px solid green;")

        self.left_v_layout.addWidget(self.image_label)

    def __add_controls(self):
        # Horizontal layout for left_v_layout to arrange the buttons.
        self.button_h_layout = QHBoxLayout(self)
        self.button_h_layout.setContentsMargins(10, 10, 10, 10)
        self.left_v_layout.addLayout(self.button_h_layout)

        # Add buttons to button_h_layout of left_v_layout
        self.auto_button = QRadioButton(self)
        self.auto_button.setChecked(True)
        self.auto_button.setText('Auto')
        self.auto_button.setStyleSheet("border: 2px solid red;")

        self.capture_button = QPushButton(self)
        self.capture_button.setText('Capture')
        self.capture_button.setStyleSheet("border: 2px solid red;")

        self.upload_button = QPushButton(self)
        self.upload_button.setText('Upload')
        self.upload_button.setStyleSheet("border: 2px solid red;")

        self.button_h_layout.addWidget(self.auto_button)
        self.button_h_layout.addWidget(self.capture_button)
        self.button_h_layout.addWidget(self.upload_button)

    def __add_workspace(self):
        self.work_log = QLabel(self)
        self.work_log.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.work_log.setStyleSheet("border: 2px solid blue;")
        self.work_log.setText("Work log")
        self.work_log.setFixedHeight(500)

        self.right_v_layout.addWidget(self.work_log)

    def __add_trace(self):
        self.work_trace = QLabel(self)
        self.work_trace.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.work_trace.setStyleSheet("border: 2px solid yellow;")
        self.work_trace.setText("Work trace")

        self.right_v_layout.addWidget(self.work_trace)

        self.setWindowTitle('Scanner')

        self.center()

    def center(self):
        # Get the geometry of the primary screen
        screen_geometry = QGuiApplication.primaryScreen().geometry()

        # Get the geometry of the widget
        widget_geometry = self.geometry()

        # Calculate the center point of the screen
        center_point = screen_geometry.center()

        # Calculate the top-left position of the widget to center it
        x = center_point.x() - widget_geometry.width() // 2
        y = center_point.y() - widget_geometry.height() // 2

        # Move the widget to the calculated position
        self.move(QPoint(x, y))
