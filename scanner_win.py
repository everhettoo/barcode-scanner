import cv2
from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QPushButton

from camera import Camera


class ScannerWin(QWidget):
    def __init__(self, device=0):  # 0 for default camera
        super().__init__()

        # Set window attributes.
        self.setFixedSize(1200, 900)
        self.setWindowTitle('QR & Barcode Scanner')
        self.__center()

        # Setting the main layouts for subsections.
        self.main_h_layout = QHBoxLayout(self)
        self.main_h_layout.setContentsMargins(10, 10, 10, 10)

        self.left_v_layout = QVBoxLayout()
        self.left_v_layout.setContentsMargins(10, 10, 10, 10)

        self.right_v_layout = QVBoxLayout()
        self.right_v_layout.setContentsMargins(10, 10, 10, 10)

        self.main_h_layout.addLayout(self.left_v_layout)
        self.main_h_layout.addLayout(self.right_v_layout)

        # Setup individual components (layouts)
        self.__add_screen_layout()
        self.__add_control_layout()
        self.__add_workspace_layout()
        self.__add_trace_layout()

        # On error will throw exception.
        self.camera = Camera(device, self.frame_callback, 30)

        # Start the video feed.
        self.camera.start()

    def __add_screen_layout(self):
        # Add the screen label to left_v_layout.
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedWidth(750)
        self.image_label.setFixedHeight(800)
        self.image_label.setStyleSheet("border: 2px solid green;")

        self.left_v_layout.addWidget(self.image_label)

    def __add_control_layout(self):
        # Horizontal layout for left_v_layout to arrange the buttons.
        self.button_h_layout = QHBoxLayout()
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

    def __add_workspace_layout(self):
        self.work_log = QLabel(self)
        self.work_log.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.work_log.setStyleSheet("border: 2px solid blue;")
        self.work_log.setText("Work log")
        self.work_log.setFixedHeight(500)

        self.right_v_layout.addWidget(self.work_log)

    def __add_trace_layout(self):
        self.work_trace = QLabel(self)
        self.work_trace.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.work_trace.setStyleSheet("border: 2px solid yellow;")
        self.work_trace.setText("Work trace")

        self.right_v_layout.addWidget(self.work_trace)

    def __center(self):
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

    def frame_callback(self, frame):
        """
        This is a callback from Camera.
        :param frame: Is a matrix returned from video feed.
        :return: 0 or 1 to caller to handle consecutive error handling.
        """
        try:
            h, w, ch = frame.shape
            image = QImage(frame.data, w, h, QImage.Format.Format_RGB888)  # pyside6: QImage.Format_RGB888
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap)

            # TODO: Spawn another thread for handling image processing to release the feed.

            return 0
        except Exception as e:
            print(e)
            return 1

    def closeEvent(self, event):
        self.camera.close()
        event.accept()
