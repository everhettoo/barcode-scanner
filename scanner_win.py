from PyQt6.QtCore import QPoint, Qt, pyqtSlot
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QPushButton, QFileDialog

from job_controller import JobController


class ScannerWin(QWidget):
    H_MARGIN = 10
    V_MARGIN = 10
    WIN_WIDTH = 1200
    WIN_HEIGHT = 900
    HEIGHT_CORR = 0.05
    FRAME_INTERVAL = 30
    BUTTON_STYLE = "font-size:16px;"

    # BUTTON_STYLE = "border: 2px solid red; font-size:16px;"

    def __init__(self, device=0):  # 0 for default camera
        super().__init__()

        # Set window attributes.
        self.setFixedSize(self.WIN_WIDTH, self.WIN_HEIGHT)
        self.setWindowTitle('QR & Barcode Scanner')
        self.__center()

        # Setting the main layouts for subsections.
        self.main_h_layout = QHBoxLayout(self)
        self.main_h_layout.setContentsMargins(self.H_MARGIN, self.V_MARGIN, self.H_MARGIN, self.V_MARGIN)

        self.left_v_layout = QVBoxLayout()
        self.left_v_layout.setContentsMargins(self.H_MARGIN, self.V_MARGIN, self.H_MARGIN, self.V_MARGIN)

        self.right_v_layout = QVBoxLayout()
        # TODO: Top margin adjustment for right grid.
        self.right_v_layout.setContentsMargins(self.H_MARGIN, self.V_MARGIN, self.H_MARGIN, self.V_MARGIN)

        self.main_h_layout.addLayout(self.left_v_layout)
        self.main_h_layout.addLayout(self.right_v_layout)

        # Declare buttons and event handlers.
        self.auto_button = QRadioButton(self)
        self.auto_button.setChecked(True)
        self.auto_button.setText('[Auto Capture]')
        self.auto_button.setFixedHeight(self.WIN_HEIGHT * self.HEIGHT_CORR)
        self.auto_button.setStyleSheet(self.BUTTON_STYLE)
        self.auto_button.clicked.connect(self.auto_toggle_button_click)

        self.capture_button = QPushButton(self)
        self.capture_button.setText('[Manual Capture]')
        self.capture_button.setStyleSheet(self.BUTTON_STYLE)
        self.capture_button.setFixedHeight(self.WIN_HEIGHT * self.HEIGHT_CORR)
        self.capture_button.clicked.connect(self.manual_capture_button_click)
        self.capture_button.setDisabled(True)

        self.upload_button = QPushButton(self)
        self.upload_button.setText('[Upload]')
        self.upload_button.setStyleSheet(self.BUTTON_STYLE)
        self.upload_button.setFixedHeight(self.WIN_HEIGHT * self.HEIGHT_CORR)
        self.upload_button.clicked.connect(self.manual_upload_button_click)
        self.upload_button.setDisabled(True)

        # Setup individual components (layouts)
        self.__add_screen_layout()
        self.__add_control_layout()
        self.__add_workspace_layout()
        self.__add_trace_layout()

        # On error will throw exception.
        self.controller = JobController(device, self.frame_callback, self.FRAME_INTERVAL)

        # Start the video feed.
        self.controller.start()

    def __add_screen_layout(self):
        # Add the screen label to left_v_layout.
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedWidth(self.WIN_WIDTH - 450)
        self.image_label.setFixedHeight(self.WIN_HEIGHT - 100)
        self.image_label.setStyleSheet("border: 2px solid green;")

        self.left_v_layout.addWidget(self.image_label)

    def __add_control_layout(self):
        # Add buttons to button_h_layout of left_v_layout
        # Horizontal layout for left_v_layout to arrange the buttons.
        self.button_h_layout = QHBoxLayout()
        # self.button_h_layout.setContentsMargins(self.H_MARGIN, self.V_MARGIN, self.H_MARGIN, self.V_MARGIN)
        self.left_v_layout.addLayout(self.button_h_layout)

        self.button_h_layout.addWidget(self.auto_button)
        self.button_h_layout.addWidget(self.capture_button)
        self.button_h_layout.addWidget(self.upload_button)

    def __add_workspace_layout(self):
        self.work_log = QLabel(self)
        self.work_log.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.work_log.setStyleSheet("border: 2px solid blue;")
        self.work_log.setText("Work log")
        self.work_log.setFixedHeight(self.WIN_HEIGHT * 0.6)

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

    def auto_toggle_button_click(self):
        val = self.sender()

        if val.isChecked():
            ret = self.controller.on_auto_mode()
            print(f'auto-request (ON) :{ret}')
            self.capture_button.setDisabled(ret)
            self.upload_button.setDisabled(ret)
        else:
            ret = self.controller.off_auto_mode()
            print(f'auto-request (OFF):{ret}')
            self.capture_button.setEnabled(ret)
            self.upload_button.setEnabled(ret)

    def manual_capture_button_click(self):
        print('manual capture')

    def manual_upload_button_click(self):
        # if not self.controller.auto_mode:
        ret = self.controller.on_manual_upload()
        print(f'upload-request:{ret}')
        if ret:
            # Set buttons for upload flow.
            self.upload_button.setDisabled(ret)
            self.capture_button.setDisabled(ret)
            self.image_label.setPixmap(QPixmap())

            # Set dialog to process uploaded file.
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Open File")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                print("Selected File:", selected_files[0])
                # TODO: job controller to validate for processing.
                img = self.controller.load_image(selected_files[0])
                h, w, ch = img.shape
                image = QImage(img.data, w, h, QImage.Format.Format_Grayscale16)  # pyside6: QImage.Format_RGB888
                pixmap = QPixmap.fromImage(image)
                self.image_label.setPixmap(pixmap)

            # Close file dialog and reset upload button.
            file_dialog.close()
            self.upload_button.setEnabled(ret)

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
        # self.controller.stop()
        self.controller.close()
        event.accept()
