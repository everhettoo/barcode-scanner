import threading
import time

import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QRadioButton, QHBoxLayout, QPushButton, QFileDialog, \
    QMessageBox

import image_processor
from job_controller import JobController
from widgets.scrolllable import ScrollLabel


class ScannerWin(QWidget):
    H_MARGIN = 10
    V_MARGIN = 10
    WIN_WIDTH = 1450
    WIN_HEIGHT = 900
    HEIGHT_CORR = 0.05
    FRAME_INTERVAL = 30
    BUTTON_STYLE = "font-size:16px;"
    SCREEN_LAYOUT_WIDTH_OFFSET = 750
    SCREEN_LAYOUT_HEIGHT_OFFSET = 100
    WORKSPACE_DISPLAY_X_OFFSET = 20
    WORKSPACE_DISPLAY_Y_OFFSET = 80

    # BUTTON_STYLE = "border: 2px solid red; font-size:16px;"

    def __init__(self, device=0, basic_mode=False):  # 0 for default camera
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

        # Declare Scrollable labels
        self.work_trace = ScrollLabel(self)

        # Setup individual components (layouts)
        self.__add_screen_layout()
        self.__add_control_layout()
        self.__add_workspace_layout()
        self.__add_trace_layout()

        # On error will throw exception.
        self.controller = JobController(device, self.frame_callback, self.trace_callback, self.process_callback,
                                        self.FRAME_INTERVAL)

        # For manual file upload testing the following are added.
        if basic_mode:
            ret = self.controller.off_auto_mode()
            self.capture_button.setEnabled(ret)
            self.upload_button.setEnabled(ret)
            self.auto_button.setDisabled(ret)
            self.controller.close()
        else:
            # Start the video feed.
            self.controller.start()

    def __add_screen_layout(self):
        # Add the screen label to left_v_layout.
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedWidth(self.WIN_WIDTH - self.SCREEN_LAYOUT_WIDTH_OFFSET)
        self.image_label.setFixedHeight(self.WIN_HEIGHT - self.SCREEN_LAYOUT_HEIGHT_OFFSET)
        # self.image_label.setStyleSheet("border: 2px solid green;")

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
        self.work_log.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.work_log.setStyleSheet("border: 1px solid black; font-size:16px;")
        self.work_log.setText("[ Workspace ]")
        self.work_log.setFixedHeight(self.WIN_HEIGHT * 0.6)

        self.right_v_layout.addWidget(self.work_log)

    def __add_trace_layout(self):
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
            self.capture_button.setDisabled(ret)
            self.upload_button.setDisabled(ret)
        else:
            ret = self.controller.off_auto_mode()
            self.capture_button.setEnabled(ret)
            self.upload_button.setEnabled(ret)

    def manual_capture_button_click(self):
        print('manual capture')

    def manual_upload_button_click(self):
        """
        This event click allows user to manually upload a image file for processing. Once processing is completed,
        the event doesn't need to be reset, the upload-mode continues until user clicks auto-mode for auto-scan or
        manual capture.
        :return:
        """
        ret = self.controller.on_manual_upload()
        img_to_process = None

        if ret:
            # Set buttons for upload flow.
            self.upload_button.setDisabled(ret)
            self.capture_button.setDisabled(ret)
            self.image_label.setPixmap(QPixmap())
            self.work_log.setPixmap(QPixmap())

            # Set dialog to process uploaded file.
            file_dialog = QFileDialog(self)
            try:
                file_dialog.setWindowTitle("Open File")
                file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
                file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

                if file_dialog.exec():
                    selected_files = file_dialog.selectedFiles()
                    img = self.controller.load_image(selected_files[0])
                    if img is not None:
                        # Set autoscale for uploaded image.
                        self.set_screen(img, QImage.Format.Format_RGB32, True)
                        img_to_process = img
            except Exception as e:
                self.show_warning('Manual Capture Error', str(e))
            finally:
                # Close file dialog and reset upload button.
                file_dialog.close()
                self.upload_button.setEnabled(ret)

            if img_to_process is not None:
                # Start new thread to allow UI updating.
                t1 = threading.Thread(target=self.controller.process_image, args=(img_to_process,))
                t1.start()
                # Exception should be caught and displayed in trace.
                # self.controller.process_image(img_to_process)

    def set_screen(self, img, image_format=None, autoscale=False):
        """
        Single method to display images on the application.
        :param autoscale: Is set true for displaying uploaded image.
        :param img: cv2.Mat (ndarray)
        :param image_format: cv2 format
        :return: None
        """
        h, w, ch = img.shape
        if image_format:
            image = QImage(img.data, w, h, image_format)
        else:
            image = QImage(img.data, w, h)

        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)

        # Autoscale is true for uploaded image.
        self.image_label.setScaledContents(autoscale)

    def set_workspace(self, img, image_format=None, autoscale=False):
        """
        Single method to display images on the application.
        :param autoscale: Is set true for displaying uploaded image.
        :param img: cv2.Mar (ndarray)
        :param image_format: cv2 format
        :return: None
        """

        # Delay thread for display effect.
        time.sleep(0.8)

        # Resize image for workspace size.
        img = image_processor.resize_image(img, self.work_log.width() - self.WORKSPACE_DISPLAY_X_OFFSET,
                                           self.work_log.height() - self.WORKSPACE_DISPLAY_X_OFFSET)
        h, w, ch = img.shape
        img = np.ascontiguousarray(img)

        self.work_log.setScaledContents(autoscale)
        image = QImage(img.data, w, h, image_format)
        pixmap = QPixmap.fromImage(image)
        self.work_log.setPixmap(pixmap)

    def show_warning(self, title: str, message: str):
        box = QMessageBox()
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(title)
        box.setDetailedText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

    def closeEvent(self, event):
        self.controller.close()
        event.accept()

    def frame_callback(self, frame):
        """
        This is a callback from Camera.
        :param frame: Is a matrix returned from video feed.
        :return: 0 or 1 to caller to handle consecutive error handling.
        """
        try:
            self.set_screen(frame, QImage.Format.Format_RGB888)
            # TODO: Spawn another thread for handling image processing to release the feed.

            return 0
        except Exception as e:
            print(e)
            return 1

    def trace_callback(self, msg):
        self.work_trace.setText(self.work_trace.text() + '\n' + msg)

    def process_callback(self, image, cropped):
        """
        Job controller will send processed image for UI update only when successful. When error occurs, controller
        should handle its trace without sending any image.
        :param cropped:cv2.Mat segmented image containing barcode/qr-code.
        :param image: cv2.Mat (numpy.ndarray) - The original annotated image.
        :return: None
        """
        # Update the main screen with the annotated image.
        self.set_screen(image, QImage.Format.Format_RGB32, True)

        # Update the workspace with the annotated image.
        self.set_workspace(cropped, QImage.Format.Format_RGB32, False)

        self.work_trace.verticalScrollBar().setValue(self.work_trace.verticalScrollBar().maximum())
