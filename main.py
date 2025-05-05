# TODO: PyQT code (Don't delete!!!)
import cv2
from PyQt6 import uic
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication

from CamWin import CamWin
from CaptureForm import CaptureForm

# def button_clicked():
#     ret, frame = video_capture.read()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = frame.shape
#         image = QImage(frame.data, w, h, QImage.Format.Format_RGB888)  # pyside6: QImage.Format_RGB888
#         pixmap = QPixmap.fromImage(image)
#         win.image_lbl.setPixmap(pixmap)
#
#
# def update_frame(self, win):
#     ret, frame = self.video_capture.read()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         h, w, ch = frame.shape
#         image = QImage(frame.data, w, h, QImage.Format.Format_RGB888)  # pyside6: QImage.Format_RGB888
#         pixmap = QPixmap.fromImage(image)
#         win.image_lbl.setPixmap(pixmap)

if __name__ == '__main__':
    app = QApplication([])
    # player = CaptureForm()
    #
    player = CamWin()
    player.show()
    app.exec()
#
#


