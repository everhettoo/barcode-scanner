from VideoPlayer import VideoPlayer

#
# import sys
# from PySide6.QtWidgets import QApplication, QWidget
# from PySide6.QtCore import Slot
# from PySide6.QtUiTools import QUiLoader
#
# class MyWidget(QWidget):
#     def __init__(self, ui_file, parent=None):
#         super().__init__(parent)
#         loader = QUiLoader()
#         self.window = loader.load(ui_file, self)
#         # self.setCentralWidget(self.window)
#         self.window.pushButton.clicked.connect(self.button_clicked)
#
#     @Slot()
#     def button_clicked(self):
#         print("Button clicked!")
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     widget = MyWidget("form.ui")
#     widget.show()
#     sys.exit(app.exec())

#
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#
#         button = QPushButton("Press Me!")
#         button.setCheckable(True)
#         button.clicked.connect(self.the_button_was_clicked)
#
#         # Set the central widget of the Window.
#         self.setCentralWidget(button)
#
#     def the_button_was_clicked(self):
#         print("Clicked!")
#
# app = QApplication(sys.argv)
#
# window = MainWindow()
# window.show()
#
# app.exec()

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QApplication, QVBoxLayout, QWidget


if __name__ == '__main__':
        app = QApplication([])
        player = VideoPlayer()
        player.show()
        app.exec()
# import sys
# from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
# from PySide6.QtGui import QPalette, QColor
#
# class Color(QWidget):
#     def __init__(self, color):
#         super().__init__()
#         self.setAutoFillBackground(True)
#
#         palette = self.palette()
#         palette.setColor(QPalette.ColorRole.Window, QColor(color))
#         self.setPalette(palette)
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("My App")
#
#         layout1 = QHBoxLayout()
#         layout2 = QVBoxLayout()
#         layout3 = QVBoxLayout()
#
#         layout2.addWidget(Color('red'))
#         layout2.addWidget(Color('yellow'))
#         layout2.addWidget(Color('purple'))
#
#         layout1.addLayout( layout2 )
#
#         layout1.addWidget(Color('green'))
#
#         layout3.addWidget(Color('red'))
#         layout3.addWidget(Color('purple'))
#
#         layout1.addLayout( layout3 )
#
#         widget = QWidget()
#         widget.setLayout(layout1)
#         self.setCentralWidget(widget)
#
# app = QApplication(sys.argv)
# window = MainWindow()
# window.show()
# app.exec()