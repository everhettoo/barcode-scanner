# importing libraries
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import sys

from PyQt6.QtWidgets import QApplication


# class for scrollable label
class ScrollLabel(QScrollArea):

	# constructor
	def __init__(self, *args, **kwargs):
		QScrollArea.__init__(self, *args, **kwargs)

		# making widget resizable
		self.setWidgetResizable(True)

		# making qwidget object
		content = QWidget(self)
		self.setWidget(content)

		# vertical box layout
		lay = QVBoxLayout(content)

		# creating label
		self.label = QLabel(content)

		# setting alignment to the text
		# self.label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

		# making label multi-line
		self.label.setWordWrap(True)

		# adding label to the layout
		lay.addWidget(self.label)

	# the setText method
	def setText(self, text):
		# setting text to the label
		self.label.setText(text)

class Window(QMainWindow):

	def __init__(self):
		super().__init__()

		# setting title
		self.setWindowTitle("Python ")

		# setting geometry
		self.setGeometry(100, 100, 600, 400)

		# calling method
		self.UiComponents()

		# showing all the widgets
		self.show()

	# method for widgets
	def UiComponents(self):
		# text to show in label
		text = "There are so many options provided by Python to develop GUI " \
			"application and PyQt5 is one of them. PyQt5 is cross-platform " \
			"GUI toolkit, a set of python bindings for Qt v5. One can develop" \
			" an interactive desktop application with so much ease because " \
			"of the tools and simplicity provided by this library.A GUI application" \
			" consists of Front-end and Back-end. PyQt5 has provided a tool called " \
			"‘QtDesigner’ to design the front-end by drag and drop method so that " \
			"development can become faster and one can give more time on back-end stuff. "

		# creating scroll label
		label = ScrollLabel(self)

		# setting text to the label
		label.setText(text)

		# setting geometry
		label.setGeometry(100, 100, 200, 80)



# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

window.show()

# start the app
sys.exit(App.exec())
