from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea, QWidget, QVBoxLayout, QLabel


class ScrollLabel(QScrollArea):
    """
    Scroll widget for QScrollArea for displaying traces.
    Code was taken and modified from: https://www.geeksforgeeks.org/pyqt5-scrollable-label/
    """
    LABEL_STYLE = "font-size:16px;"

    # LABEL_STYLE = "border: 2px solid yellow; font-size:16px;"

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
        self.label.setStyleSheet(self.LABEL_STYLE)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # making label multi-line
        self.label.setWordWrap(True)

        # self.setWidget(self.label)
        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def setText(self, text):
        self.label.setText(text)

    def text(self):
        return self.label.text()
