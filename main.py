from PyQt6.QtWidgets import QApplication

from widgets.scanner_win import ScannerWin

if __name__ == '__main__':
    app = QApplication([])
    # TODO: Unable to set icon.
    # app.setWindowIcon(QIcon('resources/qr-code.png'))
    scanner = ScannerWin(0,True)
    scanner.show()
    app.exec()
