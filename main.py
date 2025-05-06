from PyQt6.QtWidgets import QApplication

from scanner_win import ScannerWin

if __name__ == '__main__':
    app = QApplication([])

    player = ScannerWin()
    player.show()
    app.exec()
#
#
