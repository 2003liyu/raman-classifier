import sys

from PySide6.QtWidgets import QApplication

# from qt_material import apply_stylesheet

from .MainWindow import MainWindow


def main():
    app = QApplication(sys.argv)
    # apply_stylesheet(app, theme='dark_amber.xml')

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
