import sys

from base import context
from gui import MainWindow

if __name__ == "__main__":
    window = MainWindow()
    window.resize(250, 150)
    window.show()
    exit_code = context.app.exec_()
    sys.exit(exit_code)