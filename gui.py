from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic

from base import context

from logic import openFile, textExtract, summary


class MainWindow(QMainWindow):
    def __init__(self):
        self.summarizeList = None
        super().__init__()
        # Loading the .ui file from the resources
        self.ui = uic.loadUi(context.get_resource("summarizon.ui"), self)

        # Binding the button to the print_data function defined in logic.py
        self.getFileButton.clicked.connect(lambda: openFile(self))
        #self.startSummarizing.clicked.connect(print_data())
        self.extractText.clicked.connect(lambda: textExtract(self))
        self.startSummarizing.clicked.connect(lambda: summary(self))