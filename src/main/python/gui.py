from PyQt5.QtWidgets import QMainWindow
from PyQt5 import uic
from base import context
from logic import openFile, textExtract, summary, kill
from PyQt5.QtWidgets import (QWidget, QApplication, QProgressBar, QMainWindow,QHBoxLayout, QPushButton)
from PyQt5.QtCore import (Qt, QObject, pyqtSignal, pyqtSlot, QRunnable, QThreadPool)
import time

class WorkerSignals(QObject):
    progress = pyqtSignal(int)


class JobRunner(QRunnable):
    
    signals = WorkerSignals()
    
    def __init__(self):
        super().__init__()
        
        self.is_paused = False
        self.is_killed = False
        
    @pyqtSlot()
    def run(self):
        for n in range(100):
            self.signals.progress.emit(n + 1)
            time.sleep(0.1)
            
            while self.is_paused:
                time.sleep(0)
                
            if self.is_killed:
                break
                
    def pause(self):
        self.is_paused = True
        
    def resume(self):
        self.is_paused = False
        
    def kill(self):
        self.is_killed = True

#possibly add a class variable to this including the summarization model so main initializes the model creation and training
class MainWindow(QMainWindow):
    def __init__(self):
        self.summarizeList = None
        self.is_paused = False
        self.is_killed = False
        super().__init__()
        # Loading the .ui file from the resources
        self.ui = uic.loadUi(context.get_resource("summarizon.ui"), self)

        # Binding the button to the print_data function defined in logic.py

        self.getFileButton.clicked.connect(lambda: openFile(self))
        self.extractText.clicked.connect(lambda: textExtract(self))
        self.startSummarizing.clicked.connect(lambda: summary(self))
        self.stopSummarizing.clicked.connect(lambda: kill(self))

        