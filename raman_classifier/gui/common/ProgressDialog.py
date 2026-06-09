from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QWidget, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout


class ProgressDialog(QDialog):

    def __init__(self, parent:QWidget=None):
        QDialog.__init__(self, parent)

        vlayout_main:QVBoxLayout = QVBoxLayout()
        vlayout_main.addStretch()

        hlayout:QHBoxLayout = QHBoxLayout()
        self.label:QLabel = QLabel(self)
        hlayout.addWidget(self.label)
        hlayout.addStretch()
        vlayout_main.addLayout(hlayout)
        vlayout_main.addSpacing(10)

        self.progress_bar:QProgressBar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(10000)
        vlayout_main.addWidget(self.progress_bar)
        vlayout_main.addStretch()
        
        self.setLayout(vlayout_main)
        self.setFixedSize(400, 150)

        self.setModal(True)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

    @property
    def progress(self)->float:
        return (self.progress_bar.value() - self.progress_bar.minimum()) / (self.progress_bar.maximum() - self.progress_bar.minimum())
    
    @progress.setter
    def progress(self, value:float)->None:
        int_value:int = int(value * (self.progress_bar.maximum() - self.progress_bar.minimum()) + self.progress_bar.minimum())
        self.progress_bar.setValue(int_value)

    @property
    def text(self)->str:
        return self.label.text()
    
    @text.setter
    def text(self, text:str)->None:
        self.label.setText(text)
    
    @property
    def title(self)->str:
        return self.windowTitle()
    
    @title.setter
    def title(self, title:str)->None:
        self.setWindowTitle(title)