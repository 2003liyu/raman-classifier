from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class AskToTrainDialog(QDialog):

    def __init__(self, parent:QWidget=None)->None:
        QDialog.__init__(self, parent)

        vlayout:QVBoxLayout = QVBoxLayout()
        self.label_prompt:QLabel = QLabel("", self)
        vlayout.addWidget(self.label_prompt)

        hlayout:QHBoxLayout = QHBoxLayout()
        self.label_train_prompt:QLabel = QLabel("训练轮数", self)
        hlayout.addWidget(self.label_train_prompt)
        self.lineedit_epochs = QLineEdit(self)
        self.lineedit_epochs.setValidator(QIntValidator(bottom=3, top=1000))
        self.lineedit_epochs.setText("100")
        hlayout.addWidget(self.lineedit_epochs)
        hlayout.addWidget(QLabel("轮"))
        vlayout.addLayout(hlayout)

        vlayout.addStretch()

        hlayout:QHBoxLayout = QHBoxLayout()
        self.button_confirm = QPushButton("确定", self)
        self.button_cancel = QPushButton("取消", self)
        hlayout.addStretch()
        hlayout.addWidget(self.button_confirm)
        hlayout.addWidget(self.button_cancel)
        vlayout.addLayout(hlayout)

        self.setLayout(vlayout)

        self.epochs:int = 0
        self.button_confirm.clicked.connect(self.on_confirm)
        self.button_cancel.clicked.connect(self.on_cancel)

    def on_cancel(self):
        self.epochs = 0
        self.close()

    def on_confirm(self):
        text:str = self.lineedit_epochs.text().strip("\r\n\t ")
        if not text:
            QMessageBox.warning(self, "警告", "训练轮数不能为空！")
            return
        
        try:
            epochs:int = int(text)
        except BaseException as e:
            QMessageBox.warning(self, "警告", "训练轮数必须为整数！")
            return
        
        if epochs < 3 or epochs > 1000:
            QMessageBox.warning(self, "警告", "训练轮数只能在 3~1000 之间")
            return
        
        self.epochs = epochs
        self.close()