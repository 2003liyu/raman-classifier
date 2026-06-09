from typing import Union

from PySide6.QtGui import QIntValidator
from PySide6.QtWidgets import QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox


class AskToContinueTrainDialog(QDialog):

    def __init__(self, parent:QWidget=None)->None:
        QDialog.__init__(self, parent)

        vlayout:QVBoxLayout = QVBoxLayout()

        hlayout:QHBoxLayout = QHBoxLayout()
        hlayout.addWidget(QLabel("训练已结束，验证集准确率"))
        self.acc_label:QLabel = QLabel("", self)
        hlayout.addWidget(self.acc_label)
        hlayout.addWidget(QLabel("你可以："))
        vlayout.addLayout(hlayout)

        vlayout.addWidget(QLabel("1. 采纳训练结果，保存模型"))
        vlayout.addWidget(QLabel("2. 拒绝训练结果，放弃保存"))

        hlayout:QHBoxLayout = QHBoxLayout()
        hlayout.addWidget(QLabel("3. 继续训练，再训练"))
        self.lineedit_add_epochs = QLineEdit(self)
        self.lineedit_add_epochs.setValidator(QIntValidator(bottom=3, top=1000))
        self.lineedit_add_epochs.setText("100")
        hlayout.addWidget(self.lineedit_add_epochs)
        hlayout.addWidget(QLabel("轮"))
        vlayout.addLayout(hlayout)

        vlayout.addStretch()

        hlayout:QHBoxLayout = QHBoxLayout()
        self.button_accept = QPushButton("采纳", self)
        self.button_reject = QPushButton("拒绝", self)
        self.button_continue = QPushButton("继续训练", self)
        hlayout.addStretch()
        hlayout.addWidget(self.button_accept)
        hlayout.addWidget(self.button_reject)
        hlayout.addWidget(self.button_continue)
        vlayout.addLayout(hlayout)

        self.setLayout(vlayout)
        self.setWindowTitle("训练已结束")

        self.choice:Union[bool, int] = True
        self.button_accept.clicked.connect(self.on_accept)
        self.button_reject.clicked.connect(self.on_reject)
        self.button_continue.clicked.connect(self.on_continue)

    def on_accept(self):
        self.choice = True
        self.close()

    def on_reject(self):
        self.choice = False
        self.close()

    def on_continue(self):
        text:str = self.lineedit_add_epochs.text().strip("\r\n\t ")
        if not text:
            QMessageBox.warning(self, "警告", "训练轮数不能为空！")
            return
        
        try:
            add_epochs:int = int(text)
        except BaseException as e:
            QMessageBox.warning(self, "警告", "训练轮数必须为整数！")
            return
        
        if add_epochs < 3 or add_epochs > 1000:
            QMessageBox.warning(self, "警告", "训练轮数只能在 3~1000 之间")
            return
        
        self.choice = add_epochs
        self.close()