from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTabWidget, QWidget, QSplitter, QGroupBox, QVBoxLayout

from .common import PlotWidget


class PlotTabWidget(QTabWidget):

    def __init__(self, parent:QWidget=None)->None:
        QTabWidget.__init__(self, parent)

        self.raman_plot_widget:PlotWidget = PlotWidget(self)
        self.raman_plot_widget.xlabel("拉曼位移 (cm$^{-1}$)")
        self.raman_plot_widget.ylabel("拉曼散射光强度")

        vsplitter = QSplitter(Qt.Orientation.Vertical, self)

        groupbox:QGroupBox = QGroupBox("损失变化曲线图")
        vlayout:QVBoxLayout = QVBoxLayout()
        self.loss_plot_widget:PlotWidget = PlotWidget(self)
        self.loss_plot_widget.xlabel("训练轮数")
        self.loss_plot_widget.ylabel("交叉熵损失")
        vlayout.addWidget(self.loss_plot_widget)
        groupbox.setLayout(vlayout)
        vsplitter.addWidget(groupbox)

        groupbox:QGroupBox = QGroupBox("准确率变化曲线图")
        vlayout:QVBoxLayout = QVBoxLayout()
        self.acc_plot_widget:PlotWidget = PlotWidget(self)
        self.acc_plot_widget.xlabel("训练轮数")
        self.acc_plot_widget.ylabel("准确率")
        vlayout.addWidget(self.acc_plot_widget)
        groupbox.setLayout(vlayout)
        vsplitter.addWidget(groupbox)

        self.addTab(self.raman_plot_widget, "拉曼光谱图")
        self.addTab(vsplitter, "训练过程图")