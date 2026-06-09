from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QGroupBox, QVBoxLayout

from .PlotOptionsWidget import PlotOptionsWidget
from .FeaturePeaksTable import FeaturePeaksTable
from .OperationWidget import OperationWidget

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class RightSidebar(QSplitter):

    def __init__(self, main_window:MainWindow)->None:
        QSplitter.__init__(self, Qt.Orientation.Vertical)
        self.main_window = main_window

        self.operation_widget = OperationWidget(main_window)
        self.plot_options_widget = PlotOptionsWidget(main_window)

        groupbox = QGroupBox("特征峰", self)
        vlayout_groupbox = QVBoxLayout()
        self.feature_peaks_table = FeaturePeaksTable(self)
        vlayout_groupbox.addWidget(self.feature_peaks_table)
        groupbox.setLayout(vlayout_groupbox)

        self.addWidget(self.operation_widget)
        self.addWidget(self.plot_options_widget)
        self.addWidget(groupbox)

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 1)
        self.setStretchFactor(2, 2)

    def set_as_trainning(self, trainning:bool)->None:
        self.operation_widget.set_as_trainning(trainning)

    def set_as_deleting(self, deleting:bool)->None:
        self.operation_widget.set_as_deleting(deleting)

    def set_as_classifing(self, classifing:bool)->None:
        self.operation_widget.set_as_classifing(classifing)
