from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QGroupBox, QVBoxLayout

from .TrainDataTree import TrainDataTree
from .ProtoTrainDataTree import ProtoTrainDataTree
from .ProtoSupportDataTree import ProtoSupportDataTree
from .QueryDataTree import QueryDataTree

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class LeftSidebar(QSplitter):

    def __init__(self, main_window:MainWindow)->None:
        QSplitter.__init__(self, Qt.Orientation.Vertical, main_window)
        self.main_window:MainWindow = main_window

        self.train_data_tree:TrainDataTree = TrainDataTree(main_window)
        self.proto_train_data_tree:ProtoTrainDataTree = ProtoTrainDataTree(main_window)
        self.proto_support_data_tree:ProtoSupportDataTree = ProtoSupportDataTree(main_window)
        self.query_data_tree:QueryDataTree = QueryDataTree(main_window)

        self.train_groupbox:QGroupBox = QGroupBox("训练数据")
        vlayout:QVBoxLayout = QVBoxLayout()
        vlayout.addWidget(self.train_data_tree)
        self.train_groupbox.setLayout(vlayout)
        self.addWidget(self.train_groupbox)

        self.proto_train_groupbox:QGroupBox = QGroupBox("训练数据")
        vlayout:QVBoxLayout = QVBoxLayout()
        vlayout.addWidget(self.proto_train_data_tree)
        self.proto_train_groupbox.setLayout(vlayout)
        self.addWidget(self.proto_train_groupbox)

        self.proto_support_groupbox:QGroupBox = QGroupBox("支撑数据")
        vlayout:QVBoxLayout = QVBoxLayout()
        vlayout.addWidget(self.proto_support_data_tree)
        self.proto_support_groupbox.setLayout(vlayout)
        self.addWidget(self.proto_support_groupbox)

        self.query_groupbox:QGroupBox = QGroupBox("待分类数据")
        vlayout:QVBoxLayout = QVBoxLayout()
        vlayout.addWidget(self.query_data_tree)
        self.query_groupbox.setLayout(vlayout)
        self.addWidget(self.query_groupbox)

        self.set_as_proto(False)

    def set_as_proto(self, flag:bool)->None:
        something_changed:bool = False
        self.train_groupbox.setVisible(not flag)
        something_changed = (self.train_data_tree.set_lines_visible(not flag) or something_changed)

        self.proto_train_groupbox.setVisible(flag)
        something_changed = (self.proto_train_data_tree.set_lines_visible(flag) or something_changed)

        self.proto_support_groupbox.setVisible(flag)
        something_changed = (self.proto_support_data_tree.set_lines_visible(flag) or something_changed)

        if something_changed:
            self.main_window.plot_tab_widget.raman_plot_widget.update()

    def set_as_trainning(self, trainning:bool)->None:
        self.train_data_tree.set_as_trainning(trainning)
        self.proto_train_data_tree.set_as_trainning(trainning)
        self.proto_support_data_tree.set_as_trainning(trainning)
        self.query_data_tree.set_as_trainning(trainning)

    def set_as_deleting(self, deleting:bool)->None:
        self.train_data_tree.set_as_deleting(deleting)
        self.proto_train_data_tree.set_as_deleting(deleting)
        self.proto_support_data_tree.set_as_deleting(deleting)
        self.query_data_tree.set_as_deleting(deleting)

    def set_as_classifing(self, classifing:bool)->None:
        self.train_data_tree.set_as_classifing(classifing)
        self.proto_train_data_tree.set_as_classifing(classifing)
        self.proto_support_data_tree.set_as_classifing(classifing)
        self.query_data_tree.set_as_classifing(classifing)
