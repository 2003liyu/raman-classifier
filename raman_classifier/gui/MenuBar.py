from __future__ import annotations

import traceback
from datetime import datetime
from typing import TYPE_CHECKING, List

from PySide6.QtWidgets import QMenuBar, QMessageBox

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class MenuBar(QMenuBar):

    def __init__(self, main_window:MainWindow)->None:
        QMenuBar.__init__(self, main_window)

        self.main_window:MainWindow = main_window


        # 项目菜单
        file_menu = self.addMenu("项目(&P)")
        self.action_new = file_menu.addAction("新建项目")
        self.action_new.setShortcut("Ctrl+N")
        self.action_new.triggered.connect(self.on_new)

        self.action_open = file_menu.addAction("打开项目")
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.triggered.connect(self.on_open)

        self.action_save = file_menu.addAction("保存项目")
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.triggered.connect(self.on_save)

        self.action_save_as = file_menu.addAction("项目另存为")
        self.action_save_as.setShortcut("Ctrl+Shift+S")
        self.action_save_as.triggered.connect(self.on_save_as)

        # 导入菜单
        import_menu = self.addMenu("导入(&I)")
        self.action_import_train = import_menu.addAction("导入训练数据")
        self.action_import_train.triggered.connect(self.main_window.left_sidebar.train_data_tree.import_data)

        self.action_import_proto_train = import_menu.addAction("导入原型网络训练数据")
        self.action_import_proto_train.triggered.connect(self.main_window.left_sidebar.proto_train_data_tree.import_data)

        self.action_import_proto_support = import_menu.addAction("导入原型网络支撑数据")
        self.action_import_proto_support.triggered.connect(self.main_window.left_sidebar.proto_support_data_tree.import_data)

        self.action_import_query = import_menu.addAction("导入待分类数据")
        self.action_import_query.triggered.connect(self.main_window.left_sidebar.query_data_tree.import_data)

    def set_as_trainning(self, trainning:bool)->None:
        self.action_new.setEnabled(not trainning)
        self.action_open.setEnabled(not trainning)
        self.action_save.setEnabled(not trainning)
        self.action_save_as.setEnabled(not trainning)

        self.action_import_train.setEnabled(not trainning)
        self.action_import_proto_train.setEnabled(not trainning)
        self.action_import_proto_support.setEnabled(not trainning)
        self.action_import_query.setEnabled(not trainning)

    def set_as_classifing(self, classifing:bool)->None:
        self.action_new.setEnabled(not classifing)
        self.action_open.setEnabled(not classifing)
        self.action_save.setEnabled(not classifing)
        self.action_save_as.setEnabled(not classifing)

        self.action_import_train.setEnabled(not classifing)
        self.action_import_proto_train.setEnabled(not classifing)
        self.action_import_proto_support.setEnabled(not classifing)
        self.action_import_query.setEnabled(not classifing)

    def set_as_deleting(self, deleting:bool)->None:
        self.action_new.setEnabled(not deleting)
        self.action_open.setEnabled(not deleting)
        self.action_save.setEnabled(not deleting)
        self.action_save_as.setEnabled(not deleting)

        self.action_import_train.setEnabled(not deleting)
        self.action_import_proto_train.setEnabled(not deleting)
        self.action_import_proto_support.setEnabled(not deleting)
        self.action_import_query.setEnabled(not deleting)

    def on_open(self)->None:
        file_name:str = self.main_window.file_chooser.openFile("打开项目", filter="HDF5 Files (*.h5)")
        if not file_name:
            return
        
        try:
            self.main_window.api.open(file_name)
        except BaseException as e:
            error_message:str = traceback.format_exc()
            print(error_message)
            QMessageBox.critical(self, "打开项目失败", error_message)
            return

    def on_save(self)->None:
        if self.main_window.api.is_temp:
            self.on_save_as()
            return
        
        try:
            self.main_window.api.project_file.save()
        except BaseException as e:
            error_message:str = traceback.format_exc()
            print(error_message)
            QMessageBox.critical(self, "保存项目失败", error_message)
            return

    def on_save_as(self)->None:
        default_file_name = self.main_window.file_chooser.last_folder + "/光谱分类-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name:List[str] = self.main_window.file_chooser.saveFile("保存项目", dir=default_file_name, filter="HDF5 Files (*.h5)")
        if not file_name:
            return

        try:
            self.main_window.api.project_file.save_as(file_name)
        except BaseException as e:
            error_message:str = traceback.format_exc()
            print(error_message)
            QMessageBox.critical(self, "另存项目失败", error_message)
            return
        
    def ask_to_save(self)->None:
        if self.main_window.api.is_temp and self.main_window.api.is_dirty:
            button = QMessageBox.question(self, "问题", "当前项目已修改，是否保存当前项目？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if button == QMessageBox.StandardButton.Yes:
                self.on_save()

    def on_new(self)->None:
        self.ask_to_save()

        try:
            self.main_window.api.open_new()
        except BaseException as e:
            error_message:str = traceback.format_exc()
            print(error_message)
            QMessageBox.critical(self, "新建项目失败", error_message)
            return


