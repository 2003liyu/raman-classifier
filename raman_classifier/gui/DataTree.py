from __future__ import annotations

import textwrap
from typing import Optional, List, TYPE_CHECKING, Dict, Any, Tuple

import numpy as np
from matplotlib.lines import Line2D

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QMenu, QMessageBox

from .common import CheckableTree, CheckableTreeItem, ProgressDialog

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class DataTree(CheckableTree):

    def __init__(self, main_window:MainWindow, key:str, group_text_key:str, title:str)->None:
        CheckableTree.__init__(self)
        self.main_window = main_window

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_right_clicked)

        self.__add_data_func = getattr(self.main_window.api, f"add_{key}_data")
        self.__remove_group_func = getattr(self.main_window.api, f"remove_{key}_class")
        self.__remove_row_func = getattr(self.main_window.api, f"remove_{key}_row")
        self.__clear_func = getattr(self.main_window.api, f"clear_{key}_classes")
        self.__key:str = key
        self.__group_text_key:str = group_text_key
        self.__title:str = title
        self.__data_info_name:str = f"{key}_data_info"
        self.__right_clicked_item:Optional[CheckableTreeItem] = None
        self.__item_menu:Optional[QMenu] = None
        self.__background_menu:Optional[QMenu] = None
        self.__progress_dialog:Optional[ProgressDialog] = None
        self.__current_line_type:str = "line_full_raw_data"
        self.__lines_visible:bool = True

        self.currentItemChanged.connect(self.on_current_item_changed)
        self.itemChanged.connect(self.on_item_changed)
        self.itemsChanged.connect(self.on_items_changed)

        pca_reconstructed_data_changed_signal = getattr(self.main_window.api, f"{key}_pca_reconstructed_data_changed")
        pca_reconstructed_data_changed_signal.connect(self.on_pca_reconstructed_data_changed)

    def reload(self):
        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        self.clear()
        data_info = getattr(self.main_window.api.project_file, self.__data_info_name)
        for data_info in data_info:
            text:str = data_info[self.__group_text_key]
            class_item = self[text]
            class_item.userData[0] = data_info
            if self.columnCount() == 3:
                label:str = data_info["label"]
                confidence:float = data_info["confidence"]
                if label:
                    class_item.setText(1, label)
                    class_item.setText(2, f"{100*confidence:.2f}%")

            rows_info = data_info["rows_info"]
            for row_info in rows_info:
                number:int = row_info["number"]
                row_item = class_item[str(number)]
                row_item.userData[0] = row_info
                if self.columnCount() == 3:
                    label:str = row_info["label"]
                    confidence:float = row_info["confidence"]
                    if label:
                        row_item.setText(1, label)
                        row_item.setText(2, f"{100*confidence:.2f}%")

        self.blockSignals(old_block_signals)

    @property
    def background_menu(self)->QMenu:
        if self.__background_menu is None:
            self.__background_menu = QMenu(self)
            self.__background_menu.addAction("导入" + self.__title).triggered.connect(self.import_data)
            self.__background_menu.addAction("清空" + self.__title).triggered.connect(self.on_clear)

        return self.__background_menu
    
    @property
    def item_menu(self)->QMenu:
        if self.__item_menu is None:
            self.__item_menu = QMenu(self)
            self.__item_menu.addAction("删除").triggered.connect(self.on_delete_item)
            self.__item_menu.addAction("导入" + self.__title).triggered.connect(self.import_data)
            self.__item_menu.addAction("清空" + self.__title).triggered.connect(self.on_clear)

        return self.__item_menu

    def on_right_clicked(self, position:QPoint)->None:
        self.__right_clicked_item:Optional[CheckableTreeItem] = self.itemAt(position)
        if self.__right_clicked_item is None:
            self.background_menu.exec(self.mapToGlobal(position))
        else:
            self.item_menu.exec(self.mapToGlobal(position))

    def clear(self):
        lines_to_be_removed = []
        for top_item in self:
            for row_item in top_item:
                row_info = row_item.userData[0]
                for key in row_info:
                    if key.startswith("line_"):
                        lines_to_be_removed.append(row_info[key])

        if lines_to_be_removed:
            self.main_window.plot_tab_widget.raman_plot_widget.remove(lines_to_be_removed)

        CheckableTree.clear(self)

    def on_delete_item(self)->None:
        if self.__right_clicked_item is None:
            return

        self.main_window.set_as_deleting(True)
        self.main_window.api.remove_error.connect(self.on_delete_error)
        self.main_window.api.remove_finished.connect(self.on_delete_finished)

        item_data = self.__right_clicked_item.userData[0]
        if self.__right_clicked_item.parent() is None:
            self.__remove_group_func(item_data["class_node_path"])
        else:
            self.__remove_row_func(item_data["row_node_path"])

    def on_delete_finished(self)->None:
        is_group:bool = (self.__right_clicked_item.parent() is None)
        item_data = self.__right_clicked_item.userData[0]
        lines_to_be_removed:List[Line2D] = []
        if is_group:
            for i in range(self.__right_clicked_item.childCount()):
                row_item = self.__right_clicked_item.child(i)
                row_info = row_item.userData[0]
                for key in row_info:
                    if key.startswith("line_"):
                        lines_to_be_removed.append(row_info[key])
        else:
            for key in item_data:
                if key.startswith("line_"):
                    lines_to_be_removed.append(item_data[key])

        if lines_to_be_removed:
            self.main_window.plot_tab_widget.raman_plot_widget.remove(lines_to_be_removed)
        
        parent_item:CheckableTreeItem = self.__right_clicked_item.parent()
        if parent_item is None:
            self.takeTopLevelItem(self.indexOfTopLevelItem(self.__right_clicked_item))
            if self.__key == "train":
                self.update_class_id()
        else:
            parent_item.removeChild(self.__right_clicked_item)

        self.main_window.set_as_deleting(False)
        self.main_window.api.remove_error.disconnect(self.on_delete_error)
        self.main_window.api.remove_finished.disconnect(self.on_delete_finished)

    def on_delete_error(self, error_message:str)->None:
        self.main_window.set_as_deleting(False)
        self.main_window.api.remove_error.disconnect(self.on_delete_error)
        self.main_window.api.remove_finished.disconnect(self.on_delete_finished)
        QMessageBox.critical(self.main_window, "删除错误", error_message)

    def update_class_id(self)->None:
        for i in range(self.topLevelItemCount()):
            class_item:CheckableTreeItem = self.topLevelItem(i)
            class_info = class_item.userData[0]
            class_info["class_id"] = i
            for j in range(class_item.childCount()):
                row_item:CheckableTreeItem = class_item.child(j)
                row_info = row_item.userData[0]
                row_info["class_id"] = i

    def on_item_changed(self, item:CheckableTreeItem)->None:
        if item.parent() is None:
            return
        
        self.check_item(item, item.checkState(0) == Qt.CheckState.Checked)

    def on_items_changed(self, items:List[CheckableTreeItem])->None:
        something_changed:bool = False
        for item in items:
            if item.parent() is None:
                continue
            self.check_item(item, item.checkState(0) == Qt.CheckState.Checked, False)
            something_changed = True

        if something_changed:
            self.main_window.plot_tab_widget.raman_plot_widget.update()

    def check_item(self, row_item:CheckableTreeItem, checked:bool, update:bool=True)->None:
        row_info:Dict[str, Any] = row_item.userData[0]

        line_type:str = self.main_window.right_sidebar.plot_options_widget.line_type
        key = line_type[len("line_"):]
        something_changed:bool = False

        if line_type in row_info:
            for line_key in row_info:
                if not line_key.startswith("line_"):
                    continue
                
                line:Line2D = row_info[line_key]
                if line_key != line_type:
                    if line.get_visible():
                        line.set_visible(False)
                        something_changed = True
                else:
                    if line.get_visible() != checked:
                        line.set_visible(checked)
                        something_changed = True
        elif checked:
            x = self.main_window.api.project_file.raman_shift
            row_node = self.main_window.api.project_file.get_node(row_info["row_node_path"])
            if key in row_node:
                y = row_node[key].read()
                if key in ["normalized_data", "proto_pca_reconstructed_data", "pca_reconstructed_data", "gaussian_fitted_data"]:
                    y *= 1000

                if key == "full_raw_data":
                    x = self.main_window.api.project_file.full_raman_shift
            else:
                y = np.full(x.shape, np.nan)

            row_info[line_type] = self.main_window.plot_tab_widget.raman_plot_widget.plot(x, y, label=row_info[self.__group_text_key], update=False)
            something_changed = True

        if something_changed and update:
            self.main_window.plot_tab_widget.raman_plot_widget.update()

    def set_lines_visible(self, visible:bool)->bool:
        if self.__lines_visible == visible:
            return
        
        self.__lines_visible = visible

        something_changed:bool = False
        for class_item in self:
            for row_item in class_item:
                row_info:Dict[str, Any] = row_item.userData[0]
                for key in row_info:
                    if not key.startswith("line_"):
                        continue

                    line:Line2D = row_info[key]
                    target_visible = (visible and key == self.__current_line_type and row_item.checkState(0) == Qt.CheckState.Checked)
                    something_changed = (something_changed or line.get_visible() != target_visible)
                    line.set_visible(target_visible)

        return something_changed

    def on_clear(self)->None:
        self.main_window.api.clear_finished.connect(self.on_clear_finished)
        self.main_window.api.clear_error.connect(self.on_clear_error)
        self.main_window.set_as_deleting(True)
        self.__clear_func()

    def on_clear_error(self, error_message:str)->None:
        self.main_window.api.clear_finished.disconnect(self.on_clear_finished)
        self.main_window.api.clear_error.disconnect(self.on_clear_error)
        self.main_window.set_as_deleting(False)
        QMessageBox.critical(self, f"清空{self.__title}错误", error_message)

    def on_clear_finished(self)->None:
        self.clear()

        self.main_window.api.clear_finished.disconnect(self.on_clear_finished)
        self.main_window.api.clear_error.disconnect(self.on_clear_error)
        self.main_window.set_as_deleting(False)

    def import_data(self)->None:
        file_names:List[str] = self.main_window.file_chooser.openFiles(f"导入{self.__title}", filter="Excel Files (*.xlsx)")
        if not file_names:
            return
        
        self.main_window.api.add_data_error.connect(self.on_add_data_error)
        self.main_window.api.add_data_finished.connect(self.on_add_data_finished)
        self.main_window.api.progress_changed.connect(self.on_progress_changed)
        self.main_window.api.progress_stage_changed.connect(self.on_progress_stage_changed)
        self.progress_dialog.show()

        self.__add_data_func(file_names)

    def on_progress_stage_changed(self, state:str):
        self.progress_dialog.text = f"正在{state}..."

    def on_progress_changed(self, progress:float):
        self.progress_dialog.progress = progress

    def on_add_data_error(self, error_message):
        self.main_window.api.add_data_error.disconnect(self.on_add_data_error)
        self.main_window.api.add_data_finished.disconnect(self.on_add_data_finished)
        self.main_window.api.progress_changed.disconnect(self.on_progress_changed)
        self.main_window.api.progress_stage_changed.disconnect(self.on_progress_stage_changed)
        self.progress_dialog.close()

        QMessageBox.critical(self.main_window, f"添加{self.__title}失败", error_message)


    def on_add_data_finished(self, result):
        self.main_window.api.add_data_error.disconnect(self.on_add_data_error)
        self.main_window.api.add_data_finished.disconnect(self.on_add_data_finished)
        self.main_window.api.progress_changed.disconnect(self.on_progress_changed)
        self.main_window.api.progress_stage_changed.disconnect(self.on_progress_stage_changed)
        self.progress_dialog.close()

        self.reload()
        self.main_window.right_sidebar.operation_widget.load_model_info(False)

        added_classes:List[str] = result["added_classes"]
        updated_classes:List[str] = result["updated_classes"]
        ignored_data:Dict[Tuple[str,str], List[int]] = result["ignored_data"]

        message:str = ""
        if len(added_classes) > 0:
            message += "成功添加以下类别：\n" + textwrap.fill(" ".join(added_classes), width=50) + "\n\n"

        if len(updated_classes) > 0:
            message += "成功更新以下类别：\n" + textwrap.fill(" ".join(updated_classes), width=50) + "\n\n"

        if len(ignored_data) > 0:
            message += "忽略以下数据，因为原来的数据中已包含：\n"
            for key in ignored_data:
                file_name, label = key
                indices = ignored_data[key]
                if indices:
                    for col in indices:
                        message += f"文件 {file_name} 中表单 {label} 的第 {col+1} 列\n"
                else:
                    message += f"文件 {file_name} 中表单 {label} 的所有列\n"

        if message != "":
            QMessageBox.information(self.main_window, f"成功添加{self.__title}", message)

    def on_current_item_changed(self, current_item:CheckableTreeItem)->None:
        if current_item is None or current_item.parent() is None:
            self.main_window.right_sidebar.feature_peaks_table.clear()
        else:
            row_info = current_item.userData[0]
            row_node = self.main_window.api.project_file.get_node(row_info["row_node_path"])
            self.main_window.right_sidebar.feature_peaks_table.set_peaks_info(
                row_node.peaks_amp, row_node.peaks_mu, row_node.peaks_sigma
            )

    def on_line_type_changed(self, line_type:str)->None:
        self.__current_line_type = line_type
        if not self.__lines_visible:
            return

        something_changed:bool = False
        for i in range(self.topLevelItemCount()):
            class_item = self.topLevelItem(i)
            if class_item.checkState(0) == Qt.CheckState.Unchecked:
                continue
            
            for j in range(class_item.childCount()):
                row_item:CheckableTreeItem = class_item.child(j)
                if row_item.checkState(0) == Qt.CheckState.Unchecked:
                    continue

                row_info = row_item.userData[0]
                for key in row_info:
                    if key.startswith("line_") and key != line_type and row_info[key].get_visible():
                        row_info[key].set_visible(False)
                        something_changed:bool = True

                if line_type in row_info:
                    if not row_info[line_type].get_visible():
                        row_info[line_type].set_visible(True)
                        something_changed:bool = True
                else:
                    key = line_type[len("line_"):]
                    x = self.main_window.api.project_file.raman_shift
                    row_node = self.main_window.api.project_file.get_node(row_info["row_node_path"])
                    if key in row_node:
                        y = row_node[key].read()
                        if line_type in ["line_normalized_data", "line_pca_reconstructed_data", "line_proto_pca_reconstructed_data", "line_gaussian_fitted_data"]:
                            y *= 1000
                        if line_type == "line_full_raw_data":
                            x = self.main_window.api.project_file.full_raman_shift
                    else:
                        y = np.full(x.shape, np.nan)

                    row_info[line_type] = self.main_window.plot_tab_widget.raman_plot_widget.plot(x, y, label=row_info[self.__group_text_key], update=False)
                    something_changed:bool = True

        if something_changed:
            self.main_window.plot_tab_widget.raman_plot_widget.update()

    def on_pca_reconstructed_data_changed(self, all_data:List[Tuple[str, np.ndarray]])->None:
        line_key:str = ("line_pca_reconstructed_data" if self.__key.startswith("proto") else "line_proto_pca_reconstructed_data")
        should_update:bool = False
        for row in all_data:
            path_items:List[str] = row[0].split("/")
            pca_reconstructed_data:np.ndarray = row[1]
            class_name:str = path_items[0]
            row_number:str = path_items[1]

            if (
                class_name not in self or
                row_number not in self[class_name] or
                0 not in self[class_name][row_number].userData or
                line_key not in self[class_name][row_number].userData[0]
            ):
                continue

            line:Line2D = self[class_name][row_number].userData[0][line_key]
            line.set_ydata(1000 * pca_reconstructed_data)
            if line.get_visible():
                should_update = True

        if should_update:
            self.main_window.plot_tab_widget.raman_plot_widget.update()

    @property
    def progress_dialog(self)->ProgressDialog:
        if self.__progress_dialog is None:
            self.__progress_dialog:ProgressDialog = ProgressDialog(self.main_window)
            self.__progress_dialog.title = "正在导入" + self.__title

        return self.__progress_dialog

    def set_as_trainning(self, trainning:bool)->None:
        for action in self.item_menu.actions():
            action.setEnabled(not trainning)

        for action in self.background_menu.actions():
            action.setEnabled(not trainning)

    def set_as_deleting(self, deleting:bool)->None:
        self.set_as_trainning(deleting)

    def set_as_classifing(self, classifing:bool)->None:
        self.set_as_trainning(classifing)