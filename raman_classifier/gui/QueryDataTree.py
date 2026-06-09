from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any

import numpy as np
import tables

from PySide6.QtCore import Qt

from .DataTree import DataTree
from .common.CheckableTree import CheckableTreeItem

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class QueryDataTree(DataTree):

    def __init__(self, main_window:MainWindow)->None:
        DataTree.__init__(self, main_window, "query", "sheet_name", "待分类数据")
        self.setColumnCount(3)
        self.setHeaderLabels(["表单名", "分类结果", "测试准确率"])
        self.setMinimumWidth(300)
        header_item = self.headerItem()
        for i in range(self.columnCount()):
            header_item.setTextAlignment(i, Qt.AlignmentFlag.AlignCenter)

        self.reload()
    
    def classify_all(self)->None:
        classifier = self.main_window.api
        is_proto:bool = self.main_window.is_proto
        is_deep:bool = self.main_window.is_deep
        if is_deep:
            features_key:str = "normalized_data"
        else:
            if is_proto:
                features_key:str = "proto_pca_features"
            else:
                features_key:str = "pca_features"

        features = []
        groups_info:List[Dict[str, Any]] = []
        for class_item in self:
            for row_item in class_item:
                row_node:tables.Group = self.main_window.api.project_file.get_node(row_item.userData[0]["row_node_path"])
                if is_proto:
                    features.append(row_node[features_key].read())
                else:
                    features.append(row_node[features_key].read())
            groups_info.append(len(class_item))
        features = np.array(features)
        classifier.classify(features, groups_info)

    def classify_finished(self, classify_result:List[Dict[str, Any]])->None:
        classifier = self.main_window.api
        project_file = classifier.project_file

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        for class_item, group_info in zip(self, classify_result):
            label:str = group_info["best_label"]
            confidence:float = group_info["best_confidence"]
            labels:np.ndarray = group_info["labels"]
            confidences:np.ndarray = group_info["confidences"]
            outputs:np.ndarray = group_info["net_outputs"]

            class_item.setText(1, label)
            class_item.setText(2, f"{(100*confidence):.2f}%")
            class_item.userData[0]["label"] = label
            class_item.userData[0]["confidence"] = confidence
            class_node:tables.Group = self.main_window.api.project_file.get_node(class_item.userData[0]["class_node_path"])
            project_file.set_attr(class_node, "label", label)
            project_file.set_attr(class_node, "confidence", confidence)
            for i, row_item in enumerate(class_item):
                row_node:tables.Group = self.main_window.api.project_file.get_node(row_item.userData[0]["row_node_path"])
                row_item.setText(1, labels[i])
                row_item.setText(2, f"{(100*confidences[i]):.2f}%")
                row_item.userData[0]["label"] = labels[i]
                row_item.userData[0]["confidence"] = confidences[i]
                project_file.set_attr(row_node, "label", labels[i])
                project_file.set_attr(row_node, "confidence", confidences[i])
                if "net_output" in row_node:
                    project_file.remove_node(row_node, "net_output")

                project_file.create_array(row_node, "net_output", outputs[i, :])

        self.blockSignals(old_block_signals)
