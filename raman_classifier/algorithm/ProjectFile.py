from PySide6.QtCore import QObject, Signal

import warnings
import os
import shutil
import time
from typing import Optional, Dict, List, Union, Any, Tuple


import numpy as np
import tables
from tables import NaturalNameWarning

import torch

from ..common.utils import echo
from .PCA import PCA
from .models import BaseNet

warnings.filterwarnings('ignore', category=NaturalNameWarning)


class ProjectFile(QObject):

    project_changed = Signal()
    project_saved = Signal()
    project_name_changed = Signal(str)
    train_pca_reconstructed_data_changed = Signal(list)
    proto_train_pca_reconstructed_data_changed = Signal(list)
    proto_support_pca_reconstructed_data_changed = Signal(list)
    query_pca_reconstructed_data_changed = Signal(list)
    query_proto_pca_reconstructed_data_changed = Signal(list)

    def __init__(self, file_name:str=""):
        QObject.__init__(self)
        self.__active_project_txt:str = os.path.dirname(os.path.abspath(__file__)) + "/../data/active_project.txt"
        self.__file_name:str = ""
        self.__file:Optional[tables.File] = None
        self.__raman_shift:Optional[np.ndarray] = None
        self.__full_raman_shift:Optional[np.ndarray] = None
        self.__dirty:bool = False

        if file_name != "":
            self.open(file_name)

    def __del__(self)->None:
        self.close()

    @property
    def is_dirty(self)->bool:
        return self.__dirty

    @property
    def tables_file(self)->tables.File:
        return self.__file

    @property
    def file_name(self)->str:
        return self.__file_name
    
    @property
    def is_open(self)->bool:
        return (self.__file is not None)

    def close(self)->None:
        if self.__file is None:
            return
        
        echo(self.__file_name, self.__active_project_txt)
        self.__file.close()
        self.__file:Optional[tables.File] = None
        self.__raman_shift:Optional[np.ndarray] = None
        self.__full_raman_shift:Optional[np.ndarray] = None

    def open(self, file_name:str)->bool:
        abs_name:str = os.path.abspath(file_name).replace("\\", "/")
        if self.__file_name == abs_name:
            return False
        
        self.close()
        file_exists:bool = os.path.isfile(abs_name)

        abs_folder:str = os.path.dirname(abs_name)
        if not os.path.isdir(abs_folder):
            os.makedirs(abs_folder)

        self.__file:Optional[tables.File] = tables.open_file(abs_name, mode="a")

        if file_exists:
            self.__full_raman_shift:Optional[np.ndarray] = self.__file.root.full_raman_shift.read()
            self.__raman_shift:Optional[np.ndarray] = self.__file.root.raman_shift.read()
        else:
            train_data = self.create_group("/", "train_data")
            proto_train_data = self.create_group("/", "proto_train_data")
            proto_support_data = self.create_group("/", "proto_support_data")
            query_data = self.create_group("/", "query_data")
            self.create_group("/", "protos_data")

            self.create_group(train_data, "classes")
            self.create_group(proto_train_data, "classes")
            self.create_group(proto_support_data, "classes")
            self.create_group(query_data, "classes")
            
            self.create_group(train_data, "label_to_id")
            self.create_group(train_data, "id_to_label")

            self.create_group(proto_train_data, "label_to_id")
            self.create_group(proto_train_data, "id_to_label")

            self.create_group(proto_support_data, "label_to_id")
            self.create_group(proto_support_data, "id_to_label")

            self.create_array("/", "raman_shift", np.array([], dtype=np.float32))
            self.create_array("/", "full_raman_shift", np.array([], dtype=np.float32))

            models = self.create_group("/", "models")
            self.set_attr(models, "batch_size", 16)
            self.set_attr(models, "train_ratio", 0.7)
            self.set_attr(models, "val_ratio", 0.2)
            self.set_attr(models, "test_ratio", 0.1)
            self.set_attr(models, "hidden_sizes", [32, 64])
            self.set_attr(models, "deep_hidden_sizes", [32, 64, 128, 256])
            self.set_attr(models, "n_proto_outputs", 32)
            # 新增：gMLP 专用参数配置（不影响其他模型）
            self.set_attr(models, "gmlp_config", {})
            # 新增：默认测试集准确率
            self.set_attr(models, "test_accuracy", 0.0)

        self.__file_name:str = abs_name
        self.__dirty:bool = False
        self.project_name_changed.emit(self.__file_name)

        return True

    def save(self)->None:
        self.__file.flush()
        self.__dirty:bool = False
        self.project_saved.emit()
    
    def save_as(self, file_name:str)->None:
        abs_file_name:str = os.path.abspath(file_name).replace("\\", "/")
        if self.__file_name == abs_file_name:
            return
        
        target_folder:str = os.path.dirname(abs_file_name)
        if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
            os.makedirs(target_folder)

        self.close()
        shutil.copy2(self.__file_name, abs_file_name)
        self.open(abs_file_name)
    
    @property
    def query_data(self)->tables.Group:
        return self.__file.root.query_data
    
    @property
    def train_data(self)->tables.Group:
        return self.__file.root.train_data
    
    @property
    def proto_support_data(self)->tables.Group:
        return self.__file.root.proto_support_data
    
    @property
    def proto_train_data(self)->tables.Group:
        return self.__file.root.proto_train_data
    
    @property
    def protos_data(self)->tables.Group:
        return self.__file.root.protos_data
    
    @property
    def query_data_info(self):
        query_data_info = []

        for class_node in self.query_classes:
            sheet_id:int = int(self.get_attr(class_node, "sheet_id"))
            class_id:int = int(self.get_attr(class_node, "class_id"))
            label:str = str(self.get_attr(class_node, "label"))
            confidence:float = float(self.get_attr(class_node, "confidence"))
            sheet_name:str = class_node._v_name
            rows_info = []
            query_data_info.append({
                "sheet_id": sheet_id,
                "class_node_path": class_node._v_pathname,
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
                "sheet_name": sheet_name,
                "rows_info": rows_info
            })

            for row_node in class_node:
                number:int = int(self.get_attr(row_node, "number"))
                class_id:int = int(self.get_attr(row_node, "class_id"))
                label:str = str(self.get_attr(row_node, "label"))
                confidence:float = float(self.get_attr(row_node, "confidence"))
                rows_info.append({
                    "class_node_path": class_node._v_pathname,
                    "class_id": class_id,
                    "sheet_id": sheet_id,
                    "label": label,
                    "confidence": confidence,
                    "sheet_name": sheet_name,
                    "row_node_path": row_node._v_pathname,
                    "md5": self.get_attr(row_node, "md5"),
                    "number": number
                })

        query_data_info.sort(key=lambda x: x["sheet_id"])
        for class_info in query_data_info:
            class_info["rows_info"].sort(key=lambda x: x["number"])

        return query_data_info

    @property
    def train_data_info(self):
        train_data_info = []

        for class_node in self.train_classes:
            class_id:int = int(self.get_attr(class_node, "class_id"))
            label:str = self.get_attr(class_node, "label")

            print("train label:", label, "id:", class_id)


            rows_info = []
            train_data_info.append({
                "class_node_path": class_node._v_pathname,
                "class_id": class_id,
                "label": label,
                "rows_info": rows_info
            })

            for row_node in class_node:
                number:int = int(self.get_attr(row_node, "number"))
                rows_info.append({
                    "class_node_path": class_node._v_pathname,
                    "class_id": class_id,
                    "label": label,
                    "row_node_path": row_node._v_pathname,
                    "md5": self.get_attr(row_node, "md5"),
                    "number": number
                })

        train_data_info.sort(key=lambda x: x["class_id"])
        for class_info in train_data_info:
            class_info["rows_info"].sort(key=lambda x: x["number"])

        return train_data_info
    
    @property
    def proto_train_data_info(self):
        proto_train_data_info = []

        for class_node in self.proto_train_classes:
            class_id:int = int(self.get_attr(class_node, "class_id"))
            label:str = self.get_attr(class_node, "label")

            print("query label:", label, "id:", class_id)


            rows_info = []
            proto_train_data_info.append({
                "class_node_path": class_node._v_pathname,
                "class_id": class_id,
                "label": label,
                "rows_info": rows_info
            })

            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                number:int = int(self.get_attr(row_node, "number"))
                rows_info.append({
                    "class_node_path": class_node._v_pathname,
                    "class_id": class_id,
                    "label": label,
                    "row_node_path": row_node._v_pathname,
                    "md5": self.get_attr(row_node, "md5"),
                    "number": number
                })

        proto_train_data_info.sort(key=lambda x: x["class_id"])
        for class_info in proto_train_data_info:
            class_info["rows_info"].sort(key=lambda x: x["number"])

        return proto_train_data_info

    @property
    def proto_support_data_info(self):
        proto_support_data_info = []

        for class_node in self.proto_support_classes:
            class_id:int = int(self.get_attr(class_node, "class_id"))
            label:str = self.get_attr(class_node, "label")
            rows_info = []
            proto_support_data_info.append({
                "class_node_path": class_node._v_pathname,
                "class_id": class_id,
                "label": label,
                "rows_info": rows_info
            })

            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                number:int = int(self.get_attr(row_node, "number"))
                rows_info.append({
                    "class_node_path": class_node._v_pathname,
                    "class_id": class_id,
                    "label": label,
                    "row_node_path": row_node._v_pathname,
                    "md5": self.get_attr(row_node, "md5"),
                    "number": number
                })

        proto_support_data_info.sort(key=lambda x: x["class_id"])
        for class_info in proto_support_data_info:
            class_info["rows_info"].sort(key=lambda x: x["number"])

        return proto_support_data_info

    def update_modify_time(self, node:tables.Node)->None:
        now = time.time()
        while node is not None:
            node._v_attrs.modify_time = now
            if node is self.__file.root:
                break

            node = node._v_parent

        self.__dirty:bool = True
        self.project_changed.emit()

    def remove_node(self, where:Union[tables.Node, str], name:Optional[str]=None)->None:
        try:
            active_node = self.__file.get_node(where, name)
            self.update_modify_time(active_node)
            self.__file.remove_node(where, name, recursive=True)
        except:
            pass

    def create_group(self, where:Union[str, tables.Group], name:str):
        group = self.__file.create_group(where, name)
        self.set_attr(group, "create_time", time.time())

        return group

    def create_array(self, where:Union[str, tables.Group], name:str, data:np.ndarray)->tables.Array:
        array = self.__file.create_array(where, name, data)
        self.set_attr(array, "create_time", time.time())
        
        return array
    
    def get_node(self, where, name=None)->tables.Node:
        return self.__file.get_node(where, name)

    # def get_attr(self, node:tables.Node, name:str)->Any:
    #     return getattr(node._v_attrs, name)

    def get_attr(self, group, name, default=None):
        """
        安全读取 PyTables 组属性。
        group: HDF5 group
        name: 属性名
        default: 若属性不存在则返回该值
        """
        try:
            return getattr(group._v_attrs, name)
        except Exception:
            return default
#

    def has_attr(self, node:tables.Node, name:str)->bool:
        return (name in node._v_attrs)
    
    def set_attr(self, node:tables.Node, name:str, value:Any)->None:
        if hasattr(node._v_attrs, name) and getattr(node._v_attrs, name) == value:
            return

        setattr(node._v_attrs, name, value)
        self.update_modify_time(node)
    
    def train_class(self, label:str)->tables.Group:
        return self.__file.get_node(self.train_classes, label)
    
    def proto_train_class(self, label:str)->tables.Group:
        return self.__file.get_node(self.proto_train_classes, label)
    
    def proto_support_class(self, label:str)->tables.Group:
        return self.__file.get_node(self.proto_support_classes, label)
    
    def query_class(self, sheet_name:str)->tables.Group:
        return self.__file.get_node(self.query_classes, sheet_name)
    
    def model(self, title:str)->tables.Group:
        return self.__file.get_node(self.models, title)

    def create_train_class(self, label:str)->tables.Group:
        class_id:int = self.train_classes._v_nchildren
        train_class_node:tables.Group = self.create_group(self.train_classes, label)
        self.set_attr(train_class_node, "class_id", class_id)
        self.set_attr(train_class_node, "label", label)
        self.set_attr(train_class_node, "first_available_row_number", 1)
        self.set_attr(self.label_to_id, label, class_id)
        self.set_attr(self.id_to_label, str(class_id), label)
        return train_class_node
    
    def create_proto_train_class(self, label:str)->tables.Group:
        class_id:int = self.proto_train_classes._v_nchildren
        proto_train_class_node:tables.Group = self.create_group(self.proto_train_classes, label)
        self.set_attr(proto_train_class_node, "class_id", class_id)
        self.set_attr(proto_train_class_node, "label", label)
        self.set_attr(proto_train_class_node, "first_available_row_number", 1)
        self.set_attr(self.proto_train_label_to_id, label, class_id)
        self.set_attr(self.proto_train_id_to_label, str(class_id), label)
        self.create_group(proto_train_class_node, "protos")

        return proto_train_class_node
    
    def create_proto_support_class(self, label:str)->tables.Group:
        class_id:int = self.proto_support_classes._v_nchildren
        proto_support_class_node:tables.Group = self.create_group(self.proto_support_classes, label)
        self.set_attr(proto_support_class_node, "class_id", class_id)
        self.set_attr(proto_support_class_node, "label", label)
        self.set_attr(proto_support_class_node, "first_available_row_number", 1)
        self.set_attr(self.proto_support_label_to_id, label, class_id)
        self.set_attr(self.proto_support_id_to_label, str(class_id), label)
        self.create_group(proto_support_class_node, "protos")

        return proto_support_class_node
    
    def create_query_class(self, sheet_name:str)->tables.Group:
        sheet_id:int = self.query_classes._v_nchildren
        sheet_node:tables.Group = self.create_group(self.query_classes, sheet_name)
        self.set_attr(sheet_node, "class_id", -1)
        self.set_attr(sheet_node, "label", "")
        self.set_attr(sheet_node, "confidence", 0.0)
        self.set_attr(sheet_node, "sheet_name", sheet_name)
        self.set_attr(sheet_node, "first_available_row_number", 1)
        self.set_attr(sheet_node, "sheet_id", sheet_id)
        return sheet_node

    def add_row(self, class_node:tables.Group, md5:str, data:Dict[str, np.ndarray])->tables.Group:
        row_number:str = self.get_attr(class_node, "first_available_row_number")
        row_group:tables.Group = self.create_group(class_node, md5)
        self.set_attr(row_group, "number", row_number)
        self.set_attr(row_group, "md5", md5)
        self.set_attr(row_group, "class_id", 0)
        self.set_attr(row_group, "label", "")
        self.set_attr(row_group, "confidence", 0.0)
        self.set_attr(class_node, "first_available_row_number", row_number + 1)
        for key in data:
            self.create_array(row_group, key, data[key])

        if str(class_node._v_parent._v_parent._v_name).startswith("proto"):
            self.create_group(row_group, "protos")

        return row_group

    def remove_train_class(self, class_node_path:str)->None:
        class_node:tables.Group = self.get_node(class_node_path)
        removed_class_id:int = self.get_attr(class_node, "class_id")
        self.clear_attrs(self.label_to_id)
        self.clear_attrs(self.id_to_label)
        self.remove_node(class_node)

        for class_node in self.train_classes:
            class_id:int = self.get_attr(class_node, "class_id")
            label:str = self.get_attr(class_node, "label")
            if class_id > removed_class_id:
                class_id -= 1
                self.set_attr(class_node, "class_id", class_id)

            self.set_attr(self.label_to_id, label, class_id)
            self.set_attr(self.id_to_label, str(class_id), label)

    def remove_proto_train_class(self, class_node_path:str)->None:
        class_node:tables.Group = self.get_node(class_node_path)
        removed_class_id:int = self.get_attr(class_node, "class_id")
        self.clear_attrs(self.proto_train_label_to_id)
        self.clear_attrs(self.proto_train_id_to_label)
        self.remove_node(class_node)

        for class_node in self.proto_train_classes:
            class_id:int = self.get_attr(class_node, "class_id")
            label:str = self.get_attr(class_node, "label")
            if class_id > removed_class_id:
                class_id -= 1
                self.set_attr(class_node, "class_id", class_id)

            self.set_attr(self.proto_train_label_to_id, label, class_id)
            self.set_attr(self.proto_train_id_to_label, str(class_id), label)

    def remove_query_class(self, class_node_path:str)->None:
        class_node:tables.Group = self.get_node(class_node_path)
        removed_sheet_id:int = self.get_attr(class_node, "sheet_id")
        self.remove_node(class_node)

        for class_node in self.train_classes:
            sheet_id:int = self.get_attr(class_node, "sheet_id")
            if sheet_id > removed_sheet_id:
                sheet_id -= 1
                self.set_attr(class_node, "sheet_id", sheet_id)

    @property
    def label_to_id(self)->tables.Group:
        return self.__file.root.train_data.label_to_id
    
    @property
    def id_to_label(self)->tables.Group:
        return self.__file.root.train_data.id_to_label
    
    @property
    def proto_train_label_to_id(self)->tables.Group:
        return self.__file.root.proto_train_data.label_to_id
    
    @property
    def proto_support_label_to_id(self)->tables.Group:
        return self.__file.root.proto_support_data.label_to_id
    
    @property
    def proto_train_id_to_label(self)->tables.Group:
        return self.__file.root.proto_train_data.id_to_label
    
    @property
    def proto_support_id_to_label(self)->tables.Group:
        return self.__file.root.proto_support_data.id_to_label
    
    @property
    def raman_shift(self)->Optional[np.ndarray]:
        if self.__raman_shift is None or self.__raman_shift.size == 0:
            return None
        
        return self.__raman_shift
    
    @raman_shift.setter
    def raman_shift(self, raman_shift:np.ndarray)->None:
        if self.__raman_shift is not None and self.__raman_shift.shape == raman_shift.shape and np.all(self.__raman_shift == raman_shift):
            return

        self.remove_node("/", "raman_shift")
        self.create_array("/", "raman_shift", raman_shift)
        self.__raman_shift = raman_shift

    @property
    def full_raman_shift(self)->Optional[np.ndarray]:
        if self.__full_raman_shift is None or self.__full_raman_shift.size == 0:
            return None
        
        return self.__full_raman_shift
    
    @full_raman_shift.setter
    def full_raman_shift(self, full_raman_shift:np.ndarray)->None:
        if self.__full_raman_shift is not None and self.__full_raman_shift.shape == full_raman_shift.shape and np.all(self.__full_raman_shift == full_raman_shift):
            return

        self.remove_node("/", "full_raman_shift")
        self.create_array("/", "full_raman_shift", full_raman_shift)
        self.__full_raman_shift = full_raman_shift

    @property
    def train_classes(self)->tables.Group:
        return self.__file.root.train_data.classes
    
    @property
    def proto_train_classes(self)->tables.Group:
        return self.__file.root.proto_train_data.classes
    
    @property
    def proto_support_classes(self)->tables.Group:
        return self.__file.root.proto_support_data.classes
    
    @property
    def query_classes(self)->tables.Group:
        return self.__file.root.query_data.classes
    
    @property
    def models(self)->tables.Group:
        return self.__file.root.models
    
    @property
    def batch_size(self)->int:
        return int(self.get_attr(self.models, "batch_size"))
    
    @batch_size.setter
    def batch_size(self, batch_size:int)->None:
        self.set_attr(self.models, "batch_size", batch_size)

    @property
    def train_ratio(self)->float:
        return self.get_attr(self.models, "train_ratio")
    
    @train_ratio.setter
    def train_ratio(self, train_ratio:float)->None:
        self.set_attr(self.models, "train_ratio", train_ratio)

    @property
    def val_ratio(self)->float:
        return self.get_attr(self.models, "val_ratio")
    
    @val_ratio.setter
    def val_ratio(self, val_ratio:float)->None:
        self.set_attr(self.models, "val_ratio", val_ratio)

    @property
    def test_ratio(self)->float:
        return self.get_attr(self.models, "test_ratio")
    
    @test_ratio.setter
    def test_ratio(self, test_ratio:float)->None:
        self.set_attr(self.models, "test_ratio", test_ratio)

    @property
    def n_features(self)->int:
        try:
            return int(self.__file.get_node("/train_data/pca", "components").shape[0])
        except:
            return 0
        
    @property
    def n_classes(self)->int:
        return len(self.id_to_label._v_attrs._f_list()) - 2
        
    @property
    def n_proto_features(self)->int:
        try:
            return int(self.__file.get_node("/proto_train_data/pca", "components").shape[0])
        except:
            return 0
        
    @property
    def n_proto_outputs(self)->int:
        return int(self.get_attr(self.models, "n_proto_outputs"))
    
    @n_proto_outputs.setter
    def n_proto_outputs(self, num:int):
        self.set_attr(self.models, "n_proto_outputs", num)

    # @property
    # def hidden_sizes(self)->List[int]:
    #     return self.get_attr(self.models, "hidden_sizes")

    @property
    def hidden_sizes(self) -> List:
        """
        hidden_sizes 可能是：
        - [32,64]
        - [512,256,256,{dropout1:0.3,...}]（gMLP）
        因此必须允许 object 类型，不要强制转换。
        """
        val = self.get_attr(self.models, "hidden_sizes")
        return val

    # @hidden_sizes.setter
    # def hidden_sizes(self, hidden_sizes:List[int])->None:
    #     self.set_attr(self.models, "hidden_sizes", hidden_sizes)

    @hidden_sizes.setter
    def hidden_sizes(self, hidden_sizes: List):
        """
        写入时允许 hidden_sizes 包含 dict，例如：
        [512,256,256, {"dropout1":0.3, ...}]
        存为 dtype=object，保持结构完全一致。
        """
        arr = np.empty(1, dtype=object)
        arr[0] = hidden_sizes
        self.set_attr(self.models, "hidden_sizes", arr[0])

    @property
    def deep_hidden_sizes(self)->List[int]:
        return self.get_attr(self.models, "deep_hidden_sizes")
    
    @deep_hidden_sizes.setter
    def deep_hidden_sizes(self, deep_hidden_sizes:List[int])->None:
        self.set_attr(self.models, "deep_hidden_sizes", deep_hidden_sizes)

    # --------------------------------------------------------
    # 添加 net 属性（当前选择的模型名称，如 "MLP" / "gMLP"）
    # GUI 在切换模型时会写入该字段
    # ----------------------------------------------------------
    @property
    def net(self):
        return self.get_attr(self.models, "net", default="MLP")

    @net.setter
    def net(self, value):
        self.set_attr(self.models, "net", value)

    # ----------------------------------------------------------
    # gMLP 配置（用于保存 dropout / kd / proto 参数）
    # ----------------------------------------------------------
    @property
    def gmlp_config(self):
        return self.get_attr(self.models, "gmlp_config", default=None)

    @gmlp_config.setter
    def gmlp_config(self, cfg):
        self.set_attr(self.models, "gmlp_config", cfg)

    #-----------------------------------------------------------------
    # test_accuracy: 保存测试集准确率
    # -------------------------------------------------------
    @property
    def test_accuracy(self):
        return float(self.get_attr(self.models, "test_accuracy", default=0.0))

    @test_accuracy.setter
    def test_accuracy(self, acc: float):
        self.set_attr(self.models, "test_accuracy", float(acc))
    # -------------------------------------------------------

    def load_model(self, model:BaseNet)->bool:
        title:str = model.title
        full_path = "/models/" + title
        if full_path not in self.__file:
            return False
        
        model_node = self.get_node(full_path)
        state_dict_node = self.get_node(full_path + "/state_dict")
        
        if hasattr(model, "n_inputs"):
            n_inputs:int = self.get_attr(model_node, "n_inputs")
            if model.n_inputs != n_inputs:
                return False
        
        if hasattr(model, "n_outputs"):
            n_outputs:int = self.get_attr(model_node, "n_outputs")
            if model.n_outputs != n_outputs:
                return False

        if hasattr(model, "hidden_sizes"):
            hidden_sizes = self.get_attr(model_node, "hidden_sizes")
            if model.hidden_sizes != hidden_sizes:
                return False
        
        state_dict = {}
        for child in state_dict_node:
            state_dict[child._v_name] = torch.tensor(child.read(), dtype=torch.float32)

        model.load_state_dict(state_dict, strict=False)
        return True
    
    def save_model(self, model:BaseNet, time_list:np.ndarray, train_loss_list:np.ndarray, train_acc_list:np.ndarray, val_loss_list:np.ndarray, val_acc_list:np.ndarray, best_epoch:int)->None:
        title:str = model.title
        full_path = "/models/" + title
        if full_path in self.__file:
            self.remove_node(full_path)

        model_node = self.create_group(self.models, title)
        state_dict_node = self.create_group(model_node, "state_dict")
        self.create_array(model_node, "time_list", time_list)
        self.create_array(model_node, "train_loss_list", train_loss_list)
        self.create_array(model_node, "train_acc_list", train_acc_list)
        self.create_array(model_node, "val_loss_list", val_loss_list)
        self.create_array(model_node, "val_acc_list", val_acc_list)

        self.set_attr(model_node, "best_epoch", best_epoch)
        self.set_attr(model_node, "best_acc", val_acc_list[best_epoch])

        if hasattr(model, "n_inputs"):
            self.set_attr(model_node, "n_inputs", model.n_inputs)

        if hasattr(model, "n_outputs"):
            self.set_attr(model_node, "n_outputs", model.n_outputs)

        if hasattr(model, "hidden_sizes"):
            self.set_attr(model_node, "hidden_sizes", model.hidden_sizes)

        state_dict = model.state_dict()
        for key in state_dict:
            self.create_array(state_dict_node, key, state_dict[key].cpu().numpy())

    def load_model_info(self, model_title:str)->Dict[str, Any]:
        model_node = self.get_node(self.models, model_title)
        model_info = {
            "time_list": model_node.time_list.read(),
            "train_loss_list": model_node.train_loss_list.read(),
            "train_acc_list": model_node.train_acc_list.read(),
            "val_loss_list": model_node.val_loss_list.read(),
            "val_acc_list": model_node.val_acc_list.read(),
            "best_epoch": int(model_node._v_attrs.best_epoch),
            "best_acc": float(model_node._v_attrs.best_acc),

        }
        
        return model_info

    def clear_attrs(self, node:tables.Node):
        attr_names = node._v_attrs._f_list()
        for attr_name in attr_names:
            if attr_name not in ["modify_time", "create_time"]:
                del node._v_attrs[attr_name]

        self.update_modify_time(node)

    def clear_train_classes(self)->None:
        self.remove_node("/train_data", "classes")
        self.create_group("/train_data", "classes")
        self.clear_attrs(self.label_to_id)
        self.clear_attrs(self.id_to_label)
        self.remove_node("/train_data", "pca")
        self.clear_query_pca()

    def clear_proto_train_classes(self)->None:
        self.remove_node("/proto_train_data", "classes")
        self.create_group("/proto_train_data", "classes")
        self.clear_attrs(self.proto_train_label_to_id)
        self.clear_attrs(self.proto_train_id_to_label)
        self.remove_node("/proto_train_data", "pca")
        self.clear_query_proto_pca()
        self.clear_proto_support_pca()

    def clear_proto_support_classes(self)->None:
        self.remove_node("/proto_support_data", "classes")
        self.create_group("/proto_support_data", "classes")
        self.clear_attrs(self.proto_support_label_to_id)
        self.clear_attrs(self.proto_support_id_to_label)
        self.remove_node("/proto_support_data", "pca")

    def clear_query_classes(self)->None:
        self.remove_node("/query_data", "classes")
        self.create_group("/query_data", "classes")

    @property
    def normalized_train_data(self)->np.ndarray:
        data_list:List[np.ndarray] = []
        for class_node in self.train_classes:
            for row_node in class_node:
                data_list.append(row_node.normalized_data.read())

        return np.array(data_list)
    
    @property
    def normalized_proto_train_data(self)->np.ndarray:
        data_list:List[np.ndarray] = []
        for class_node in self.proto_train_classes:
            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                data_list.append(row_node.normalized_data.read())

        return np.array(data_list)
    
    @property
    def normalized_proto_support_data(self)->np.ndarray:
        data_list:List[np.ndarray] = []
        for class_node in self.proto_support_classes:
            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                data_list.append(row_node.normalized_data.read())

        return np.array(data_list)
    
    @property
    def normalized_query_data(self)->np.ndarray:
        data_list:List[np.ndarray] = []
        for sheet_node in self.query_classes:
            for row_node in sheet_node:
                data_list.append(row_node.normalized_data.read())

        return np.array(data_list)

    def save_pca(self, pca:PCA, pca_features:np.ndarray, pca_reconstructed_data:np.ndarray)->None:
        if "/train_data/pca" in self.__file:
            self.remove_node("/train_data", "pca")

        pca_node = self.create_group("/train_data", "pca")
        self.set_attr(pca_node, "n_components", pca._n_components)
        self.create_array(pca_node, "components", pca._components)
        self.create_array(pca_node, "mean", pca._mean)
        self.create_array(pca_node, "explained_variance", pca._explained_variance)
        
        updated_pca = []

        i:int = 0
        for class_node in self.train_classes:
            for row_node in class_node:
                if "pca_features" in row_node:
                    self.remove_node(row_node, "pca_features")
                if "pca_reconstructed_data" in row_node:
                    self.remove_node(row_node, "pca_reconstructed_data")

                self.create_array(row_node, "pca_features", pca_features[i, :])
                self.create_array(row_node, "pca_reconstructed_data", pca_reconstructed_data[i, :])
                updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), pca_reconstructed_data[i, :]))
                i += 1

        if updated_pca:
            self.train_pca_reconstructed_data_changed.emit(updated_pca)

    def save_proto_pca(self, pca:PCA, pca_features:np.ndarray, pca_reconstructed_data:np.ndarray)->None:
        if "/proto_train_data/pca" in self.__file:
            self.remove_node("/proto_train_data", "pca")

        pca_node = self.create_group("/proto_train_data", "pca")
        self.set_attr(pca_node, "n_components", pca._n_components)
        self.create_array(pca_node, "components", pca._components)
        self.create_array(pca_node, "mean", pca._mean)
        self.create_array(pca_node, "explained_variance", pca._explained_variance)
        
        updated_pca = []

        i:int = 0
        for class_node in self.proto_train_classes:
            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                if "proto_pca_features" in row_node:
                    self.remove_node(row_node, "proto_pca_features")
                if "proto_pca_reconstructed_data" in row_node:
                    self.remove_node(row_node, "proto_pca_reconstructed_data")

                self.create_array(row_node, "proto_pca_features", pca_features[i, :])
                self.create_array(row_node, "proto_pca_reconstructed_data", pca_reconstructed_data[i, :])
                updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), pca_reconstructed_data[i, :]))
                i += 1

        if updated_pca:
            self.proto_train_pca_reconstructed_data_changed.emit(updated_pca)

    def save_proto_support_pca(self, pca_features:np.ndarray, pca_reconstructed_data:np.ndarray)->None:
        updated_pca = []

        i:int = 0
        for class_node in self.proto_support_classes:
            for row_node in class_node:
                if "pca_features" in row_node:
                    self.remove_node(row_node, "pca_features")
                if "pca_reconstructed_data" in row_node:
                    self.remove_node(row_node, "pca_reconstructed_data")

                self.create_array(row_node, "pca_features", pca_features[i, :])
                self.create_array(row_node, "pca_reconstructed_data", pca_reconstructed_data[i, :])
                updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), pca_reconstructed_data[i, :]))
                i += 1

        if updated_pca:
            self.proto_support_pca_reconstructed_data_changed.emit(updated_pca)

    def save_query_pca(self, pca_features:np.ndarray, pca_reconstructed_data:np.ndarray)->None:
        updated_pca = []

        i:int = 0
        for class_node in self.query_classes:
            for row_node in class_node:
                if "pca_features" in row_node:
                    self.remove_node(row_node, "pca_features")
                if "pca_reconstructed_data" in row_node:
                    self.remove_node(row_node, "pca_reconstructed_data")

                self.create_array(row_node, "pca_features", pca_features[i, :])
                self.create_array(row_node, "pca_reconstructed_data", pca_reconstructed_data[i, :])
                updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), pca_reconstructed_data[i, :]))
                i += 1

        if updated_pca:
            self.query_pca_reconstructed_data_changed.emit(updated_pca)

    def save_query_proto_pca(self, pca_features:np.ndarray, pca_reconstructed_data:np.ndarray)->None:
        updated_pca = []

        i:int = 0
        for class_node in self.query_classes:
            for row_node in class_node:
                if "proto_pca_features" in row_node:
                    self.remove_node(row_node, "proto_pca_features")
                if "proto_pca_reconstructed_data" in row_node:
                    self.remove_node(row_node, "proto_pca_reconstructed_data")

                self.create_array(row_node, "proto_pca_features", pca_features[i, :])
                self.create_array(row_node, "proto_pca_reconstructed_data", pca_reconstructed_data[i, :])
                updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), pca_reconstructed_data[i, :]))
                i += 1

        if updated_pca:
            self.query_proto_pca_reconstructed_data_changed.emit(updated_pca)

    def clear_query_pca(self):
        updated_pca = []
        for class_node in self.query_classes:
            for row_node in class_node:
                if "pca_features" in row_node:
                    self.remove_node(row_node, "pca_features")
                if "pca_reconstructed_data" in row_node:
                    old_pca_shape = row_node["pca_reconstructed_data"].shape
                    self.remove_node(row_node, "pca_reconstructed_data")
                    updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), np.full(old_pca_shape, np.nan)))

        if updated_pca:
            self.query_pca_reconstructed_data_changed.emit(updated_pca)

    def clear_query_proto_pca(self):
        updated_pca = []
        for class_node in self.query_classes:
            for row_node in class_node:
                if "proto_pca_features" in row_node:
                    self.remove_node(row_node, "proto_pca_features")
                if "proto_pca_reconstructed_data" in row_node:
                    old_pca_shape = row_node["proto_pca_reconstructed_data"].shape
                    self.remove_node(row_node, "proto_pca_reconstructed_data")
                    updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), np.full(old_pca_shape, np.nan)))

        if updated_pca:
            self.query_proto_pca_reconstructed_data_changed.emit(updated_pca)

    def clear_proto_support_pca(self):
        updated_pca = []
        for class_node in self.proto_support_classes:
            for row_node in class_node:
                if "proto_pca_features" in row_node:
                    self.remove_node(row_node, "proto_pca_features")
                if "proto_pca_reconstructed_data" in row_node:
                    old_pca_shape = row_node["proto_pca_reconstructed_data"].shape
                    self.remove_node(row_node, "proto_pca_reconstructed_data")
                    updated_pca.append((class_node._v_name + "/" + str(self.get_attr(row_node, "number")), np.full(old_pca_shape, np.nan)))

        if updated_pca:
            self.proto_support_pca_reconstructed_data_changed.emit(updated_pca)

    def load_pca(self)->Optional[PCA]:
        try:
            pca_node = self.train_data.pca
            pca = PCA(self.get_attr(pca_node, "n_components"))
            pca._components = pca_node.components.read()
            pca._mean = pca_node.mean.read()
            pca._explained_variance = pca_node.explained_variance.read()
            return pca
        except:
            return None
        
    def load_proto_pca(self)->Optional[PCA]:
        try:
            pca_node = self.proto_train_data.pca
            pca = PCA(self.get_attr(pca_node, "n_components"))
            pca._components = pca_node.components.read()
            pca._mean = pca_node.mean.read()
            pca._explained_variance = pca_node.explained_variance.read()
            return pca
        except:
            return None