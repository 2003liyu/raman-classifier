import uuid
import os
from typing import Dict, Any, List

import numpy as np

from PySide6.QtCore import Signal, QObject, QThread

from .DataManager import DataManager
from .NormalClassifier import NormalClassifier
from .ProtoClassifier import ProtoClassifier
from .BaseClassifier import BaseClassifier
from .ProjectFile import ProjectFile
from ..common.utils import cat
from .models import gMLP, MLP, KAN, CNN, ResNet, ResNetV2, ResNeXt, ResNeXtV2, DRSN_CS, DRSN_CW

from .ReverseVerifier import PWMVerifier
from .DeepLiftVerifier import DeepLiftVerifier
import torch
import pandas as pd


class API(QObject):
    progress_stage_changed = Signal(str)
    progress_changed = Signal(float)

    add_data_finished = Signal(dict)
    add_data_error = Signal(str)

    train_pca_reconstructed_data_changed = Signal(list)
    proto_train_pca_reconstructed_data_changed = Signal(list)
    proto_support_pca_reconstructed_data_changed = Signal(list)
    query_pca_reconstructed_data_changed = Signal(list)
    query_proto_pca_reconstructed_data_changed = Signal(list)

    train_curve_reset = Signal(int)
    train_curve_changed = Signal(float, float, float, float, int)
    train_finished = Signal()
    train_error = Signal(str)
    ask_to_continue_train = Signal(float)
    ask_to_save_model = Signal(float)
    ask_to_train = Signal()
    # [新增] PWM 验证完成信号：传递权重数组和类别名称
    reverse_validation_finished = Signal(np.ndarray, str)

    classify_finished = Signal(list)
    classify_error = Signal(str)

    remove_finished = Signal()
    remove_error = Signal(str)

    clear_finished = Signal()
    clear_error = Signal(str)

    _add_train_data = Signal(list)
    _add_proto_train_data = Signal(list)
    _add_proto_support_data = Signal(list)
    _add_query_data = Signal(list)
    _train = Signal(int, bool)
    _classify = Signal(np.ndarray, list)
    _remove_train_class = Signal(str)
    _remove_train_row = Signal(str)
    _remove_proto_train_class = Signal(str)
    _remove_proto_train_row = Signal(str)
    _remove_proto_support_class = Signal(str)
    _remove_proto_support_row = Signal(str)
    _remove_query_class = Signal(str)
    _remove_query_row = Signal(str)
    _clear_train_classes = Signal()
    _clear_proto_train_classes = Signal()
    _clear_proto_support_classes = Signal()
    _clear_query_classes = Signal()

    def __init__(self):
        QObject.__init__(self)

        self.__self_folder: str = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        active_project_txt: str = self.__self_folder + "/../data/active_project.txt"
        active_project: str = ""
        if os.path.isfile(active_project_txt):
            active_project: str = os.path.abspath(cat(active_project_txt)).replace("\\", "/")

        self.__temp_folder: str = os.path.abspath(self.__self_folder + "/../data/temp").replace("\\", "/")
        if os.path.isdir(self.__temp_folder):
            for temp_file in os.listdir(self.__temp_folder):
                full_file_name: str = os.path.abspath(self.__temp_folder + "/" + temp_file).replace("\\", "/")
                if active_project != full_file_name:
                    try:
                        os.remove(full_file_name)
                    except:
                        pass

        if os.path.isfile(active_project):
            self.__project_file: ProjectFile = ProjectFile(active_project)
        else:
            self.__project_file: ProjectFile = ProjectFile(self.__temp_folder + f"/{uuid.uuid4()}.h5")

        self.__project_file.train_pca_reconstructed_data_changed.connect(self.train_pca_reconstructed_data_changed.emit)
        self.__project_file.proto_train_pca_reconstructed_data_changed.connect(
            self.proto_train_pca_reconstructed_data_changed.emit)
        self.__project_file.proto_support_pca_reconstructed_data_changed.connect(
            self.proto_support_pca_reconstructed_data_changed.emit)
        self.__project_file.query_pca_reconstructed_data_changed.connect(self.query_pca_reconstructed_data_changed.emit)
        self.__project_file.query_proto_pca_reconstructed_data_changed.connect(
            self.query_proto_pca_reconstructed_data_changed.emit)

        self.data_manager: DataManager = DataManager(self.__project_file)
        self.data_manager.progress_changed.connect(self.progress_changed.emit)
        self.data_manager.progress_stage_changed.connect(self.progress_stage_changed.emit)
        self.data_manager.add_data_finished.connect(self.add_data_finished.emit)
        self.data_manager.add_data_error.connect(self.add_data_error.emit)
        self.data_manager.remove_error.connect(self.remove_error.emit)
        self.data_manager.remove_finished.connect(self.remove_finished.emit)
        self.data_manager.clear_error.connect(self.clear_error.emit)
        self.data_manager.clear_finished.connect(self.clear_finished.emit)

        self._add_train_data.connect(self.data_manager.add_train_data)
        self._add_query_data.connect(self.data_manager.add_query_data)
        self._add_proto_support_data.connect(self.data_manager.add_proto_support_data)
        self._add_proto_train_data.connect(self.data_manager.add_proto_train_data)
        self._remove_train_class.connect(self.data_manager.remove_train_class)
        self._remove_train_row.connect(self.data_manager.remove_train_row)
        self._remove_proto_train_class.connect(self.data_manager.remove_proto_train_class)
        self._remove_proto_train_row.connect(self.data_manager.remove_proto_train_row)
        self._remove_proto_support_class.connect(self.data_manager.remove_proto_support_class)
        self._remove_proto_support_row.connect(self.data_manager.remove_proto_support_row)
        self._remove_query_class.connect(self.data_manager.remove_query_class)
        self._remove_query_row.connect(self.data_manager.remove_query_row)
        self._clear_train_classes.connect(self.data_manager.clear_train_classes)
        self._clear_query_classes.connect(self.data_manager.clear_query_classes)
        self._clear_proto_train_classes.connect(self.data_manager.clear_proto_train_classes)
        self._clear_proto_support_classes.connect(self.data_manager.clear_proto_support_classes)

        self.__active_net: str = "MLP"
        self.__classifiers: Dict[str, BaseClassifier] = {}
        self.__update_async_thread()

    def __update_async_thread(self):
        self.__async_thread = QThread()
        self.__async_thread.setObjectName("Async Operation Thread")
        self.__async_thread.start()

        self.data_manager.moveToThread(self.__async_thread)
        for classifier in self.__classifiers.values():
            classifier.moveToThread(self.__async_thread)

    @property
    def progress_stage(self) -> str:
        return self.data_manager.progress_stage

    @property
    def progress(self) -> float:
        return self.data_manager.progress

    @property
    def is_temp(self) -> bool:
        current_folder: str = os.path.dirname(self.__project_file.file_name)
        return (current_folder == self.__temp_folder)

    @property
    def is_dirty(self) -> bool:
        return self.project_file.is_dirty

    def open(self, project_file_name: str) -> None:
        if not self.__project_file.open(project_file_name):
            return

        self.data_manager.update_dataloader()
        self.data_manager.update_deep_dataloader()
        self.data_manager.update_proto_dataloader()
        self.data_manager.update_deep_proto_dataloader()

    def clear_train_classes(self) -> None:
        self._clear_train_classes.emit()

    def clear_query_classes(self) -> None:
        self._clear_query_classes.emit()

    def clear_proto_train_classes(self) -> None:
        self._clear_proto_train_classes.emit()

    def clear_proto_support_classes(self) -> None:
        self.project_file.clear_proto_support_classes()

    def remove_train_row(self, row_node_path: str) -> None:
        self._remove_train_row.emit(row_node_path)

    def remove_proto_train_row(self, row_node_path: str) -> None:
        self._remove_proto_train_row.emit(row_node_path)

    def remove_query_row(self, row_node_path: str) -> None:
        self._remove_query_row.emit(row_node_path)

    def remove_train_class(self, class_node_path: str) -> None:
        self._remove_train_class.emit(class_node_path)

    def remove_proto_train_class(self, class_node_path: str) -> None:
        self._remove_proto_train_class.emit(class_node_path)

    def remove_query_class(self, class_node_path: str) -> None:
        self._remove_query_class.emit(class_node_path)

    def remove_proto_support_class(self, class_node_path: str) -> None:
        self._remove_proto_support_class.emit(class_node_path)

    def remove_proto_support_row(self, row_node_path: str) -> None:
        self._remove_proto_support_row.emit(row_node_path)

    def open_new(self) -> None:
        self.close()
        self.__update_async_thread()
        self.__project_file.open(self.__temp_folder + f"/{uuid.uuid4()}.h5")

    def close(self) -> None:
        self.data_manager.move_to_main_thread()
        for classifier in self.__classifiers.values():
            classifier.move_to_main_thread()

        if self.__async_thread.isRunning():
            self.__async_thread.quit()
            self.__async_thread.wait()

        self.__project_file.close()
        self.__mlp_classifier = None
        self.__proto_mlp_classifier = None
        self.data_manager.clear()

    @property
    def n_features(self) -> int:
        return self.__project_file.n_features

    @property
    def active_net(self) -> str:
        return self.__active_net

    # @active_net.setter
    # def active_net(self, net:str)->None:
    #     self.__active_net = net

    # 给gMLP加超参
    @active_net.setter
    def active_net(self, net: str) -> None:
        self.__active_net = net

        if net in ["gMLP", "ProtoGMLP"]:
            self.__project_file.gmlp_config = {
                "hidden_sizes": [512, 256, 256],
                "dropout1": 0.3,
                "dropout2": 0.2,
                "proto_lr_ratio": 0.5,
                "kd_alpha": 0.7,
                "kd_temperature": 4
            }
            print(">>> gMLP 参数已写入 ProjectFile.gmlp_config")

    @property
    def n_classes(self) -> int:
        return self.data_manager.n_classes

    @property
    def hidden_sizes(self) -> List[int]:
        return self.__project_file.hidden_sizes

    @property
    def n_parameters(self) -> int:
        if not self.data_manager:
            raise RuntimeError("no data is added")

        return self.classifier.n_parameters

    @hidden_sizes.setter
    def hidden_sizes(self, hidden_sizes: List[int]):
        if self.hidden_sizes == hidden_sizes:
            return

        self.__project_file.hidden_sizes = hidden_sizes

        if self.__mlp_classifier is not None:
            self.__mlp_classifier.hidden_sizes = hidden_sizes

        if self.__proto_mlp_classifier is not None:
            self.__proto_mlp_classifier.hidden_sizes = hidden_sizes

    def get_classifier(self, is_deep: bool, is_proto: bool, model_class: type, base_name: str):
        if base_name not in self.__classifiers:
            if not self.data_manager:
                raise RuntimeError("no data is added")

            if is_proto:
                classifier = ProtoClassifier(self.data_manager, is_deep, model_class, base_name)
            else:
                classifier = NormalClassifier(self.data_manager, is_deep, model_class, base_name)

            classifier.moveToThread(self.__async_thread)
            classifier.train_curve_reset.connect(self.train_curve_reset.emit)
            classifier.train_curve_changed.connect(self.train_curve_changed.emit)
            classifier.train_finished.connect(self.train_finished.emit)
            classifier.train_error.connect(self.train_error.emit)
            classifier.ask_to_continue_train.connect(self.ask_to_continue_train.emit)
            classifier.ask_to_save_model.connect(self.ask_to_save_model.emit)
            classifier.ask_to_train.connect(self.ask_to_train.emit)
            classifier.classify_error.connect(self.classify_error.emit)
            classifier.classify_finished.connect(self.classify_finished.emit)
            self.__classifiers[base_name] = classifier

        return self.__classifiers[base_name]

    #
    @property
    def gmlp_classifier(self) -> NormalClassifier:
        return self.get_classifier(False, False, gMLP, "gMLP")

    @property
    def proto_gmlp_classifier(self) -> ProtoClassifier:
        return self.get_classifier(False, True, gMLP, "ProtoGMLP")

    #

    @property
    def mlp_classifier(self) -> NormalClassifier:
        return self.get_classifier(False, False, MLP, "MLP")

    @property
    def proto_mlp_classifier(self) -> ProtoClassifier:
        return self.get_classifier(False, True, MLP, "ProtoMLP")

    @property
    def kan_classifier(self) -> NormalClassifier:
        return self.get_classifier(False, False, KAN, "KAN")

    @property
    def proto_kan_classifier(self) -> ProtoClassifier:
        return self.get_classifier(False, True, KAN, "ProtoKAN")

    @property
    def cnn_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, CNN, "CNN")

    @property
    def proto_cnn_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, CNN, "ProtoCNN")

    @property
    def resnet_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, ResNet, "ResNet")

    @property
    def proto_resnet_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, ResNet, "ProtoResNet")

    @property
    def resnetv2_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, ResNetV2, "ResNetV2")

    @property
    def proto_resnetv2_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, ResNetV2, "ProtoResNetV2")

    @property
    def resnext_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, ResNeXt, "ResNeXt")

    @property
    def proto_resnext_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, ResNeXt, "ProtoResNeXt")

    @property
    def resnextv2_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, ResNeXtV2, "ResNeXtV2")

    @property
    def proto_resnextv2_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, ResNeXtV2, "ProtoResNeXtV2")

    @property
    def drsn_cs_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, DRSN_CS, "DRSN_CS")

    @property
    def proto_drsn_cs_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, DRSN_CS, "ProtoDRSN_CS")

    @property
    def drsn_cw_classifier(self) -> NormalClassifier:
        return self.get_classifier(True, False, DRSN_CW, "DRSN_CW")

    @property
    def proto_drsn_cw_classifier(self) -> ProtoClassifier:
        return self.get_classifier(True, True, DRSN_CW, "ProtoDRSN_CW")

    def add_cmd(self, epochs: int):
        self.classifier.cmd_queue.put(epochs)

    def stop_train(self):
        self.classifier.training = False

    @property
    def trainning(self) -> bool:
        return self.classifier.training

    @property
    def project_file(self) -> ProjectFile:
        return self.__project_file

    @property
    def classifier(self) -> BaseClassifier:
        if not self.data_manager:
            raise RuntimeError("no data is added")

        if self.active_net == 'MLP':
            return self.mlp_classifier
        elif self.active_net == 'ProtoMLP':
            return self.proto_mlp_classifier
        #
        elif self.active_net == "gMLP":
            return self.gmlp_classifier
        elif self.active_net == "ProtoGMLP":
            return self.proto_gmlp_classifier
        #
        elif self.active_net == 'KAN':
            return self.kan_classifier
        elif self.active_net == 'ProtoKAN':
            return self.proto_kan_classifier
        elif self.active_net == 'CNN':
            return self.cnn_classifier
        elif self.active_net == 'ProtoCNN':
            return self.proto_cnn_classifier
        elif self.active_net == "ResNet":
            return self.resnet_classifier
        elif self.active_net == "ProtoResNet":
            return self.proto_resnet_classifier
        elif self.active_net == "ResNetV2":
            return self.resnetv2_classifier
        elif self.active_net == "ProtoResNetV2":
            return self.proto_resnetv2_classifier
        elif self.active_net == "ResNeXt":
            return self.resnext_classifier
        elif self.active_net == "ProtoResNeXt":
            return self.proto_resnext_classifier
        elif self.active_net == "ResNeXtV2":
            return self.resnextv2_classifier
        elif self.active_net == "ProtoResNeXtV2":
            return self.proto_resnextv2_classifier
        elif self.active_net == "DRSN_CS":
            return self.drsn_cs_classifier
        elif self.active_net == "ProtoDRSN_CS":
            return self.proto_drsn_cs_classifier
        elif self.active_net == "DRSN_CW":
            return self.drsn_cw_classifier
        elif self.active_net == "ProtoDRSN_CW":
            return self.proto_drsn_cw_classifier

    @property
    def is_trained(self) -> bool:
        return self.classifier.is_trained

    @property
    def is_proto(self) -> bool:
        return self.active_net.lower().startswith("proto")

    @property
    def is_deep(self) -> bool:
        return self.active_net in [
            "CNN", "ProtoCNN",
            "ResNet", "ProtoResNet",
            "ResNetV2", "ProtoResNetV2",
            "ResNeXt", "ProtoResNeXt",
            "ResNeXtV2", "ProtoResNeXtV2",
            "DRSN_CS", "ProtoDRSN_CS",
            "DRSN_CW", "ProtoDRSN_CW",
        ]

    def add_train_data(self, excel_file_names: str) -> None:
        if isinstance(excel_file_names, str):
            excel_file_names = [excel_file_names]

        self._add_train_data.emit(excel_file_names)

    def add_proto_train_data(self, excel_file_names: str) -> None:
        if isinstance(excel_file_names, str):
            excel_file_names = [excel_file_names]

        self._add_proto_train_data.emit(excel_file_names)

    def add_proto_support_data(self, excel_file_names: str) -> None:
        if isinstance(excel_file_names, str):
            excel_file_names = [excel_file_names]

        self._add_proto_support_data.emit(excel_file_names)

    def add_query_data(self, excel_file_names: str) -> None:
        if isinstance(excel_file_names, str):
            excel_file_names = [excel_file_names]

        self._add_query_data.emit(excel_file_names)

    def classify(self, data: np.ndarray, groups_info: Dict[str, Dict[str, Any]]) -> None:
        if not self.data_manager:
            raise RuntimeError("no data is added")

        self._classify.connect(self.classifier.classify)
        self._classify.emit(data, groups_info)
        self._classify.disconnect(self.classifier.classify)


    def run_reverse_validation(self, class_id: int) -> None:

        if not self.classifier or not self.classifier.model:
            print("错误: 模型未加载或未训练")
            return


        data_np, labels_np = self.data_manager.get_raw_test_data()

        print("DeepLIFT输入shape:", data_np.shape)


        from .DeepLiftVerifier import DeepLiftVerifier
        verifier = DeepLiftVerifier(self.classifier)


        label_name = self.data_manager.label_of(class_id)
        raman_shift = self.data_manager.project_file.raman_shift


        print(f">>> 开始计算类别 [{label_name}] 的 DeepLIFT 归因权重...")
        weights = verifier.generate_report(data_np, labels_np, class_id, raman_shift)


        if weights is not None:
            self.reverse_validation_finished.emit(weights, label_name)



    def train(self, epochs: int = 100, force_retrain: bool = False) -> bool:
        if not self.data_manager:
            raise RuntimeError("no data is added")

        self._train.connect(self.classifier.train)
        self._train.emit(epochs, force_retrain)
        self._train.disconnect(self.classifier.train)

    def label_of(self, class_id: int) -> str:
        return self.data_manager.label_of(class_id)

    def class_id_of(self, label: str) -> int:
        return self.data_manager.class_id_of(label)






