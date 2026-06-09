from __future__ import annotations
import traceback
from queue import Queue
from typing import Optional, Dict, Any, List

import numpy as np
import tables

from PySide6.QtCore import QObject, Signal, QCoreApplication, Qt

import torch
from torch import nn, optim
import torch.nn.functional as F

from .DataManager import DataManager
from .ProjectFile import ProjectFile
from .Chronoscope import Chronoscope
from .models import BaseNet


class BaseClassifier(QObject):
    train_curve_changed = Signal(float, float, float, float, int)
    train_curve_reset = Signal(int)
    train_finished = Signal()
    train_error = Signal(str)
    ask_to_continue_train = Signal(float)
    ask_to_save_model = Signal(float)
    ask_to_train = Signal()
    classify_error = Signal(str)
    classify_finished = Signal(list)
    model_info_changed = Signal(int, int)
    _request_move_to_main_thread = Signal()

    def __init__(self, data_manager: DataManager, is_deep: bool, model_class: type, base_name: str):
        QObject.__init__(self)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[BaseNet] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.data_manager: DataManager = data_manager
        self.criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.cmd_queue: Queue = Queue()
        self.training: bool = False
        self.chronoscope: Chronoscope = Chronoscope()

        self.__is_deep: bool = is_deep
        self.__base_name: str = base_name
        self.__model_class: type = model_class

        self._request_move_to_main_thread.connect(
            self.on_move_to_main_thread,
            Qt.ConnectionType.BlockingQueuedConnection
        )

    @property
    def is_deep(self) -> bool:
        return self.__is_deep

    @property
    def base_name(self) -> str:
        return self.__base_name

    @property
    def project_file(self) -> ProjectFile:
        return self.data_manager.project_file

    @property
    def model_title(self) -> str:
        return (self.model.title if self.model else "")

    def aggregate_outputs(self, outputs: torch.Tensor, group_rows_counts: List[int]):
        _, predicted = torch.max(outputs, 1)

        test_acc = float(self.project_file.test_accuracy)
        confidences = np.full((outputs.shape[0],), test_acc)
        #--------------------------------------------------------------------

        classify_result: List[Dict[str, Any]] = []
        last_index: int = 0
        for group_rows_count in group_rows_counts:
            result = {}
            for i in range(last_index, last_index + group_rows_count):
                class_id = int(predicted[i])
                if class_id not in result:
                    result[class_id] = confidences[i] / group_rows_count
                else:
                    result[class_id] += confidences[i] / group_rows_count

            best_class_id = -1
            best_confidence = 0
            for class_id, confidence in result.items():
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_class_id = class_id

            classify_result.append({
                "best_class_id": best_class_id,
                "best_confidence": best_confidence,
                "class_ids": predicted[last_index:last_index + group_rows_count].cpu().numpy(),
                "confidences": confidences[last_index:last_index + group_rows_count],
                "net_outputs": outputs[last_index:last_index + group_rows_count, :].cpu().numpy()
            })

            last_index += group_rows_count

        return classify_result

    def update_optimizer(self):
        if self.model:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0003, weight_decay=1e-4)
        else:
            self.optimizer = None

    def _train(self, epochs: int = 100, force_retrain: bool = False) -> float:
        pass


    def _classify(self, data: np.ndarray, group_rows_counts: List[int]) -> None:

        if self.model is None:
            self.create_model()

        self.model.eval()


        x = torch.tensor(data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)


        classify_result = self.aggregate_outputs(outputs, group_rows_counts)


        self.classify_finished.emit(classify_result)
#-----------------------------------------------------------------------------
    def train(self, epochs: int = 100, force_retrain: bool = False) -> float:
        try:
            return self._train(epochs, force_retrain)
        except BaseException as e:
            print(traceback.format_exc())
            self.train_error.emit(str(e))
            return 0

    def classify(self, data: np.ndarray, group_rows_counts: List[int]) -> None:
        try:
            self._classify(data, group_rows_counts)
        except BaseException as e:
            print(traceback.format_exc())
            self.classify_error.emit(str(e))

    @property
    def is_trained(self) -> bool:
        is_proto: bool = self.__class__.__name__.lower().startswith("proto")
        train_classes: tables.Group = self.project_file.train_classes
        if is_proto:
            train_classes = self.project_file.proto_train_classes

        is_new_model: bool = False
        if not self.model and self.n_inputs > 0 and self.n_outputs > 0:
            self.create_model()
            self.model_info_changed.emit(self.n_parameters, self.flops)
            is_new_model = True

        model_title: str = self.model.title if self.model else ""
        if model_title not in self.project_file.models:
            return False

        is_trained = (
                self.project_file.models[model_title].state_dict._v_attrs.modify_time
                >= train_classes._v_attrs.modify_time
        )
        if not is_trained:
            return False

        if is_new_model:
            self.project_file.load_model(self.model)

        return True

    @property
    def n_parameters(self) -> Optional[int]:
        self.is_trained
        if self.model is None:
            return None
        return self.model.n_parameters

    @property
    def flops(self) -> int:
        self.is_trained
        if self.model is None:
            return None

        if hasattr(self.model, "n_inputs"):
            return self.model.flops(torch.rand((self.model.n_inputs,), dtype=torch.float32))
        else:
            return self.model.flops(self.data_manager.project_file.raman_shift)

    @property
    def n_outputs(self) -> int:
        pass

    @property
    def n_inputs(self) -> int:
        pass

    @property
    def hidden_sizes(self) -> List[int]:
        return (self.project_file.hidden_sizes if not self.is_deep else self.project_file.deep_hidden_sizes)

    @hidden_sizes.setter
    def hidden_sizes(self, hidden_sizes: List[int]) -> None:
        if self.hidden_sizes == hidden_sizes:
            return

        if self.is_deep:
            self.project_file.deep_hidden_sizes = hidden_sizes
        else:
            self.project_file.hidden_sizes = hidden_sizes


    def create_model(self):
        if self.n_inputs > 0 and self.n_outputs > 0:

            # 基础参数，必须包含 base_name 以修复 KeyError
            model_kwargs = {
                "base_name": self.base_name,
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
            }



            if self.base_name in ["gMLP", "ProtoGMLP"]:

                config = self.project_file.gmlp_config

                default_config = {
                    "hidden_sizes": [512, 256, 256],
                    "dropout1": 0.3,
                    "dropout2": 0.2,
                    "proto_lr_ratio": 0.5,
                    "kd_alpha": 0.7,
                    "kd_temperature": 4
                }

                if config is None:

                    config = default_config
                elif isinstance(config, list) or isinstance(config, tuple) or isinstance(config, np.ndarray):

                    print(f"Warning: gmlp_config is {type(config)}, resetting to default dict with extracted sizes.")


                    extracted_sizes = []
                    for x in config:
                        if isinstance(x, (int, float, np.number)):
                            extracted_sizes.append(int(x))


                    if not extracted_sizes:
                        extracted_sizes = default_config["hidden_sizes"]


                    config = default_config.copy()
                    config["hidden_sizes"] = extracted_sizes


                    self.project_file.gmlp_config = config


                hs_list = list(config.get("hidden_sizes", [512, 256, 256]))


                extra_cfg = {k: v for k, v in config.items() if k != "hidden_sizes"}


                combined_hidden_sizes = hs_list + [extra_cfg]

                model_kwargs["hidden_sizes"] = combined_hidden_sizes

            elif self.is_deep:

                model_kwargs["hidden_sizes"] = self.project_file.deep_hidden_sizes

            else:

                raw_sizes = self.project_file.hidden_sizes
                clean_sizes = []

                if isinstance(raw_sizes, (list, tuple, np.ndarray)):
                    for x in raw_sizes:
                        if isinstance(x, (int, float, np.number)):
                            clean_sizes.append(int(x))

                model_kwargs["hidden_sizes"] = clean_sizes


            self.model = self.__model_class(**model_kwargs)
            self.model.to(self.device)

        self.update_optimizer()
    def move_to_main_thread(self):
        self._request_move_to_main_thread.emit()

    def on_move_to_main_thread(self):
        self.moveToThread(QCoreApplication.instance().thread())


    def save_detailed_results(self, y_true, y_pred, y_probs, model_name):
        import pandas as pd
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import datetime
        import os


        n_classes = self.data_manager.n_classes


        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

        class_stats = {}
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            class_stats[i] = {
                "TPR (Recall)": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 7),
                "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 7)
            }


        pr_data_dir = os.path.join(os.getcwd(), f"{model_name}_KD_Aug_three_PR_Data11")
        if not os.path.exists(pr_data_dir):
            os.makedirs(pr_data_dir)


        bacteria_labels = [self.data_manager.label_of(i) for i in range(n_classes)]

        print(f">>> 正在导出各菌种的 PR 原始数据至: {pr_data_dir}")

        for class_idx, label_name in enumerate(bacteria_labels):
            pr_rows = []
            for i in range(len(y_true)):

                sample_id = i
                is_positive = 1 if int(y_true[i]) == class_idx else 0

                score = round(float(y_probs[i][class_idx]), 7)
                pr_rows.append([sample_id, is_positive, score])


            pr_file_path = os.path.join(pr_data_dir, f"py_{model_name}_{label_name}.csv")


            df_pr = pd.DataFrame(pr_rows)
            df_pr.to_csv(pr_file_path, index=False, header=False, float_format='%.7f')


        rows = []
        for idx in range(len(y_true)):
            t_idx = int(y_true[idx])
            p_idx = int(y_pred[idx])

            prob_row = y_probs[idx]
            sorted_indices = np.argsort(prob_row)[::-1]

            top1_idx = sorted_indices[0]
            top2_idx = sorted_indices[1]

            top1_prob = prob_row[top1_idx]
            top2_prob = prob_row[top2_idx]
            true_class_prob = prob_row[t_idx]

            rows.append({
                "样本编号": idx + 1,
                "真实菌种": self.data_manager.label_of(t_idx),
                "预测菌种": self.data_manager.label_of(p_idx),
                "是否正确": "√" if t_idx == p_idx else "×",
                "预测置信度(Top1)": round(float(top1_prob), 7),
                "正确菌种概率": round(float(true_class_prob), 7),
                "第二名概率": round(float(top2_prob), 7),
                "置信度差距": round(float(top1_prob - top2_prob), 7),
                "第二名菌种": self.data_manager.label_of(top2_idx),
                "所属类别整体_TPR": class_stats[t_idx]["TPR (Recall)"],
                "所属类别整体_FPR": class_stats[t_idx]["FPR"]
            })


        df_excel = pd.DataFrame(rows)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


        excel_filename = f"{model_name}_KD_Aug_three_{timestamp}.xlsx"
        df_excel.to_excel(excel_filename, index=False)



        print(f">>> 测试完成！")
        print(f">>> 1. 可视化 PR 数据目录: {pr_data_dir}")
        print(f">>> 2. 详细 Excel 报告: {os.path.join(os.getcwd(), excel_filename)}")

        return rows