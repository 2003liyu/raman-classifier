import traceback
from typing import List, Dict, Optional, Union, Tuple, Any, Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import white_tophat, median_filter
from pybaselines import Baseline
import tables

from PySide6.QtCore import QObject, Signal, QThread, QCoreApplication, Qt
import torch

from torch.utils.data import Dataset, DataLoader, random_split


from ..common.ReadOnlyExcel import ReadOnlyExcel
from ..common.utils import md5sum
from .RamanDataset import RamanDataset
from .ProtoRamanDataset import ProtoRamanDataset
from .ProjectFile import ProjectFile
from .PCA import PCA
from .GaussianFitTool import GaussianFitTool
from .ProgressStage import ProgressStage
from .RamanSubDataset import RamanSubDataset


class DataManager(QObject):

    progress_stage_changed = Signal(str)
    progress_changed = Signal(float)

    add_data_finished = Signal(dict)
    add_data_error = Signal(str)

    remove_error = Signal(str)
    remove_finished = Signal()

    clear_error = Signal(str)
    clear_finished = Signal()

    _request_move_to_main_thread = Signal()

    def __init__(self, project_file:ProjectFile):
        QObject.__init__(self)

        self.__progress_stage:str = ""
        self.__progress:float = 0.0

        self.__project_file:ProjectFile = project_file

        self.raw_data = []

        self.__full_dataset:Optional[RamanDataset] = None
        self.__train_dataset:Optional[Dataset] = None
        self.__val_dataset:Optional[Dataset] = None

        self.__proto_full_base_dataset:Optional[RamanDataset] = None
        self.__proto_train_base_dataset:Optional[RamanDataset] = None
        self.__proto_val_base_dataset:Optional[RamanDataset] = None

        self.__proto_train_dataset:Optional[ProtoRamanDataset] = None
        self.__proto_val_dataset:Optional[ProtoRamanDataset] = None

        self.__train_loader:Optional[DataLoader] = None
        self.__val_loader:Optional[DataLoader] = None
        # 新增：单独的测试集（使用 Query 数据作为测试数据）-----------
        self.__test_dataset = None
        self.__test_loader = None
        #-----------------------------------------------------

        self.__proto_train_loader:Optional[DataLoader] = None
        self.__proto_val_loader:Optional[DataLoader] = None

        self.__deep_full_dataset:Optional[RamanDataset] = None
        self.__deep_train_dataset:Optional[Dataset] = None
        self.__deep_val_dataset:Optional[Dataset] = None

        self.__deep_proto_full_base_dataset:Optional[RamanDataset] = None
        self.__deep_proto_train_base_dataset:Optional[RamanDataset] = None
        self.__deep_proto_val_base_dataset:Optional[RamanDataset] = None

        self.__deep_proto_train_dataset:Optional[ProtoRamanDataset] = None
        self.__deep_proto_val_dataset:Optional[ProtoRamanDataset] = None

        self.__deep_train_loader:Optional[DataLoader] = None
        self.__deep_val_loader:Optional[DataLoader] = None

        self.__deep_proto_train_loader:Optional[DataLoader] = None
        self.__deep_proto_val_loader:Optional[DataLoader] = None

        self._request_move_to_main_thread.connect(self.on_move_to_main_thread, Qt.ConnectionType.BlockingQueuedConnection)


    @property
    def progress(self)->float:
        return self.__progress

    @progress.setter
    def progress(self, progress:float)->None:
        if self.__progress == progress:
            return

        self.__progress:float = progress
        self.progress_changed.emit(progress)

    @property
    def progress_stage(self)->str:
        return self.__progress_stage

    @progress_stage.setter
    def progress_stage(self, state:str)->None:
        if self.__progress_stage == state:
            return

        self.__progress_stage:str = state
        self.progress_stage_changed.emit(state)

    @property
    def project_file(self)->ProjectFile:
        return self.__project_file

    @property
    def n_classes(self)->int:
        return self.__project_file.n_classes

    @property
    def n_proto_features(self)->int:
        return self.__project_file.n_proto_features

    @property
    def n_proto_outputs(self)->int:
        return self.__project_file.n_proto_outputs

    def clear(self)->None:
        self.__full_dataset:Optional[RamanDataset] = None
        self.__train_dataset:Optional[Dataset] = None
        self.__val_dataset:Optional[Dataset] = None

        self.__proto_full_base_dataset:Optional[RamanDataset] = None
        self.__proto_train_base_dataset:Optional[RamanDataset] = None
        self.__proto_val_base_dataset:Optional[RamanDataset] = None

        self.__proto_train_dataset:Optional[ProtoRamanDataset] = None
        self.__proto_val_dataset:Optional[ProtoRamanDataset] = None

        self.__train_loader:Optional[DataLoader] = None
        self.__val_loader:Optional[DataLoader] = None

        self.__proto_train_loader:Optional[DataLoader] = None
        self.__proto_val_loader:Optional[DataLoader] = None

        self.__deep_full_dataset:Optional[RamanDataset] = None
        self.__deep_train_dataset:Optional[Dataset] = None
        self.__deep_val_dataset:Optional[Dataset] = None

        self.__deep_proto_full_base_dataset:Optional[RamanDataset] = None
        self.__deep_proto_train_base_dataset:Optional[RamanDataset] = None
        self.__deep_proto_val_base_dataset:Optional[RamanDataset] = None

        self.__deep_proto_train_dataset:Optional[ProtoRamanDataset] = None
        self.__deep_proto_val_dataset:Optional[ProtoRamanDataset] = None

        self.__deep_train_loader:Optional[DataLoader] = None
        self.__deep_val_loader:Optional[DataLoader] = None

        self.__deep_proto_train_loader:Optional[DataLoader] = None
        self.__deep_proto_val_loader:Optional[DataLoader] = None

    def update_train_pca(self)->None:
        normalized_data:np.ndarray = self.__project_file.normalized_train_data
        pca = PCA(0.95)
        pca_features:np.ndarray = pca.fit_transform(normalized_data)
        pca_reconstructed_data:np.ndarray = pca.inverse_transform(pca_features)
        self.__project_file.save_pca(pca, pca_features, pca_reconstructed_data)
        self.update_query_pca(pca)

    def update_proto_train_pca(self)->None:
        normalized_data:np.ndarray = self.__project_file.normalized_proto_train_data
        pca = PCA(0.95)
        pca_features:np.ndarray = pca.fit_transform(normalized_data)
        pca_reconstructed_data:np.ndarray = pca.inverse_transform(pca_features)
        self.__project_file.save_proto_pca(pca, pca_features, pca_reconstructed_data)
        self.update_proto_support_pca(pca)
        self.update_query_proto_pca(pca)

    def update_proto_support_pca(self, pca:Optional[PCA]=None)->None:
        if pca is None:
            pca:Optional[PCA] = self.__project_file.load_proto_pca()

        if pca is None:
            self.__project_file.clear_proto_support_pca()
        else:
            normalized_data:np.ndarray = self.__project_file.normalized_proto_support_data
            if normalized_data.shape[0] == 0:
                return

            pca_features:np.ndarray = pca.transform(normalized_data)
            pca_reconstructed_data:np.ndarray = pca.inverse_transform(pca_features)
            self.__project_file.save_proto_support_pca(pca_features, pca_reconstructed_data)

    def update_query_pca(self, pca:Optional[PCA]=None)->None:
        if pca is None:
            pca:Optional[PCA] = self.__project_file.load_pca()

        if pca is None:
            self.__project_file.clear_query_pca()
        else:
            normalized_data:np.ndarray = self.__project_file.normalized_query_data
            if normalized_data.shape[0] == 0:
                return

            pca_features:np.ndarray = pca.transform(normalized_data)
            pca_reconstructed_data:np.ndarray = pca.inverse_transform(pca_features)
            self.__project_file.save_query_pca(pca_features, pca_reconstructed_data)

    def update_query_proto_pca(self, pca:Optional[PCA]=None)->None:
        if pca is None:
            pca:Optional[PCA] = self.__project_file.load_proto_pca()

        if pca is None:
            self.__project_file.clear_query_proto_pca()
        else:
            normalized_data:np.ndarray = self.__project_file.normalized_query_data
            if normalized_data.shape[0] == 0:
                return

            pca_features:np.ndarray = pca.transform(normalized_data)
            pca_reconstructed_data:np.ndarray = pca.inverse_transform(pca_features)
            self.__project_file.save_query_proto_pca(pca_features, pca_reconstructed_data)

    def update_dataloader(self, update_full_dataset:bool=True):
        if self.__full_dataset is None:
            self.__full_dataset = RamanDataset(self.__project_file, False, False)
        elif update_full_dataset:
            self.__full_dataset.update()

        total_size = len(self.__full_dataset)
        train_size = int(self.__project_file.train_ratio * total_size)
        val_size = total_size - train_size

        if total_size == 0:
            self.__full_dataset = None
            self.__train_dataset = None
            self.__val_dataset = None
            self.__train_loader = None
            self.__val_loader = None
            return

        print("重新切分数据集")
        self.__train_dataset, self.__val_dataset = random_split(self.__full_dataset, [train_size, val_size])
        self.__train_loader:Optional[DataLoader] = DataLoader(self.__train_dataset, batch_size=self.__project_file.batch_size, shuffle=True)
        self.__val_loader:Optional[DataLoader] = DataLoader(self.__val_dataset, batch_size=self.__project_file.batch_size, shuffle=False)


    def update_deep_dataloader(self, update_full_dataset: bool = True):


        if not hasattr(self, "raw_data"):
            self.raw_data = []

        all_labels = [d['label'] for d in self.raw_data if 'label' in d]

        if all_labels:
            self.__project_file.categories = sorted(list(set(all_labels)))


        if self.__deep_full_dataset is None:
            self.__deep_full_dataset = RamanDataset(self.__project_file, True, False)
        elif update_full_dataset:

            self.__deep_full_dataset.update()

        total_size = len(self.__deep_full_dataset)
        train_size = int(self.__project_file.train_ratio * total_size)
        val_size = total_size - train_size

        if total_size == 0:
            self.__deep_full_dataset = None
            self.__deep_train_dataset = None
            self.__deep_val_dataset = None
            self.__deep_train_loader = None
            self.__deep_val_loader = None
            return

        print("重新切分深度学习数据集 (CNN/ResNeXt)")

        self.__deep_train_dataset, self.__deep_val_dataset = random_split(
            self.__deep_full_dataset, [train_size, val_size]
        )


        self.__deep_train_loader = DataLoader(
            self.__deep_train_dataset,
            batch_size=self.__project_file.batch_size,
            shuffle=True
        )
        self.__deep_val_loader = DataLoader(
            self.__deep_val_dataset,
            batch_size=self.__project_file.batch_size,
            shuffle=False
        )


    def update_test_dataloader(self):

        normalized_query = self.__project_file.normalized_query_data

        if normalized_query is None or normalized_query.shape[0] == 0:
            self.__test_dataset = None
            self.__test_loader = None
            return


        self.__test_dataset = RamanDataset(self.__project_file, False, False, is_query=True)

        self.__test_loader = DataLoader(
            self.__test_dataset,
            batch_size=self.__project_file.batch_size,
            shuffle=False
        )


    def update_test_dataloader(self):
        pf = self._DataManager__project_file
        normalized_query = pf.normalized_query_data

        if normalized_query is None or normalized_query.shape[0] == 0:
            self._DataManager__test_dataset = None
            self._DataManager__test_loader = None
            return


        valid_label_map = {}
        try:
            raw_map = pf.label_to_id
            if isinstance(raw_map, dict) and len(raw_map) > 0:
                valid_label_map = raw_map
        except:
            pass

        if not valid_label_map:
            try:
                for class_node in pf.train_classes:
                    valid_label_map[class_node._v_name.strip()] = int(class_node._v_attrs.class_id)
            except:
                pass

        print(">>> Test label map:", valid_label_map)

        if not valid_label_map:
            self._DataManager__test_loader = None
            return


        labels = []
        for class_node in pf.query_classes:
            raw_name = class_node._v_name.strip()
            target_name = raw_name[:-1] + "T" if raw_name.upper().endswith("X") else raw_name
            true_id = -1
            for k, v in valid_label_map.items():
                if str(k).strip().upper() == target_name.upper():
                    true_id = int(v)
                    break
            for _ in class_node:
                labels.append(true_id)


        try:

            n_features_raw = normalized_query.shape[1]
            n_features_expected = pf.n_features

            # 如果模型输入维度小于原始光谱，说明使用了 PCA（MLP/gMLP）
            if n_features_expected < n_features_raw:

                print(f">>> 检测到 PCA 模型: {n_features_raw} -> {n_features_expected}")

                pca_tool = pf.load_pca()

                if pca_tool is None:
                    raise RuntimeError("PCA 模型未找到")

                processed_features = pca_tool.transform(normalized_query)

                print(f">>> PCA 降维完成: {processed_features.shape}")

            else:

                processed_features = normalized_query

                print(f">>> CNN/ResNeXt 使用原始特征: {processed_features.shape}")


            labels_array = np.array(labels, dtype=np.int64)
            features_tensor = torch.tensor(processed_features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # 关键修正点

            valid_mask = labels_tensor != -1
            valid_count = torch.sum(valid_mask).item()

            from torch.utils.data import TensorDataset
            self._DataManager__test_dataset = TensorDataset(features_tensor[valid_mask], labels_tensor[valid_mask])
            self._DataManager__test_loader = DataLoader(
                self._DataManager__test_dataset,
                batch_size=pf.batch_size,
                shuffle=False
            )
            print(f">>> 测试集预处理及加载完成，样本数: {valid_count}")
        except Exception as e:
            print(f"预处理或构建 Loader 失败: {e}")
    #----------------------------------------------------------------

    def get_raw_test_data(self) -> Tuple[np.ndarray, np.ndarray]:

        pf = self.__project_file

        raw_query_data = pf.normalized_query_data
        if raw_query_data is None or len(raw_query_data) == 0:
            return np.array([]), np.array([])


        class_to_id = {}
        for i, node in enumerate(pf.train_classes):
            class_to_id[node._v_name.strip()] = i


        all_labels = []
        for class_node in pf.query_classes:
            raw_name = class_node._v_name.strip()

            search_name = raw_name[:-1] if (raw_name.endswith('X') or raw_name.endswith('T')) else raw_name


            target_id = -1
            for name, cid in class_to_id.items():
                if name.startswith(search_name):
                    target_id = cid
                    break

            for _ in class_node:
                all_labels.append(target_id)

        return raw_query_data, np.array(all_labels)



    @staticmethod
    def split_dataset_by_class(full_dataset: RamanDataset, split_sizes:List[int])->List[RamanSubDataset]:
        datasets = []
        sum_size:int = 0
        for size in split_sizes:
            current_classes = full_dataset.class_ids[sum_size:sum_size+size]
            current_dataset = RamanSubDataset(full_dataset, current_classes)
            datasets.append(current_dataset)
            sum_size += size

        return datasets

    def update_proto_dataloader(self, update_full_dataset:bool=True):
        if self.__proto_full_base_dataset is None:
            self.__proto_full_base_dataset = RamanDataset(self.__project_file, False, True)
        elif update_full_dataset:
            self.__proto_full_base_dataset.update()

        total_size = len(self.__proto_full_base_dataset.class_ids)
        train_size = int(self.__project_file.train_ratio * total_size)
        val_size = total_size - train_size

        if total_size == 0:
            self.__proto_full_base_dataset = None
            self.__proto_train_base_dataset = None
            self.__proto_val_base_dataset = None
            self.__proto_train_dataset = None
            self.__proto_val_dataset = None
            self.__proto_train_loader = None
            self.__proto_val_loader = None
            return

        (
            self.__proto_train_base_dataset,
            self.__proto_val_base_dataset
        ) = self.split_dataset_by_class(
            self.__proto_full_base_dataset,
            [train_size, val_size]
        )
        self.__proto_train_dataset = ProtoRamanDataset(self.__proto_train_base_dataset)
        self.__proto_val_dataset = ProtoRamanDataset(self.__proto_val_base_dataset)

        print("重新切分数据集")
        self.__proto_train_loader:Optional[DataLoader] = DataLoader(self.__proto_train_dataset, batch_size=1, shuffle=True)
        self.__proto_val_loader:Optional[DataLoader] = DataLoader(self.__proto_val_dataset, batch_size=1, shuffle=False)

    def update_deep_proto_dataloader(self, update_full_dataset:bool=True):
        if self.__deep_proto_full_base_dataset is None:
            self.__deep_proto_full_base_dataset = RamanDataset(self.__project_file, True, True)
        elif update_full_dataset:
            self.__deep_proto_full_base_dataset.update()

        total_size = len(self.__deep_proto_full_base_dataset.class_ids)
        train_size = int(self.__project_file.train_ratio * total_size)
        val_size = total_size - train_size

        if total_size == 0:
            self.__deep_proto_full_base_dataset = None
            self.__deep_proto_train_base_dataset = None
            self.__deep_proto_val_base_dataset = None
            self.__deep_proto_train_dataset = None
            self.__deep_proto_val_dataset = None
            self.__deep_proto_train_loader = None
            self.__deep_proto_val_loader = None
            return

        (
            self.__deep_proto_train_base_dataset,
            self.__deep_proto_val_base_dataset
        ) = self.split_dataset_by_class(
            self.__deep_proto_full_base_dataset,
            [train_size, val_size]
        )
        self.__deep_proto_train_dataset = ProtoRamanDataset(self.__deep_proto_train_base_dataset)
        self.__deep_proto_val_dataset = ProtoRamanDataset(self.__deep_proto_val_base_dataset)

        print("重新切分数据集")
        self.__deep_proto_train_loader:Optional[DataLoader] = DataLoader(self.__deep_proto_train_dataset, batch_size=1, shuffle=True)
        self.__deep_proto_val_loader:Optional[DataLoader] = DataLoader(self.__deep_proto_val_dataset, batch_size=1, shuffle=False)

    @property
    def train_loader(self)->DataLoader:
        if self.__train_loader is None:
            self.update_dataloader()

        return self.__train_loader

    @property
    def val_loader(self)->DataLoader:
        if self.__val_loader is None:
            self.update_dataloader()

        return self.__val_loader

    #增加测试集-----------------------------------
    @property
    def test_loader(self):
        if self.__test_loader is None:
            self.update_test_dataloader()
        return self.__test_loader
    #----------------------------------------
    @property
    def deep_train_loader(self)->DataLoader:
        if self.__deep_train_loader is None:
            self.update_deep_dataloader()

        return self.__deep_train_loader

    @property
    def deep_val_loader(self)->DataLoader:
        if self.__deep_val_loader is None:
            self.update_deep_dataloader()

        return self.__deep_val_loader

    @property
    def proto_train_dataset(self)->ProtoRamanDataset:
        if self.__proto_train_dataset is None:
            self.update_proto_dataloader()

        return self.__proto_train_dataset

    @property
    def proto_val_dataset(self)->ProtoRamanDataset:
        if self.__proto_val_dataset is None:
            self.update_proto_dataloader()

        return self.__proto_val_dataset

    @property
    def proto_train_loader(self)->DataLoader:
        if self.__proto_train_loader is None:
            self.update_proto_dataloader()

        return self.__proto_train_loader

    @property
    def proto_val_loader(self)->DataLoader:
        if self.__proto_val_loader is None:
            self.update_proto_dataloader()

        return self.__proto_val_loader

    @property
    def deep_proto_train_dataset(self)->ProtoRamanDataset:
        if self.__deep_proto_train_dataset is None:
            self.update_deep_proto_dataloader()

        return self.__deep_proto_train_dataset

    @property
    def deep_proto_val_dataset(self)->ProtoRamanDataset:
        if self.__deep_proto_val_dataset is None:
            self.update_deep_proto_dataloader()

        return self.__deep_proto_val_dataset

    @property
    def deep_proto_train_loader(self)->DataLoader:
        if self.__deep_proto_train_loader is None:
            self.update_deep_proto_dataloader()

        return self.__deep_proto_train_loader

    @property
    def deep_proto_val_loader(self)->DataLoader:
        if self.__deep_proto_val_loader is None:
            self.update_deep_proto_dataloader()

        return self.__deep_proto_val_loader

    def stage(self, name:str, total_progress:float)->ProgressStage:
        return ProgressStage(name, total_progress, self)

    def gaussian_fit(self, data_list:List[Dict[str, np.ndarray]], normalized_data:np.ndarray):
        x = self.__project_file.raman_shift / 1000
        with self.stage("高斯拟合", 0.5) as stage:
            results = GaussianFitTool.fit(x, normalized_data, stage)
            i:int = 0
            for data in data_list:
                params:np.ndarray = results[i][0]
                n_peaks:int = (params.shape[0] - 1) // 3
                data["peaks_amp"] = params[:n_peaks]
                data["peaks_mu"] = params[n_peaks:2*n_peaks]
                data["peaks_sigma"] = params[2*n_peaks:3*n_peaks]
                data["peaks_offset"] = params[3*n_peaks]
                data["gaussian_fitted_data"] = results[i][1]
                i += 1

    def add_data(self, excel_file_names:Union[List[str], str], group_key:str)->Tuple[Dict[str, Any], np.ndarray]:
        if isinstance(excel_file_names, str):
            excel_file_names = [excel_file_names]

        cell_counts = []
        total_cell_count = 0
        excel_files:List[ReadOnlyExcel] = []
        total_cols:int = 0
        for excel_file_name in excel_file_names:
            excel:ReadOnlyExcel = ReadOnlyExcel(excel_file_name)
            excel_files.append(excel)
            for sheet_name in excel.sheet_names:
                excel.sheet(sheet_name)
                cell_count = excel.rows * excel.cols
                cell_counts.append(cell_count)
                total_cell_count += cell_count
                total_cols += (excel.cols - 1)

        clip_raw_data_list:List[np.ndarray] = []
        data_list:List[Dict[str, np.ndarray]] = []

        added_classes:List[str] = []
        updated_classes:List[str] = []
        ignored_data:Dict[Tuple[str, str], List[int]] = {}

        classes_node:tables.Group = getattr(self.__project_file, group_key + "_classes")
        get_class_func:Callable[[str], tables.Group] = getattr(self.__project_file, group_key + "_class")
        create_class_func:Callable[[str], tables.Group] = getattr(self.__project_file, "create_" + group_key + "_class")

        k:int = 0
        imported_basic_cols:int = 0
        for excel in excel_files:
            for sheet in range(len(excel.sheet_names)):
                sheet_name:str = excel.sheet_names[sheet]

                with self.stage(f"读取 {excel.base_name} 中的 {sheet_name}", 0.1*cell_counts[k]/total_cell_count) as stage:
                    excel.sheet(sheet)
                    raw_data:np.ndarray = excel.full_data(stage)
                    k += 1

                if self.__project_file.raman_shift is None:
                    x = raw_data[:, 0]
                    mask = ((x >= 500) & (x <= 2800))
                    self.__project_file.full_raman_shift = x
                    self.__project_file.raman_shift = x[mask]

                ignored_rows:List[int] = []
                has_class:bool = (sheet_name in classes_node)
                if has_class:
                    class_node:tables.Group = get_class_func(sheet_name)
                else:
                    class_node:tables.Group = create_class_func(sheet_name)

                full_raw_data:np.ndarray = self.interpolate(raw_data, True)
                clip_raw_data:np.ndarray = self.interpolate(raw_data, False)
                for i in range(full_raw_data.shape[0]):
                    full_row:np.ndarray = full_raw_data[i, :]
                    clip_row:np.ndarray = clip_raw_data[i, :]
                    md5:str = md5sum(full_row)
                    if md5 in class_node:
                        print("ignore", sheet_name, "colum", i + 1)
                        ignored_rows.append(i)
                        imported_basic_cols += 1
                        continue

                    row_node:tables.Group = self.__project_file.add_row(class_node, md5, {"full_raw_data": full_row, "clip_raw_data": clip_row})
                    data_list.append({
                        "row_node": row_node,
                        "full_raw_data": full_row,
                        "clip_raw_data": clip_row,
                        "md5": md5
                    })
                    clip_raw_data_list.append(clip_row)
                    imported_basic_cols += 1

                if len(ignored_rows) == full_raw_data.shape[0]:
                    ignored_data[excel.base_name, sheet_name] = []
                elif len(ignored_rows) > 0:
                    ignored_data[excel.base_name, sheet_name] = ignored_rows
                elif has_class:
                    updated_classes.append(sheet_name)
                else:
                    added_classes.append(sheet_name)

        result:Dict[str, Any] = {
            "added_classes": added_classes,
            "updated_classes": updated_classes,
            "ignored_data": ignored_data
        }

        if not data_list:
            return result, None, []

        with self.stage("合并原始数据", 0.02) as stage:
            clip_raw_data:np.ndarray = np.array(clip_raw_data_list)

        with self.stage("去除宇宙射线", 0.01) as stage:
            cosmic_rays_removed_data:np.ndarray = self.remove_cosmic_rays(clip_raw_data)

        with self.stage("校正基线", 0.02) as stage:
            baseline_removed_data, baseline_data = self.remove_baseline(cosmic_rays_removed_data)

        with self.stage("滤波", 0.02) as stage:
            filtered_data:np.ndarray = self.filter(baseline_removed_data)

        with self.stage("归一化", 0.01) as stage:
            normalized_data:np.ndarray = self.normalize(filtered_data)

        self.gaussian_fit(data_list, normalized_data)

        with self.stage("写入数据", 0.18) as stage:
            row_nodes:List[tables.Group] = []
            i:int = 0
            for data in data_list:
                row_node = data["row_node"]
                row_nodes.append(row_node)
                self.__project_file.create_array(row_node, "cosmic_rays_removed_data", cosmic_rays_removed_data[i, :])
                self.__project_file.create_array(row_node, "baseline_removed_data", baseline_removed_data[i, :])
                self.__project_file.create_array(row_node, "baseline_data", baseline_data[i, :])
                self.__project_file.create_array(row_node, "filtered_data", filtered_data[i, :])
                self.__project_file.create_array(row_node, "normalized_data", normalized_data[i, :])
                self.__project_file.create_array(row_node, "peaks_amp", data["peaks_amp"])
                self.__project_file.create_array(row_node, "peaks_mu", data["peaks_mu"])
                self.__project_file.create_array(row_node, "peaks_sigma", data["peaks_sigma"])
                self.__project_file.create_array(row_node, "peaks_offset", data["peaks_offset"])
                self.__project_file.create_array(row_node, "gaussian_fitted_data", data["gaussian_fitted_data"])
                i += 1
                stage.progress = i / total_cols

        return result, normalized_data, row_nodes

    def add_train_data(self, excel_file_names:Union[List[str], str])->None:
        self.progress_stage:str = ""
        self.progress:float = 0.0

        try:
            result, normalized_data, row_nodes = self.add_data(excel_file_names, "train")

            if normalized_data is not None:
                with self.stage("主成分分析", 0.02) as stage:
                    self.update_train_pca()
                    self.update_dataloader()
                    self.update_deep_dataloader()

            self.add_data_finished.emit(result)
        except BaseException as e:
            print(traceback.format_exc())
            self.add_data_error.emit(str(e))

    def add_proto_support_data(self, excel_file_names:Union[List[str], str])->None:
        self.progress_stage:str = ""
        self.progress:float = 0.0

        try:
            result, normalized_data, row_nodes = self.add_data(excel_file_names, "proto_support")

            if normalized_data is not None:
                with self.stage("主成分分析", 0.02) as stage:
                    pca:Optional[PCA] = self.__project_file.load_proto_pca()
                    if pca is None:
                        self.add_data_finished.emit(result)
                        return

                    pca_features = pca.transform(normalized_data)
                    pca_reconstructed_data = pca.inverse_transform(pca_features)

                    i:int = 0
                    for row_node in row_nodes:
                        self.__project_file.create_array(row_node, "pca_features", pca_features[i, :])
                        self.__project_file.create_array(row_node, "pca_reconstructed_data", pca_reconstructed_data[i, :])
                        i += 1

            self.add_data_finished.emit(result)
        except BaseException as e:
            print(traceback.format_exc())
            self.add_data_error.emit(str(e))

    def add_proto_train_data(self, excel_file_names:Union[List[str], str])->None:
        self.progress_stage:str = ""
        self.progress:float = 0.0

        try:
            result, normalized_data, row_nodes = self.add_data(excel_file_names, "proto_train")

            if normalized_data is not None:
                with self.stage("主成分分析", 0.02) as stage:
                    self.update_proto_train_pca()
                    self.update_proto_dataloader()
                    self.update_deep_proto_dataloader()

            self.add_data_finished.emit(result)
        except BaseException as e:
            print(traceback.format_exc())
            self.add_data_error.emit(str(e))

    def add_query_data(self, excel_file_names:Union[List[str], str])->None:
        self.progress_stage:str = ""
        self.progress:float = 0.0

        try:
            result, normalized_data, row_nodes = self.add_data(excel_file_names, "query")

            if normalized_data is not None:
                with self.stage("主成分分析", 0.02) as stage:
                    pca:Optional[PCA] = self.__project_file.load_pca()
                    if pca is not None:
                        pca_features = pca.transform(normalized_data)
                        pca_reconstructed_data = pca.inverse_transform(pca_features)

                        i:int = 0
                        for row_node in row_nodes:
                            self.__project_file.create_array(row_node, "pca_features", pca_features[i, :])
                            self.__project_file.create_array(row_node, "pca_reconstructed_data", pca_reconstructed_data[i, :])
                            i += 1

                    pca:Optional[PCA] = self.__project_file.load_proto_pca()
                    if pca is not None:
                        pca_features = pca.transform(normalized_data)
                        pca_reconstructed_data = pca.inverse_transform(pca_features)

                        i:int = 0
                        for row_node in row_nodes:
                            self.__project_file.create_array(row_node, "proto_pca_features", pca_features[i, :])
                            self.__project_file.create_array(row_node, "proto_pca_reconstructed_data", pca_reconstructed_data[i, :])
                            i += 1

            self.add_data_finished.emit(result)
        except BaseException as e:
            print(traceback.format_exc())
            self.add_data_error.emit(str(e))

    def remove_train_class(self, class_node_path:str):
        try:
            self.__project_file.remove_train_class(class_node_path)
            self.update_train_pca()
            self.update_dataloader()
            self.update_deep_dataloader()
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_train_row(self, row_node_path:str):
        try:
            self.__project_file.remove_node(row_node_path)
            self.update_train_pca()
            self.update_dataloader()
            self.update_deep_dataloader()
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_proto_train_class(self, class_node_path:str):
        try:
            self.__project_file.remove_proto_train_class(class_node_path)
            self.update_proto_train_pca()
            self.update_proto_dataloader()
            self.update_deep_proto_dataloader()
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_proto_train_row(self, row_node_path:str):
        try:
            self.__project_file.remove_node(row_node_path)
            self.update_proto_train_pca()
            self.update_proto_dataloader()
            self.update_deep_proto_dataloader()
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_proto_support_class(self, class_node_path:str):
        try:
            self.__project_file.remove_node(class_node_path)
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_proto_support_row(self, row_node_path:str):
        try:
            self.__project_file.remove_node(row_node_path)
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_query_class(self, class_node_path:str):
        try:
            self.__project_file.remove_query_class(class_node_path)
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def remove_query_row(self, row_node_path:str):
        try:
            self.__project_file.remove_node(row_node_path)
            self.remove_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.remove_error.emit(str(e))

    def clear_train_classes(self)->None:
        try:
            self.__project_file.clear_train_classes()
            self.update_dataloader()
            self.update_deep_dataloader()
            self.clear_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.clear_error.emit(str(e))

    def clear_query_classes(self)->None:
        try:
            self.__project_file.clear_query_classes()
            self.clear_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.clear_error.emit(str(e))

    def clear_proto_train_classes(self)->None:
        try:
            self.__project_file.clear_proto_train_classes()
            self.update_proto_dataloader()
            self.update_deep_proto_dataloader()
            self.clear_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.clear_error.emit(str(e))

    def clear_proto_support_classes(self)->None:
        try:
            self.__project_file.clear_proto_support_classes()
            self.clear_finished.emit()
        except BaseException as e:
            print(traceback.format_exc())
            self.clear_error.emit(str(e))

    def remove_cosmic_rays(self, data:np.ndarray):
        w = 3
        c = 20
        used_data = data.copy()

        resid = white_tophat(used_data, size=(1, w))
        med_resid = np.median(resid, axis=1, keepdims=True)
        std_resid = np.std(resid, axis=1, keepdims=True)
        thr = med_resid + c * std_resid
        mask = (resid > thr)
        filtered = median_filter(used_data, size=(1, w))
        used_data[mask] = filtered[mask]

        return used_data

    def interpolate(self, data:np.ndarray, full:bool)->np.ndarray:
        x = self.__project_file.raman_shift if not full else self.__project_file.full_raman_shift
        len_x = len(x)
        if data.shape[0] != len_x or np.any(data[:, 0] != x):
            new_data = np.zeros((len_x, data.shape[1]))
            new_data[:, 0] = x

            f = interp1d(data[:, 0], data[:, 1:], axis=0, kind='linear', fill_value="extrapolate")
            new_data[:, 1:] = f(x)

            data = new_data

        return data[:, 1:].transpose()

    def remove_baseline(self, data:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        used_data:np.ndarray = data.copy()
        baseline_fitter = Baseline(x_data=self.__project_file.raman_shift)
        baseline_data = []
        for i in range(used_data.shape[0]):
            # baseline, _ = baseline_fitter.asls(used_data[i, :], lam=10000, p=0.01)
            baseline, _ = baseline_fitter.airpls(used_data[i, :])
            used_data[i, :] -= baseline
            baseline_data.append(baseline)

        return used_data, np.array(baseline_data)

    def filter(self, data:np.ndarray)->np.ndarray:
        return savgol_filter(
            data,
            window_length=21,
            polyorder=2,
            mode='interp',
            axis=1
        )

    def normalize(self, data:np.ndarray)->np.ndarray:
        n = data.shape[1]
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        return np.sqrt(n) * np.where(norms == 0, data, data / norms)

    def label_of(self, class_id:int)->str:
        return str(self.__project_file.get_attr(self.__project_file.id_to_label, str(class_id)))

    def class_id_of(self, label:str)->int:
        return int(self.__project_file.get_attr(self.__project_file.label_to_id, label))

    def move_to_main_thread(self):
        self._request_move_to_main_thread.emit()

    def on_move_to_main_thread(self):
        self.moveToThread(QCoreApplication.instance().thread())
