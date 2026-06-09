from typing import Optional, Tuple, List, Dict, Any, Union

import tables
import numpy as np
import torch
import torch.nn.functional as F

from .BaseClassifier import BaseClassifier


class ProtoClassifier(BaseClassifier):

    @property
    def n_inputs(self)->int:
        return self.project_file.n_proto_features
    
    @property
    def n_outputs(self)->int:
        return self.data_manager.n_proto_outputs

    def validate(self)->Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if self.is_deep:
            train_dataset = self.data_manager.deep_proto_train_dataset
            val_loader = self.data_manager.deep_proto_val_loader
        else:
            train_dataset = self.data_manager.proto_train_dataset
            val_loader = self.data_manager.proto_val_loader

        n_ways:int = train_dataset.n_ways
        n_support:int = train_dataset.n_support

        with torch.no_grad():
            for X_s, X_q, y_s, y_q in val_loader:
                X_s = X_s.squeeze(0).to(self.device)
                X_q = X_q.squeeze(0).to(self.device)
                y_q = y_q.squeeze(0).to(self.device)

                z_s = self.model(X_s)
                z_q = self.model(X_q)

                protos = z_s.view(n_ways, n_support, -1).mean(dim=1)
                dists = torch.cdist(z_q, protos)  # [N_q, n_ways]
                logits = -dists

                loss = F.cross_entropy(logits, y_q)
                preds = logits.argmax(dim=1)
                correct = preds.eq(y_q).sum().item()

                total_loss += loss.item() * y_q.size(0)
                total_correct += correct
                total_samples += y_q.size(0)

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        return val_loss, val_acc

    def _train(self, epochs:int=100, force_retrain:bool=False, interactive:bool=True)->float:
        if self.is_trained:
            if not force_retrain:
                print("模型已训练，忽略训练")
                return 0
            
        self.create_model()
        if self.is_deep:
            self.data_manager.update_deep_proto_dataloader(False)
            train_loader = self.data_manager.deep_proto_train_loader
            train_dataset = self.data_manager.deep_proto_train_dataset
        else:
            self.data_manager.update_proto_dataloader(False)
            train_loader = self.data_manager.proto_train_loader
            train_dataset = self.data_manager.proto_train_dataset
        
        if train_loader is None:
            raise RuntimeError("训练数据集为空")
        
        time_list = []
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        self.train_curve_reset.emit(epochs)
        best_val_acc = 0.0
        best_model_weights = None
        best_epoch = 0
        
        n_ways:int = train_dataset.n_ways
        n_support:int = train_dataset.n_support

        self.training = True
        epoch:int = 0
        choice:bool = True
        self.chronoscope.stop()
        while True:
            self.chronoscope.start()
            self.model.train()
            train_loss_accum = 0.0
            train_correct = 0
            train_samples = 0

            for X_s, X_q, y_s, y_q in train_loader:
                X_s = X_s.squeeze(0).to(self.device)
                X_q = X_q.squeeze(0).to(self.device)
                y_q = y_q.squeeze(0).to(self.device)

                z_s = self.model(X_s)
                z_q = self.model(X_q)
                protos = z_s.view(n_ways, n_support, -1).mean(dim=1)
                dists = torch.cdist(z_q, protos)
                loss = F.cross_entropy(-dists, y_q)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = (-dists).argmax(dim=1)
                train_correct += preds.eq(y_q).sum().item()
                train_samples += y_q.size(0)
                train_loss_accum += loss.item() * y_q.size(0)

            train_loss = train_loss_accum / train_samples
            train_acc = train_correct / train_samples

            val_loss, val_acc = self.validate()

            time_list.append(self.chronoscope.time())
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            self.train_curve_changed.emit(train_loss, train_acc, val_loss, val_acc, best_epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()
                best_epoch = epoch

            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            print('-' * 50)

            epoch += 1

            if interactive:
                if epoch >= epochs:
                    self.chronoscope.pause()
                    while self.cmd_queue.qsize() > 0:
                        self.cmd_queue.get()

                    self.ask_to_continue_train.emit(best_val_acc)
                    choice:Union[bool, int] = self.cmd_queue.get()
                    if isinstance(choice, bool):
                        break

                    epochs += choice

                elif not self.training:
                    self.chronoscope.pause()
                    while self.cmd_queue.qsize() > 0:
                        self.cmd_queue.get()

                    self.ask_to_save_model.emit(best_val_acc)
                    choice:Union[bool, int] = self.cmd_queue.get()
                    break
            else:
                if epoch >= epochs:
                    break

        self.chronoscope.stop()

        if choice:
            if best_model_weights is not None:
                self.model.load_state_dict(best_model_weights)

            self.project_file.save_model(
                self.model,
                np.array(time_list),
                np.array(train_loss_list),
                np.array(train_acc_list),
                np.array(val_loss_list),
                np.array(val_acc_list),
                best_epoch
            )
            self.train_finished.emit()
        else:
            self.model = None

        return best_val_acc
    
    def _classify(self, data:np.ndarray, groups_info:Dict[str, Dict[str, Any]]):
        protos_mat, labels = self.update_protos()

        inputs = torch.tensor(data, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            dists = torch.cdist(outputs, protos_mat)
        
        classify_result = self.aggregate_outputs(-dists, groups_info)
        for group_info in classify_result:
            group_info["best_label"] = labels[group_info["best_class_id"]]
            group_info["labels"] = [labels[class_id] for class_id in group_info["class_ids"]]

        self.classify_finished.emit(classify_result)

    def update_protos(self)->Tuple[torch.Tensor, List[str]]:
        if not self.is_trained:
            while self.cmd_queue.qsize() > 0:
                self.cmd_queue.get()
            
            self.ask_to_train.emit()
            epochs = self.cmd_queue.get()
            if epochs == 0:
                raise RuntimeError("训练已取消")
            
            self._train(epochs)
        
        proto_train_mtime:float = self.project_file.proto_train_classes._v_attrs.modify_time
        proto_support_mtime:float = self.project_file.proto_support_classes._v_attrs.modify_time
        model_title:str = self.model.title

        protos_mat_node:Optional[tables.Array] = None
        protos_mat_mtime:float = 0.0
        if model_title in self.project_file.protos_data:
            protos_mat_node:Optional[tables.Array] = self.project_file.protos_data[model_title]
            protos_mat_mtime:float = protos_mat_node._v_attrs.modify_time

        if protos_mat_mtime >= max(proto_train_mtime, proto_support_mtime):
            protos_mat = protos_mat_node.read()
            return torch.tensor(protos_mat, dtype=torch.float32).to(self.device), list(protos_mat_node._v_attrs.labels)

        need_update_protos_rows = []
        need_update_protos_rows_pca_features = []
        need_update_protos_classes = []
        if self.is_deep:
            features_key:str = "normalized_data"
        else:
            features_key:str = "proto_pca_features"

        proto_train_classes_protos:Dict[str, np.ndarray] = {}
        for class_node in self.project_file.proto_train_classes:
            class_protos_node:Optional[tables.Array] = None
            class_protos_mtime:float = 0.0
            if model_title in class_node.protos:
                class_protos_node:Optional[tables.Array] = class_node.protos[model_title]
                class_protos_mtime:float = class_protos_node._v_attrs.modify_time

            if class_protos_mtime >= class_node._v_attrs.modify_time:
                proto_train_classes_protos[str(class_node._v_attrs.label)] = class_protos_node.read()
                continue

            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                row_protos_node:Optional[tables.Array] = None
                row_protos_mtime:float = 0.0
                if model_title in row_node.protos:
                    row_protos_node:Optional[tables.Array] = row_node.protos[model_title]
                    row_protos_mtime:float = row_protos_node._v_attrs.modify_time

                if row_protos_mtime >= row_node[features_key]._v_attrs.modify_time:
                    continue

                need_update_protos_rows.append(row_node)
                need_update_protos_rows_pca_features.append(row_node[features_key].read())

            need_update_protos_classes.append(class_node)

        proto_support_classes_protos:Dict[str, np.ndarray] = {}
        for class_node in self.project_file.proto_support_classes:
            class_protos_node:Optional[tables.Array] = None
            class_protos_mtime:float = 0.0
            if model_title in class_node.protos:
                class_protos_node:Optional[tables.Array] = class_node.protos[model_title]
                class_protos_mtime:float = class_protos_node._v_attrs.modify_time

            if class_protos_mtime >= class_node._v_attrs.modify_time:
                proto_support_classes_protos[str(class_node._v_attrs.label)] = class_protos_node.read()
                continue

            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                row_protos_node:Optional[tables.Array] = None
                row_protos_node:float = 0.0
                if model_title in row_node.protos:
                    row_protos_node:Optional[tables.Array] = row_node.protos[model_title]
                    row_protos_mtime:float = row_protos_node._v_attrs.modify_time

                if row_protos_mtime >= row_node[features_key]._v_attrs.modify_time:
                    continue

                need_update_protos_rows.append(row_node)
                need_update_protos_rows_pca_features.append(row_node[features_key].read())

            need_update_protos_classes.append(class_node)

        if need_update_protos_rows_pca_features:
            need_update_protos_rows_pca_features = torch.tensor(np.array(need_update_protos_rows_pca_features), dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                need_update_protos_rows_protos = self.model(need_update_protos_rows_pca_features)
                for i, row_node in enumerate(need_update_protos_rows):
                    if model_title in row_node.protos:
                        self.project_file.remove_node(row_node.protos, model_title)

                    self.project_file.create_array(row_node.protos, model_title, need_update_protos_rows_protos[i, :].cpu().numpy())

        for class_node in need_update_protos_classes:
            rows_protos = []
            for row_node in class_node:
                if row_node._v_name == "protos":
                    continue

                rows_protos.append(row_node.protos[model_title].read())

            class_protos = np.array(rows_protos).mean(axis=0)
            if model_title in class_node.protos:
                self.project_file.remove_node(class_node.protos, model_title)

            self.project_file.create_array(class_node.protos, model_title, class_protos)
            class_name:str = str(class_node._v_name)
            if class_node._v_parent._v_parent._v_name == "proto_train_data":
                proto_train_classes_protos[class_name] = class_protos
            elif class_node._v_parent._v_parent._v_name == "proto_support_data":
                proto_support_classes_protos[class_name] = class_protos

        protos_map = {}
        for class_name, train_class_protos in proto_train_classes_protos.items():
            if class_name in proto_support_classes_protos:
                train_weight = len(self.project_file.proto_train_classes[class_name]) - 1
                support_weight = len(self.project_file.proto_support_classes[class_name]) - 1
                sum_weight = train_weight + support_weight
                train_weight /= sum_weight
                support_weight /= sum_weight

                class_protos = train_weight * train_class_protos + support_weight * proto_support_classes_protos[class_name]
                protos_map[class_name] = class_protos
            else:
                protos_map[class_name] = train_class_protos

        for class_name, support_class_protos in proto_support_classes_protos.items():
            if class_name in protos_map:
                continue

            protos_map[class_name] = support_class_protos

        if model_title in self.project_file.protos_data:
            self.project_file.remove_node(self.project_file.protos_data, model_title)

        protos_mat = np.array(list(protos_map.values()))
        labels = list(protos_map.keys())
        self.project_file.create_array(self.project_file.protos_data, model_title, protos_mat)
        self.project_file.set_attr(self.project_file.protos_data[model_title], "labels", labels)
        self.project_file.update_modify_time(self.project_file.models[model_title].state_dict)

        return torch.tensor(protos_mat, dtype=torch.float32).to(self.device), labels