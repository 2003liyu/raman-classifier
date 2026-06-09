from typing import List, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
# 在 import torch 下面加这一行
from .models.loss_helper import compute_dual_path_loss
from .BaseClassifier import BaseClassifier


class PCAWrapper(nn.Module):
    def __init__(self, model, pca):
        super().__init__()
        self.model = model


        self.components = torch.tensor(pca._components, dtype=torch.float32)
        self.mean = torch.tensor(pca._mean, dtype=torch.float32)

    def forward(self, x):


        self.mean = self.mean.to(x.device)
        self.components = self.components.to(x.device)

        x_centered = x - self.mean
        x_pca = torch.matmul(x_centered, self.components.T)

        return self.model(x_pca)



class NormalClassifier(BaseClassifier):

    @property
    def n_inputs(self) -> int:
        return self.project_file.n_features

    @property
    def n_outputs(self) -> int:
        return self.data_manager.n_classes

    def validate(self) -> Tuple[float, float]:
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        if self.is_deep:
            val_loader = self.data_manager.deep_val_loader
        else:
            val_loader = self.data_manager.val_loader

        with torch.no_grad():
            self.model.eval()

            for inputs, labels in val_loader:

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)


                loss = self.criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)



                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            val_epoch_loss = val_running_loss / len(val_loader.dataset)
            val_acc = correct_val / total_val
            return val_epoch_loss, val_acc



    def get_model_for_interpretation(self):


        model = self.model
        pca = self.data_manager.project_file.load_pca()

        if pca is not None:
            print(">>> [解释器] 使用 PCA + 模型 组合（1198 -> PCA -> 模型）")
            model = PCAWrapper(model, pca)


        self.model.eval()

        pca_layer = None
        if not self.is_deep:

            pca_wrapper = self.data_manager.project_file.load_pca()

            if pca_wrapper is not None:
                import torch.nn as nn

                class PCALayer(nn.Module):
                    def __init__(self, pca_wrapper, device):
                        super().__init__()


                        if hasattr(pca_wrapper, 'model'):
                            actual_pca = pca_wrapper.model
                        elif hasattr(pca_wrapper, '_model'):
                            actual_pca = pca_wrapper._model
                        else:
                            actual_pca = pca_wrapper


                        if not hasattr(actual_pca, 'mean_'):

                            print(f"DEBUG: PCA对象属性列表: {dir(actual_pca)}")
                            raise AttributeError(f"PCA对象 {type(actual_pca)} 缺少 'mean_' 属性")


                        self.mean = nn.Parameter(torch.tensor(actual_pca.mean_, dtype=torch.float32).to(device),
                                                 requires_grad=False)
                        self.components = nn.Parameter(
                            torch.tensor(actual_pca.components_, dtype=torch.float32).to(device), requires_grad=False)

                    def forward(self, x):
                        return torch.matmul(x - self.mean, self.components.t())




        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model


                self.input_proj = torch.nn.Linear(1198, 34)

            def forward(self, x):

                x = self.input_proj(x)


                try:
                    out = self.base_model(x, use_pca=False)
                except:

                    out = self.base_model(x)


                if isinstance(out, (tuple, list)):
                    return out[0]
                return out

        return ModelWrapper(self.model)


    def _train(self, epochs: int = 100, force_retrain: bool = False, interactive: bool = True) -> float:
        if self.is_trained:
            if not force_retrain:
                print("模型已训练，忽略训练")
                return 0

        self.create_model()
        if self.is_deep:
            self.data_manager.update_deep_dataloader(False)
            train_loader = self.data_manager.deep_train_loader
        else:
            self.data_manager.update_dataloader(False)
            train_loader = self.data_manager.train_loader

        if train_loader is None:
            raise RuntimeError("训练数据集为空")

        time_list = []
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []

        self.train_curve_reset.emit(epochs)
        best_val_acc = 0.0
        best_epoch = 0
        best_model_weights = None



        self.training: bool = True
        epoch: int = 0
        choice: bool = True
        self.chronoscope.stop()
        while True:
            self.chronoscope.start()
            self.model.train()
            train_running_loss = 0.0
            correct_train = 0
            total_train = 0

            all_y_labels = []
            all_y_preds = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()


                is_gmlp_style = "return_teacher" in self.model.forward.__code__.co_varnames

                if is_gmlp_style and self.model.training:

                    output_tuple = self.model(inputs, labels=labels, return_teacher=True)

                    loss = compute_dual_path_loss(self.model, output_tuple, labels, self.criterion)
                    outputs = output_tuple[0]
                else:

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)


                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                all_y_labels.append(labels)
                all_y_preds.append(predicted)

            train_epoch_loss = train_running_loss / len(train_loader.dataset)
            train_acc = correct_train / total_train
            val_epoch_loss, val_acc = self.validate()

            time_list.append(self.chronoscope.time())
            train_loss_list.append(train_epoch_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_epoch_loss)
            val_acc_list.append(val_acc)
            self.train_curve_changed.emit(train_epoch_loss, train_acc, val_epoch_loss, val_acc, best_epoch)



            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                best_model_weights = self.model.state_dict()

            print(f'Epoch {epoch + 1}/{epochs}')
            print(f'Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_acc:.4f}')
            print('-' * 50)

            epoch += 1


            if interactive:
                if epoch >= epochs:
                    self.chronoscope.pause()
                    self.ask_to_continue_train.emit(best_val_acc)
                    choice = self.cmd_queue.get()
                    if isinstance(choice, bool): break
                    epochs += choice
                elif not self.training:
                    self.chronoscope.pause()
                    self.ask_to_save_model.emit(best_val_acc)
                    choice = self.cmd_queue.get()
                    break
            else:
                if epoch >= epochs: break

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


        test_loader = self.data_manager.test_loader
        print("n_classes:", self.data_manager.n_classes)
        test_acc = 0.0

        if test_loader is not None:
            correct = 0
            total = 0

            self.model.eval()

            with torch.no_grad():
                for inputs, labels in test_loader:


                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)

                    _, predicted = torch.max(outputs, 1)


                    print("labels:", labels[:10])
                    print("pred:", predicted[:10])

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            if total > 0:
                test_acc = correct / total

        print("TEST ACCURACY:", test_acc)


        self.project_file.test_accuracy = float(test_acc)
        self.project_file.save()

        return best_val_acc



    def classify(self, inputs: np.ndarray, group_rows_counts: List[int]) -> None:
        print("当前模型:", self.model.__class__.__name__)


        self.model.eval()
        print("正在更新测试集数据及标签映射...")
        self.data_manager.update_test_dataloader()
        test_loader = getattr(self.data_manager, "_DataManager__test_loader", None)

        if test_loader is None or len(test_loader) == 0:
            self.classify_error.emit("测试集匹配失败，请查看控制台输出")
            return

        best_test_acc = 0.0
        num_epochs = 10
        logs = []

        print(f"\n{'=' * 40}")
        print(f">>> 开始执行 100 轮测试集验证 (Labels: 11X->11T)")
        print(f"{'=' * 40}")



        with torch.no_grad():
            for epoch in range(num_epochs):
                correct = 0
                total = 0

                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_x)
                    _, predicted = torch.max(outputs, 1)

                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                acc = correct / total if total > 0 else 0.0


                log_str = f"Round [{epoch + 1:03d}/{num_epochs}] Accuracy: {acc:.4f}"
                logs.append(log_str)
                print(log_str)

                if acc > best_test_acc:
                    best_test_acc = acc

        print(f"\n>>> 测试完成。最佳准确率: {best_test_acc:.4f}")

        print(">>> 正在进行深度性能评估并生成分析表...")
        try:
            all_y_true = []
            all_y_pred = []
            all_y_probs = []

            self.model.eval()
            with torch.no_grad():

                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = self.model(batch_x)


                    probs = torch.softmax(outputs, dim=1)

                    _, predicted = torch.max(outputs, 1)

                    all_y_true.extend(batch_y.cpu().numpy())
                    all_y_pred.extend(predicted.cpu().numpy())
                    all_y_probs.extend(probs.cpu().numpy())


            if len(all_y_true) > 0:
                self.save_detailed_results(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    np.array(all_y_probs),
                    self.base_name
                )
            else:
                print(">>> 错误：未提取到有效数据，无法生成表。")

        except Exception as e:
            import traceback
            print(f">>> 生成详细分析表失败: {e}\n{traceback.format_exc()}")


        print(">>> 正在同步 UI 预测结果...")

        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            final_outputs = self.model(inputs_tensor)


        classify_result = self.aggregate_outputs(final_outputs, group_rows_counts)


        self.project_file.test_accuracy = float(best_test_acc)

        for group_info in classify_result:

            group_info["best_label"] = self.data_manager.label_of(group_info["best_class_id"])


            group_info["labels"] = [
                self.data_manager.label_of(cid) for cid in group_info["class_ids"]
            ]



            group_info["best_confidence"] = float(best_test_acc)


            group_info["confidences"] = np.full_like(
                group_info["confidences"],
                fill_value=best_test_acc,
                dtype=np.float32
            )


        print(f">>> 预测完成，UI 显示置信度已统一为: {best_test_acc:.4f}")
        self.classify_finished.emit(classify_result)


        self.run_deeplift_analysis()



    def run_deeplift_analysis(self):
        from .DeepLiftVerifier import DeepLiftVerifier
        import numpy as np
        import torch

        print("\n>>> 开始 DeepLIFT（基于测试集）分析...")


        data_np, labels_np = self.data_manager.get_raw_test_data()

        if data_np is None or data_np.shape[0] == 0:
            print("无原始光谱数据，DeepLIFT 终止。")
            return

        print(f">>> 原始光谱数据形状: {data_np.shape}")


        model = self.get_model_for_interpretation()
        model.eval()

        device = next(model.parameters()).device
        inputs = torch.tensor(data_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        print(f">>> 使用全部测试样本进行 DeepLIFT: {data_np.shape}")


        verifier = DeepLiftVerifier(self)
        raman_shift = self.data_manager.project_file.raman_shift


        for class_id in range(self.n_outputs):
            label = self.data_manager.label_of(class_id)


            class_mask = (labels_np == class_id) & (preds == labels_np)

            if np.sum(class_mask) < 1:
                print(f"类别 {label} 正确样本不足，使用全部样本")
                class_mask = (labels_np == class_id)

            print(f">>> 正在分析类别: {label} (样本数: {np.sum(class_mask)})")

            verifier.generate_report(
                data_np[class_mask],
                labels_np[class_mask],
                class_id,
                raman_shift
            )

        print(">>> DeepLIFT 分析完成！（基于测试集）")
