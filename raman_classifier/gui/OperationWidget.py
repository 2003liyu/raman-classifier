from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QHBoxLayout, QGroupBox, QLabel, QComboBox, QMessageBox, \
    QFormLayout

from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .MainWindow import MainWindow

from .common import PlotWidget
from .AskToContinueTrainDialog import AskToContinueTrainDialog
from .AskToTrainDialog import AskToTrainDialog

from matplotlib.lines import Line2D
import numpy as np


class OperationWidget(QGroupBox):

    def __init__(self, main_window: MainWindow = None) -> None:
        QGroupBox.__init__(self, "操作选项")
        self.main_window: MainWindow = main_window
        self.line_train_loss: Optional[Line2D] = None
        self.line_val_loss: Optional[Line2D] = None
        self.line_train_acc: Optional[Line2D] = None
        self.line_val_acc: Optional[Line2D] = None
        self.line_best_epoch: Optional[Line2D] = None
        self.plot_index: int = 0
        self.should_question: int = False

        vlayout_main = QVBoxLayout()

        hlayout_models = QHBoxLayout()
        hlayout_models.addWidget(QLabel("模型", self))
        self.combobox_models = QComboBox(self)
        self.combobox_models.addItem("多层感知机 (MLP)", userData="MLP")
        self.combobox_models.addItem("门控多层感知机 (gMLP)", userData="gMLP")
        self.combobox_models.addItem("原型门控多层感知机 (ProtoGMLP)", userData="ProtoGMLP")
        self.combobox_models.addItem("Kolmogorov-Arnold 网络 (KAN)", userData="KAN")
        self.combobox_models.addItem("卷积网络 (CNN)", userData="CNN")
        self.combobox_models.addItem("残差网络 (ResNet)", userData="ResNet")
        self.combobox_models.addItem("残差网络V2 (ResNetV2)", userData="ResNetV2")
        self.combobox_models.addItem("聚合残差网络 (ResNeXt)", userData="ResNeXt")
        self.combobox_models.addItem("聚合残差网络V2 (ResNeXtV2)", userData="ResNeXtV2")
        self.combobox_models.addItem("深度残差收缩网络 CS (DRSN-CS)", userData="DRSN_CS")
        self.combobox_models.addItem("深度残差收缩网络 CW (DRSN-CW)", userData="DRSN_CW")
        self.combobox_models.addItem("原型多层感知机 (ProtoMLP)", userData="ProtoMLP")
        self.combobox_models.addItem("原型 Kolmogorov-Arnold 网络 (ProtoKAN)", userData="ProtoKAN")
        self.combobox_models.addItem("原型卷积网络 (ProtoCNN)", userData="ProtoCNN")
        self.combobox_models.addItem("原型残差网络 (ProtoResNet)", userData="ProtoResNet")
        self.combobox_models.addItem("原型残差网络V2 (ProtoResNetV2)", userData="ProtoResNetV2")
        self.combobox_models.addItem("原型聚合残差网络 (ProtoResNeXt)", userData="ProtoResNeXt")
        self.combobox_models.addItem("原型聚合残差网络V2 (ProtoResNeXtV2)", userData="ProtoResNeXtV2")
        self.combobox_models.addItem("原型深度残差收缩网络 CS (ProtoDRSN-CS)", userData="ProtoDRSN_CS")
        self.combobox_models.addItem("原型深度残差收缩网络 CW (ProtoDRSN-CW)", userData="ProtoDRSN_CW")
        hlayout_models.addWidget(self.combobox_models)
        vlayout_main.addLayout(hlayout_models)


        self.button_train = QPushButton("训练")
        self.button_classify = QPushButton("分类")
        self.button_pwm = QPushButton("反向验证")
        self.button_clear_pwm = QPushButton("清除所有验证")


        hlayout_buttons = QHBoxLayout()
        hlayout_buttons.addWidget(self.button_train)
        hlayout_buttons.addWidget(self.button_classify)


        hlayout_pwm = QHBoxLayout()
        hlayout_pwm.addWidget(self.button_pwm)
        hlayout_pwm.addWidget(self.button_clear_pwm)


        vlayout_main.addLayout(hlayout_buttons)
        vlayout_main.addLayout(hlayout_pwm)


        self.button_train.clicked.connect(self.on_train_clicked)
        self.button_classify.clicked.connect(self.on_classify_clicked)
        self.button_pwm.clicked.connect(self.on_pwm_clicked)
        self.button_clear_pwm.clicked.connect(self.main_window.clear_pwm_curves)


        form_layout = QFormLayout()
        self.label_best_acc = QLabel("N/A", self)
        self.label_n_parameters = QLabel("N/A", self)
        self.label_flops = QLabel("N/A", self)
        form_layout.addRow("准确率:", self.label_best_acc)
        form_layout.addRow("参数量:", self.label_n_parameters)
        form_layout.addRow("浮点运算量:", self.label_flops)
        vlayout_main.addLayout(form_layout)

        self.setLayout(vlayout_main)
        self.__ask_to_continue_train_dialog: Optional[AskToContinueTrainDialog] = None
        self.__ask_epochs_dialog: Optional[AskToTrainDialog] = None
        self.__ask_to_train_dialog: Optional[AskToTrainDialog] = None
        self.__ask_to_retrain_dialog: Optional[AskToTrainDialog] = None

        self.combobox_models.currentIndexChanged.connect(self.on_model_changed)
        self.main_window.api.train_finished.connect(self.on_train_finished)
        self.main_window.api.train_error.connect(self.on_train_error)
        self.main_window.api.train_curve_reset.connect(self.on_train_curve_reset)
        self.main_window.api.train_curve_changed.connect(self.on_train_curve_changed)
        self.main_window.api.classify_error.connect(self.on_classify_error)
        self.main_window.api.classify_finished.connect(self.on_classify_finished)

        self.main_window.api.ask_to_continue_train.connect(self.on_ask_to_continue_train)
        self.main_window.api.ask_to_save_model.connect(self.on_ask_to_save_model)
        self.main_window.api.ask_to_train.connect(self.on_ask_to_train)
        self.load_model_info()

    @property
    def ask_epochs_dialog(self) -> AskToTrainDialog:
        if self.__ask_epochs_dialog is None:
            self.__ask_epochs_dialog = AskToTrainDialog(self)
            self.__ask_epochs_dialog.setWindowTitle("开始训练")
            self.__ask_epochs_dialog.label_prompt.hide()
            self.__ask_epochs_dialog.label_train_prompt.setText("训练轮数")

        return self.__ask_epochs_dialog

    def on_train_clicked(self):
        if self.button_train.text() == "训练":
            if self.main_window.api.is_trained:
                self.ask_to_retrain_dialog.exec()
                if self.ask_to_retrain_dialog.epochs > 0:
                    self.should_question: int = True
                    self.main_window.api.train(epochs=self.ask_to_retrain_dialog.epochs, force_retrain=True)
                    self.main_window.set_as_trainning(True)
            else:
                self.ask_epochs_dialog.exec()
                if self.ask_epochs_dialog.epochs > 0:
                    self.should_question: int = True
                    self.main_window.api.train(epochs=self.ask_epochs_dialog.epochs)
                    self.main_window.set_as_trainning(True)
        else:
            self.main_window.api.stop_train()
            self.main_window.set_as_trainning(False)

    @property
    def ask_to_retrain_dialog(self) -> AskToTrainDialog:
        if self.__ask_to_retrain_dialog is None:
            self.__ask_to_retrain_dialog = AskToTrainDialog(self)
            self.__ask_to_retrain_dialog.setWindowTitle("模型已训练")
            self.__ask_to_retrain_dialog.label_prompt.setText("模型已训练，是否重新训练？")
            self.__ask_to_retrain_dialog.label_train_prompt.setText("训练轮数")

        return self.__ask_to_retrain_dialog

    @property
    def ask_to_train_dialog(self) -> AskToTrainDialog:
        if self.__ask_to_train_dialog is None:
            self.__ask_to_train_dialog = AskToTrainDialog(self)
            self.__ask_to_train_dialog.setWindowTitle("模型未训练")
            self.__ask_to_train_dialog.label_prompt.setText("模型未训练，是否启动训练？")
            self.__ask_to_train_dialog.label_train_prompt.setText("训练轮数")

        return self.__ask_to_train_dialog


    def on_ask_to_train(self):
        self.ask_to_train_dialog.exec()
        if self.ask_to_train_dialog.epochs > 0:
            self.main_window.set_as_trainning(True)
        else:
            self.main_window.set_as_trainning(False)

        self.main_window.api.add_cmd(self.ask_to_train_dialog.epochs)

    def on_classify_clicked(self):
        self.should_question: int = False
        self.main_window.left_sidebar.query_data_tree.classify_all()
        self.main_window.set_as_classifing(True)

    def on_train_finished(self) -> None:
        self.main_window.set_as_trainning(False)
        QMessageBox.information(self.main_window, "训练已完成", "训练已完成！")

    def on_train_error(self, error_message: str) -> None:
        self.main_window.set_as_trainning(False)
        QMessageBox.critical(self.main_window, "训练过程出错", error_message)


    def on_pwm_clicked(self):

        if self.main_window.api.classifier is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "错误", "模型尚未初始化，请先加载项目或选择模型。")
            return


        from PySide6.QtWidgets import QInputDialog, QMessageBox
        dm = self.main_window.api.data_manager

        if dm.n_classes == 0:
            QMessageBox.warning(self, "提示", "当前数据管理器中没有类别数据。")
            return


        class_items = [f"{i}: {dm.label_of(i)}" for i in range(dm.n_classes)]
        item, ok = QInputDialog.getItem(
            self,
            "选择验证类别",
            "请选择要进行 PWM 反向验证的类别:",
            class_items,
            0,
            False
        )

        if ok and item:
            try:

                class_id = int(item.split(":")[0])

                self.main_window.status_bar.showMessage("正在计算反向验证梯度...", 2000)
                self.main_window.api.run_reverse_validation(class_id)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "验证失败", f"执行过程中发生错误:\n{str(e)}")


    @property
    def ask_to_continue_train_dialog(self) -> AskToContinueTrainDialog:
        if self.__ask_to_continue_train_dialog is None:
            self.__ask_to_continue_train_dialog = AskToContinueTrainDialog(self)

        return self.__ask_to_continue_train_dialog

    def on_ask_to_save_model(self, best_val_acc: float) -> None:
        button = QMessageBox.question(self, "训练已停止",
                                      f"训练已停止，验证集最佳准确率为 {100 * best_val_acc:.2f}%，是否采纳模型？")
        choice: bool = (button == QMessageBox.StandardButton.Yes)
        self.main_window.api.add_cmd(choice)

        if not choice:
            self.clear_model_info()

        self.main_window.set_as_trainning(False)

    def on_ask_to_continue_train(self, best_val_acc: float) -> None:
        self.ask_to_continue_train_dialog.acc_label.setText(f"{100 * best_val_acc:.2f}%")
        self.ask_to_continue_train_dialog.exec()
        choice = self.ask_to_continue_train_dialog.choice
        if isinstance(choice, int):
            old_xdata = self.line_train_loss.get_xdata()
            new_epochs = len(old_xdata) + choice
            new_xdata = np.arange(new_epochs)

            old_ydata = self.line_train_loss.get_ydata()
            new_ydata = np.full((new_epochs,), np.nan)
            new_ydata[:len(old_ydata)] = old_ydata
            self.line_train_loss.set_data(new_xdata, new_ydata)

            old_ydata = self.line_val_loss.get_ydata()
            new_ydata = np.full((new_epochs,), np.nan)
            new_ydata[:len(old_ydata)] = old_ydata
            self.line_val_loss.set_data(new_xdata, new_ydata)

            old_ydata = self.line_train_acc.get_ydata()
            new_ydata = np.full((new_epochs,), np.nan)
            new_ydata[:len(old_ydata)] = old_ydata
            self.line_train_acc.set_data(new_xdata, new_ydata)

            old_ydata = self.line_val_acc.get_ydata()
            new_ydata = np.full((new_epochs,), np.nan)
            new_ydata[:len(old_ydata)] = old_ydata
            self.line_val_acc.set_data(new_xdata, new_ydata)

            self.loss_plot_widget.xlim(0, new_epochs)
            self.acc_plot_widget.xlim(0, new_epochs)

        self.main_window.api.add_cmd(choice)

        if not choice:
            self.clear_model_info()
            self.main_window.set_as_trainning(False)

    def on_classify_error(self, error_message: str):
        self.main_window.set_as_classifing(False)
        QMessageBox.warning(self.main_window, "分类过程出错", error_message)

    def on_classify_finished(self, classify_result):
        self.main_window.left_sidebar.query_data_tree.classify_finished(classify_result)
        self.main_window.set_as_classifing(False)
        QMessageBox.information(self.main_window, "分类成功", "分类已完成！")



    @property
    def loss_plot_widget(self) -> PlotWidget:
        return self.main_window.plot_tab_widget.loss_plot_widget

    @property
    def acc_plot_widget(self) -> PlotWidget:
        return self.main_window.plot_tab_widget.acc_plot_widget

    def on_train_curve_reset(self, epochs: int) -> None:
        self.plot_index = 0

        xdata = np.arange(epochs)
        ydata = np.full((epochs,), np.nan)
        if self.line_train_loss is None:
            self.line_train_loss = self.loss_plot_widget.plot(xdata, ydata, label="训练集损失")
            self.line_val_loss = self.loss_plot_widget.plot(xdata, ydata, label="验证集损失")
            self.line_train_acc = self.acc_plot_widget.plot(xdata, ydata, label="训练集准确率")
            self.line_val_acc = self.acc_plot_widget.plot(xdata, ydata, label="验证集准确率")
            self.line_best_epoch = self.acc_plot_widget.plot([0, 0], [0, 0], color="red", linestyle="--",
                                                             label="最佳验证集准确率=0%")
        else:
            self.line_train_loss.set_data(xdata, ydata)
            self.line_val_loss.set_data(xdata, ydata)
            self.line_train_acc.set_data(xdata, ydata)
            self.line_val_acc.set_data(xdata, ydata)
            self.line_best_epoch.set_data([0, 0], [0, 0])
            self.line_best_epoch.set_label("最佳验证集准确率=0%")

        self.loss_plot_widget.xlim(0, epochs)
        self.loss_plot_widget.ylim(0, 1)
        self.acc_plot_widget.xlim(0, epochs)
        self.acc_plot_widget.ylim(0, 1)

        self.loss_plot_widget.update()
        self.acc_plot_widget.update()

        self.main_window.plot_tab_widget.setCurrentIndex(1)


    def on_train_curve_changed(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float,
                               best_epoch: int) -> None:
        y = self.line_train_loss.get_ydata()
        y[self.plot_index] = train_loss
        self.line_train_loss.set_ydata(y)

        y = self.line_train_acc.get_ydata()
        y[self.plot_index] = train_acc
        self.line_train_acc.set_ydata(y)

        y = self.line_val_loss.get_ydata()
        y[self.plot_index] = val_loss
        self.line_val_loss.set_ydata(y)

        y = self.line_val_acc.get_ydata()
        y[self.plot_index] = val_acc
        self.line_val_acc.set_ydata(y)
        best_val_acc = y[best_epoch]

        self.line_best_epoch.set_data([best_epoch, best_epoch], [0, best_val_acc])
        self.line_best_epoch.set_label(f"最佳验证集准确率={100 * best_val_acc:.2f}%")
        self.label_best_acc.setText(f"{100 * best_val_acc:.2f}%")

        max_loss = max(train_loss, val_loss)
        min_acc = min(train_acc, val_acc)
        if self.plot_index == 0:
            self.loss_plot_widget.ylim(0, 1.1 * max_loss)
            self.acc_plot_widget.ylim(max(0, min_acc - 0.1), 1)
        else:
            _, current_max_loss = self.loss_plot_widget.ylim()
            if max_loss > 0.9 * current_max_loss:
                current_max_loss = 1.1 * max_loss
                self.loss_plot_widget.ylim(0, current_max_loss)

            current_min_acc, _ = self.acc_plot_widget.ylim()
            if min_acc < current_min_acc:
                current_min_acc = max(0, min_acc - 0.1)
                self.acc_plot_widget.ylim(current_min_acc, 1)

        self.loss_plot_widget.update()
        self.acc_plot_widget.update()

        self.plot_index += 1


    def clear_model_info(self):
        if self.line_train_loss is not None:
            self.loss_plot_widget.remove(self.line_train_loss)
            self.loss_plot_widget.remove(self.line_val_loss)
            self.acc_plot_widget.remove(self.line_train_acc)
            self.acc_plot_widget.remove(self.line_val_acc)
            self.acc_plot_widget.remove(self.line_best_epoch)
            self.line_train_loss = None
            self.line_val_loss = None
            self.line_train_acc = None
            self.line_val_acc = None
            self.line_best_epoch = None

        self.label_best_acc.setText("N/A")

    @staticmethod
    def numstr(num: Union[int, float]) -> str:
        K = 1000
        M = 1000 * K
        G = 1000 * M
        T = 1000 * G
        P = 1000 * T
        if num > P:
            return f"{num / P:.2f} P"
        elif num > T:
            return f"{num / T:.2f} T"
        elif num > G:
            return f"{num / G:.2f} G"
        elif num > M:
            return f"{num / M:.2f} M"
        elif num > K:
            return f"{num / K:.2f} K"
        else:
            return str(num)

    def load_model_info(self, load_curve: bool = True) -> None:
        n_parameters = self.main_window.api.classifier.n_parameters
        flops = self.main_window.api.classifier.flops
        if n_parameters is None:
            self.label_n_parameters.setText("N/A")
            self.label_flops.setText("N/A")
        else:
            self.label_n_parameters.setText(self.numstr(n_parameters))
            self.label_flops.setText(self.numstr(flops) + "Flops")

        model_title: str = self.main_window.api.classifier.model_title
        try:
            model_info = self.main_window.api.project_file.load_model_info(model_title)
            train_loss_list = model_info["train_loss_list"]
            train_acc_list = model_info["train_acc_list"]
            val_loss_list = model_info["val_loss_list"]
            val_acc_list = model_info["val_acc_list"]
            best_epoch = model_info["best_epoch"]
            best_acc = model_info["best_acc"]
            epochs = len(train_loss_list)
            xdata = np.arange(epochs)

            self.label_best_acc.setText(f"{best_acc * 100:.2f}%")

            if load_curve:
                if self.line_train_loss is None:
                    self.line_train_loss = self.loss_plot_widget.plot(xdata, train_loss_list, label="训练集损失")
                    self.line_val_loss = self.loss_plot_widget.plot(xdata, val_loss_list, label="验证集损失")
                    self.line_train_acc = self.acc_plot_widget.plot(xdata, train_acc_list, label="训练集准确率")
                    self.line_val_acc = self.acc_plot_widget.plot(xdata, val_acc_list, label="验证集准确率")
                    self.line_best_epoch = self.acc_plot_widget.plot([best_epoch, best_epoch], [0, best_acc],
                                                                     color="red", linestyle="--",
                                                                     label=f"最佳验证集准确率={100 * best_acc:.2f}%")
                else:
                    self.line_train_loss.set_data(xdata, train_loss_list)
                    self.line_val_loss.set_data(xdata, val_loss_list)
                    self.line_train_acc.set_data(xdata, train_acc_list)
                    self.line_val_acc.set_data(xdata, val_acc_list)
                    self.line_best_epoch.set_data([best_epoch, best_epoch], [0, best_acc])
                    self.line_best_epoch.set_label(f"最佳验证集准确率={100 * best_acc:.2f}%")

                self.loss_plot_widget.xlim(0, epochs)
                self.loss_plot_widget.ylim(0, max(train_loss_list.max(), val_loss_list.max()) + 0.1)
                self.acc_plot_widget.xlim(0, epochs)
                self.acc_plot_widget.ylim(max(0, min(train_acc_list.min(), val_acc_list.min()) - 0.1), 1)

                self.loss_plot_widget.update()
                self.acc_plot_widget.update()
        except:
            self.clear_model_info()

    def on_model_changed(self) -> None:
        net_name: str = self.combobox_models.currentData(Qt.ItemDataRole.UserRole)


        self.main_window.api.active_net = net_name


        self.main_window.api.project_file.net = net_name


        if net_name in ["gMLP", "ProtoGMLP"]:
            self.main_window.api.project_file.gmlp_config = [
                512, 256, 256,
                {

                    "lr": 0.0001,
                    "dropout1": 0.4,
                    "dropout2": 0.3,
                    "proto_lr_ratio": 0.2,
                    "kd_alpha": 0.2,
                    "kd_temperature": 2
                }
            ]
            print(">>> gMLP 参数已写入 ProjectFile.gmlp_config")


        if net_name.lower().startswith("proto"):
            self.main_window.left_sidebar.set_as_proto(True)
        else:
            self.main_window.left_sidebar.set_as_proto(False)


        self.main_window.right_sidebar.plot_options_widget.line_type_chagned.emit(
            self.main_window.right_sidebar.plot_options_widget.line_type
        )


        self.load_model_info()


    def set_as_trainning(self, trainning: bool) -> None:
        self.button_train.setText("停止训练" if trainning else "训练")
        self.combobox_models.setEnabled(not trainning)
        self.button_classify.setEnabled(not trainning)

    def set_as_deleting(self, deleting: bool) -> None:
        self.combobox_models.setEnabled(not deleting)
        self.button_train.setEnabled(not deleting)
        self.button_classify.setEnabled(not deleting)

    def set_as_classifing(self, classifing: bool) -> None:
        self.combobox_models.setEnabled(not classifing)
        self.button_classify.setEnabled(not classifing)