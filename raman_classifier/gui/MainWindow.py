import os

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import QMainWindow, QWidget, QSplitter

from .StatusBar import StatusBar
from .MenuBar import MenuBar
from .PlotTabWidget import PlotTabWidget
from .LeftSidebar import LeftSidebar
from .RightSidebar import RightSidebar
from .common import FileChooser
from ..algorithm.API import API
from ..algorithm.GaussianFitTool import GaussianFitTool


class MainWindow(QMainWindow):

    def __init__(self, parent:QWidget=None)->None:
        QMainWindow.__init__(self, parent)
        self.base_title:str = "拉曼光谱分类器"
        self.api:API = API()
        GaussianFitTool.start()

        self_folder:str = os.path.dirname(os.path.abspath(__file__))
        self.file_chooser:FileChooser = FileChooser(self, self_folder + "/../data/file_chooser_history.txt")

        self.status_bar = StatusBar(self)
        self.setStatusBar(self.status_bar)
        self.central_widget:QSplitter = QSplitter(self, Qt.Orientation.Horizontal)
        self.left_sidebar:LeftSidebar = LeftSidebar(self)
        self.plot_tab_widget:PlotTabWidget = PlotTabWidget(self)
        self.right_sidebar:RightSidebar = RightSidebar(self)

        self.right_sidebar.plot_options_widget.line_type_chagned.connect(self.left_sidebar.train_data_tree.on_line_type_changed)
        self.right_sidebar.plot_options_widget.line_type_chagned.connect(self.left_sidebar.proto_train_data_tree.on_line_type_changed)
        self.right_sidebar.plot_options_widget.line_type_chagned.connect(self.left_sidebar.proto_support_data_tree.on_line_type_changed)
        self.right_sidebar.plot_options_widget.line_type_chagned.connect(self.left_sidebar.query_data_tree.on_line_type_changed)
        self.right_sidebar.plot_options_widget.show_grid_changed.connect(self.on_show_grid_changed)

        self.central_widget.addWidget(self.left_sidebar)
        self.central_widget.addWidget(self.plot_tab_widget)
        self.central_widget.addWidget(self.right_sidebar)
        self.central_widget.setStretchFactor(0, 1)
        self.central_widget.setStretchFactor(1, 2)
        self.central_widget.setStretchFactor(2, 1)

        self.setCentralWidget(self.central_widget)

        self.menu_bar:MenuBar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.api.project_file.project_name_changed.connect(self.on_project_name_changed)
        self.api.project_file.project_changed.connect(self.on_project_changed)
        self.api.project_file.project_saved.connect(self.on_project_saved)
        self.on_project_saved()

        self.api.reverse_validation_finished.connect(self.on_pwm_finished)

        self.setWindowIcon(QIcon(self_folder + "/../icons/raman_classifier.png"))

    def sizeHint(self)->QSize:
        return QSize(1280, 700)

    def closeEvent(self, close_event:QCloseEvent):
        self.menu_bar.ask_to_save()
        self.api.close()
        self.status_bar.close()
        GaussianFitTool.stop()

    @property
    def is_proto(self)->bool:
        return self.api.is_proto

    @property
    def is_deep(self)->bool:
        return self.api.is_deep

    def on_project_name_changed(self, project_name:str)->None:
        self.left_sidebar.train_data_tree.reload()
        self.left_sidebar.proto_train_data_tree.reload()
        self.left_sidebar.proto_support_data_tree.reload()
        self.left_sidebar.query_data_tree.reload()
        self.right_sidebar.operation_widget.load_model_info()

        if self.api.is_temp:
            project_name = "未命名项目"

        self.setWindowTitle(f"{project_name} - {self.base_title}")

    def on_project_changed(self)->None:
        project_name:str = self.api.project_file.file_name
        if self.api.is_temp:
            project_name = "未命名项目"

        self.setWindowTitle(f"{project_name} * - {self.base_title}")

    def on_project_saved(self)->None:
        project_name:str = self.api.project_file.file_name
        if self.api.is_temp:
            project_name = "未命名项目"

        self.setWindowTitle(f"{project_name} - {self.base_title}")

    def on_show_grid_changed(self, show:bool)->None:
        self.plot_tab_widget.raman_plot_widget.grid(show)
        self.plot_tab_widget.loss_plot_widget.grid(show)
        self.plot_tab_widget.acc_plot_widget.grid(show)

    def set_as_trainning(self, trainning:bool)->None:
        self.left_sidebar.set_as_trainning(trainning)
        self.right_sidebar.set_as_trainning(trainning)
        self.menu_bar.set_as_trainning(trainning)
        if trainning:
            self.status_bar.showMessage("正在训练...")
        else:
            self.status_bar.clearMessage()

    def set_as_classifing(self, classifing:bool)->None:
        self.left_sidebar.set_as_classifing(classifing)
        self.right_sidebar.set_as_classifing(classifing)
        self.menu_bar.set_as_classifing(classifing)
        if classifing:
            self.status_bar.showMessage("正在分类...")
        else:
            self.status_bar.clearMessage()

    def set_as_deleting(self, deleting:bool)->None:
        self.left_sidebar.set_as_deleting(deleting)
        self.right_sidebar.set_as_deleting(deleting)
        self.menu_bar.set_as_deleting(deleting)
        if deleting:
            self.status_bar.showMessage("正在删除...")
        else:
            self.status_bar.clearMessage()


    import numpy as np
    def on_pwm_finished(self, weights: np.ndarray, class_name: str):
        import numpy as np
        import random
        from scipy.interpolate import interp1d


        plot_widget = self.plot_tab_widget.raman_plot_widget
        ax = plot_widget.canvas.axes
        self.plot_tab_widget.setCurrentIndex(0)


        all_lines = [line for line in ax.get_lines() if "PWM" not in line.get_label()]
        if not all_lines:
            self.status_bar.showMessage("错误：未找到原始光谱线，无法进行高亮对比", 5000)
            return


        target_line = all_lines[-1]
        curr_x = target_line.get_xdata()
        curr_y = target_line.get_ydata()


        old_indices = np.linspace(0, 1, len(weights))
        new_indices = np.linspace(0, 1, len(curr_x))
        f_interp = interp1d(old_indices, weights, kind='linear', fill_value="extrapolate")
        weights_mapped = f_interp(new_indices)


        w_min, w_max = weights_mapped.min(), weights_mapped.max()
        weights_norm = (weights_mapped - w_min) / (w_max - w_min + 1e-8)


        random_color = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))


        threshold = 0.3

        try:
            fill = ax.fill_between(
                curr_x,
                0,
                curr_y,
                where=(weights_norm > threshold),
                color=random_color,
                alpha=0.4,
                label=f"PWM: {class_name}",
                interpolate=True
            )

            fill.is_pwm_fill = True
        except Exception as e:
            print(f"高亮绘制失败: {e}")


        ax.legend(loc='upper right', fontsize='small')
        if hasattr(plot_widget, 'canvas'):
            plot_widget.canvas.draw()

        self.status_bar.showMessage(f"已根据模型权重高亮显示 [{class_name}] 关键特征区", 5000)


    def clear_pwm_curves(self):

        plot_widget = self.plot_tab_widget.raman_plot_widget
        ax = plot_widget.canvas.axes


        for poly in ax.collections[:]:
            if hasattr(poly, 'is_pwm_fill') and poly.is_pwm_fill:
                poly.remove()


        for line in ax.get_lines()[:]:
            if "PWM" in line.get_label():
                line.remove()


        handles, labels = ax.get_legend_handles_labels()
        valid_indices = [i for i, label in enumerate(labels) if "PWM" not in label]

        if valid_indices:

            ax.legend([handles[i] for i in valid_indices], [labels[i] for i in valid_indices],
                      loc='upper right', fontsize='small')
        else:

            legend = ax.get_legend()
            if legend:
                legend.remove()


        plot_widget.canvas.draw()
        self.status_bar.showMessage("反向验证图层已成功清理", 3000)