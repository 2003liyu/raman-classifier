from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QCheckBox

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class PlotOptionsWidget(QGroupBox):

    line_type_chagned = Signal(str)
    show_grid_changed = Signal(bool)

    def __init__(self, main_window:MainWindow)->None:
        QGroupBox.__init__(self, "绘图选项")
        self.main_window:MainWindow = main_window
        vlayout_main = QVBoxLayout()

        self.clip_checkbox = QCheckBox("裁剪", self)
        self.remove_cosmic_rays_checkbox = QCheckBox("去除宇宙射线", self)
        self.baseline_checkbox = QCheckBox("基线", self)
        self.remove_baseline_checkbox = QCheckBox("基线校正", self)
        self.filter_checkbox = QCheckBox("滤波", self)
        self.normalize_checkbox = QCheckBox("归一化", self)
        self.pca_reconstruct_checkbox = QCheckBox("PCA 重建", self)
        self.gaussian_fit_checkbox = QCheckBox("高斯拟合", self)
        self.show_grid_checkbox = QCheckBox("显示网格", self)
        self.show_grid_checkbox.setChecked(True)
        
        vlayout_main.addWidget(self.clip_checkbox)
        vlayout_main.addWidget(self.remove_cosmic_rays_checkbox)
        vlayout_main.addWidget(self.baseline_checkbox)
        vlayout_main.addWidget(self.remove_baseline_checkbox)
        vlayout_main.addWidget(self.filter_checkbox)
        vlayout_main.addWidget(self.normalize_checkbox)
        vlayout_main.addWidget(self.pca_reconstruct_checkbox)
        vlayout_main.addWidget(self.gaussian_fit_checkbox)
        vlayout_main.addWidget(self.show_grid_checkbox)

        self.setLayout(vlayout_main)

        self.clip_checkbox.stateChanged.connect(self.on_clip_changed)
        self.remove_cosmic_rays_checkbox.stateChanged.connect(self.on_remove_cosmic_rays_changed)
        self.baseline_checkbox.stateChanged.connect(self.on_baseline_changed)
        self.remove_baseline_checkbox.stateChanged.connect(self.on_remove_baseline_changed)
        self.filter_checkbox.stateChanged.connect(self.on_filter_changed)
        self.normalize_checkbox.stateChanged.connect(self.on_normalize_changed)
        self.pca_reconstruct_checkbox.stateChanged.connect(self.on_pca_reconstruct_changed)
        self.gaussian_fit_checkbox.stateChanged.connect(self.on_gaussian_fit_changed)
        self.show_grid_checkbox.stateChanged.connect(self.show_grid_changed.emit)

    @property
    def clipped(self)->bool:
        return self.clip_checkbox.isChecked()

    @property
    def cosmic_rays_removed(self)->bool:
        return self.remove_cosmic_rays_checkbox.isChecked()

    @property
    def baseline(self)->bool:
        return self.baseline_checkbox.isChecked()

    @property
    def baseline_removed(self)->bool:
        return self.remove_baseline_checkbox.isChecked()

    @property
    def filtered(self)->bool:
        return self.filter_checkbox.isChecked()
    
    @property
    def normalized(self)->bool:
        return self.normalize_checkbox.isChecked()
    
    @property
    def pca_reconstructed(self)->bool:
        return self.pca_reconstruct_checkbox.isChecked()

    @property
    def show_grid(self)->bool:
        return self.show_grid_checkbox.isChecked()
    
    @property
    def gaussian_fitted(self)->bool:
        return self.gaussian_fit_checkbox.isChecked()
    
    @property
    def line_type(self)->str:
        if self.gaussian_fitted:
            return "line_gaussian_fitted_data"

        if self.pca_reconstructed:
            if self.main_window.is_proto:
                return "line_proto_pca_reconstructed_data"
            else:
                return "line_pca_reconstructed_data"

        if self.normalized:
            return "line_normalized_data"
        
        if self.filtered:
            return "line_filtered_data"
        
        if self.baseline_removed:
            return "line_baseline_removed_data"
        
        if self.baseline:
            return "line_baseline_data"

        if self.cosmic_rays_removed:
            return "line_cosmic_rays_removed_data"
        
        if self.clipped:
            return "line_clip_raw_data"
        
        return "line_full_raw_data"
    
    def on_clip_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Unchecked:
            self.remove_cosmic_rays_checkbox.setChecked(False)
            self.baseline_checkbox.setChecked(False)
            self.remove_baseline_checkbox.setChecked(False)
            self.filter_checkbox.setChecked(False)
            self.normalize_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_remove_cosmic_rays_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
        else:
            self.baseline_checkbox.setChecked(False)
            self.remove_baseline_checkbox.setChecked(False)
            self.filter_checkbox.setChecked(False)
            self.normalize_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_baseline_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)
        old_block_signals = self.signalsBlocked()

        self.blockSignals(True)

        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.remove_baseline_checkbox.setChecked(False)
            self.filter_checkbox.setChecked(False)
            self.normalize_checkbox.setChecked(False)
            self.pca_reconstruct_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)

        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_remove_baseline_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)
        old_block_signals = self.signalsBlocked()

        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.baseline_checkbox.setChecked(False)
        else:
            self.filter_checkbox.setChecked(False)
            self.normalize_checkbox.setChecked(False)
            self.pca_reconstruct_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_filter_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.remove_baseline_checkbox.setChecked(True)
            self.baseline_checkbox.setChecked(False)
        else:
            self.normalize_checkbox.setChecked(False)
            self.pca_reconstruct_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_normalize_changed(self, check_state:int):
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.remove_baseline_checkbox.setChecked(True)
            self.filter_checkbox.setChecked(True)
            self.baseline_checkbox.setChecked(False)
        else:
            self.pca_reconstruct_checkbox.setChecked(False)
            self.gaussian_fit_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)
        
        self.line_type_chagned.emit(self.line_type)

    def on_pca_reconstruct_changed(self, check_state:int)->None:
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.remove_baseline_checkbox.setChecked(True)
            self.filter_checkbox.setChecked(True)
            self.gaussian_fit_checkbox.setChecked(False)
            self.normalize_checkbox.setChecked(True)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)

    def on_gaussian_fit_changed(self, check_state:bool)->None:
        check_state = Qt.CheckState(check_state)

        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        if check_state == Qt.CheckState.Checked:
            self.clip_checkbox.setChecked(True)
            self.remove_cosmic_rays_checkbox.setChecked(True)
            self.remove_baseline_checkbox.setChecked(True)
            self.filter_checkbox.setChecked(True)
            self.normalize_checkbox.setChecked(True)
            self.pca_reconstruct_checkbox.setChecked(False)
        self.blockSignals(old_block_signals)

        self.line_type_chagned.emit(self.line_type)