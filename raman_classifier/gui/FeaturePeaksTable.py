from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QTableWidget, QHeaderView, QTableWidgetItem

import numpy as np


class FeaturePeaksTable(QTableWidget):
    
    def __init__(self, parent:QWidget=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["峰位\n(cm⁻¹)", "峰强", "峰面积", "半高宽\n(cm⁻¹)"])
        self.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setMinimumWidth(250)

    def clear(self):
        self.clearContents()

    def set_peaks_info(self, peaks_amp:np.ndarray, peaks_mu:np.ndarray, peaks_sigma:np.ndarray):
        peaks_amp = peaks_amp.read()
        max_amp = peaks_amp.max()
        self.setRowCount(len(peaks_amp))
        for i in range(len(peaks_amp)):
            # 峰位
            mu = 1000 * peaks_mu[i]
            mu_item = QTableWidgetItem(f"{mu:.2f}")
            mu_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(i, 0, mu_item)

            # 峰强
            amp = peaks_amp[i] / max_amp
            amp_item = QTableWidgetItem(f"{amp:.2f}")
            amp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(i, 1, amp_item)

            # 峰面积
            sigma = 1000 * peaks_sigma[i]
            area = np.sqrt(2 * np.pi) * amp * sigma
            area_item = QTableWidgetItem(f"{area:.2f}")
            area_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(i, 2, area_item)

            # 半高宽
            half_height_width = 2* np.sqrt(2 * np.log(2)) * sigma
            half_height_width_item = QTableWidgetItem(f"{half_height_width:.2f}")
            half_height_width_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(i, 3, half_height_width_item)