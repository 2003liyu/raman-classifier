from PySide6.QtCore import QTimerEvent, QThread, Signal
from PySide6.QtWidgets import QStatusBar, QWidget, QHBoxLayout, QLabel

import psutil

from .common.NVGPUMonitor import NVGPUMonitor


class StatusBar(QStatusBar):

    _get_nvgpu_info = Signal()

    def __init__(self, parent:QWidget=None):
        QStatusBar.__init__(self, parent)

        self.nvgpu_monitor_thread = QThread()
        self.nvgpu_monitor_thread.setObjectName("Nvidia GPU Monitor Thread")
        self.nvgpu_monitor_thread.start()

        self.nvgpu_monitor = NVGPUMonitor()
        self.nvgpu_monitor.moveToThread(self.nvgpu_monitor_thread)

        self.monitor_widget = QWidget()
        self.monitor_layout = QHBoxLayout(self.monitor_widget)
        self.monitor_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_layout.setSpacing(10)

        self.cpu_label = QLabel("", self)
        self.mem_label = QLabel("", self)
        self.gpu_label = QLabel("", self)
        self.gpu_mem_label = QLabel("", self)

        self.monitor_layout.addStretch()
        self.monitor_layout.addWidget(self.cpu_label)
        self.monitor_layout.addWidget(self.mem_label)
        self.monitor_layout.addWidget(self.gpu_label)
        self.monitor_layout.addWidget(self.gpu_mem_label)
        self.monitor_layout.addSpacing(10)

        self.addPermanentWidget(self.monitor_widget)

        self.__has_nvgpu:bool = False
        try:
            self.nvgpu_monitor.get_nvgpu_info()
            self._get_nvgpu_info.connect(self.nvgpu_monitor.get_nvgpu_info)
            self.nvgpu_monitor.nvgpu_info_changed.connect(self.on_nvgpu_info_changed)
            self.__has_nvgpu:bool = True
        except:
            self.gpu_label.hide()
            self.gpu_mem_label.hide()

        self.refresh()
        self.__timer_id:int = self.startTimer(1000)

    def close(self):
        if self.nvgpu_monitor_thread.isRunning():
            self.nvgpu_monitor_thread.quit()
            self.nvgpu_monitor_thread.wait()

    def refresh(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        mem_total = mem.total
        mem_used = mem.total - mem.available
        self.cpu_label.setText(f"CPU 使用率: {cpu:.1f}%")
        self.mem_label.setText(f"内存已用: {self.storage_text(mem_used)}/{self.storage_text(mem_total)}={(mem_used/mem_total*100):.1f}%")

        if self.__has_nvgpu:
            self._get_nvgpu_info.emit()

    def on_nvgpu_info_changed(self, gpu_util:int, used_bytes:int, total_bytes:int):
        self.gpu_label.setText(f"GPU 使用率: {gpu_util:.1f}%")
        self.gpu_mem_label.setText(f"显存已用: {self.storage_text(used_bytes)}/{self.storage_text(total_bytes)}={(used_bytes/total_bytes*100):.1f}%")

    def timerEvent(self, timer_event:QTimerEvent):
        if timer_event.timerId() != self.__timer_id:
            return
        
        self.refresh()

    @staticmethod
    def storage_text(storage:int)->str:
        KB = 1024
        MB = 1024 * KB
        GB = 1024 * MB
        TB = 1024 * GB

        if storage > 0.1 * TB:
            return f"{(storage / TB):.1f}TB"
        elif storage > 0.1 * GB:
            return f"{(storage / GB):.1f}GB"
        elif storage > 0.1 * MB:
            return f"{(storage / MB):.1f}MB"
        elif storage > 0.1 * KB:
            return f"{(storage / KB):.1f}KB"
        else:
            return f"{storage}B"
