import subprocess
from typing import Tuple

from PySide6.QtCore import QObject, Signal


class NVGPUMonitor(QObject):

    nvgpu_info_changed = Signal(int, 'qint64', 'qint64')

    def get_nvgpu_info(self)->Tuple[int, int, int]:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ],
            capture_output=True,
            text=True,
            check=True
        )

        output = result.stdout.strip()
        parts = output.split(',')
        gpu_util = int(parts[0].strip())
        used_mib = int(parts[1].strip())
        total_mib = int(parts[2].strip())

        used_bytes = used_mib * 1024 * 1024
        total_bytes = total_mib * 1024 * 1024

        self.nvgpu_info_changed.emit(gpu_util, used_bytes, total_bytes)
        return gpu_util, used_bytes, total_bytes
