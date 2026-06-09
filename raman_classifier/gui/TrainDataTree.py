from __future__ import annotations

from typing import TYPE_CHECKING

from .DataTree import DataTree

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class TrainDataTree(DataTree):

    def __init__(self, main_window:MainWindow)->None:
        DataTree.__init__(self, main_window, "train", "label", "训练数据")
        self.setHeaderHidden(True)
        self.reload()