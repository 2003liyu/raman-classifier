from __future__ import annotations

from typing import TYPE_CHECKING

from .DataTree import DataTree

if TYPE_CHECKING:
    from .MainWindow import MainWindow


class ProtoSupportDataTree(DataTree):

    def __init__(self, main_window:MainWindow)->None:
        DataTree.__init__(self, main_window, "proto_support", "label", "支撑数据")
        self.setHeaderHidden(True)
        self.reload()
