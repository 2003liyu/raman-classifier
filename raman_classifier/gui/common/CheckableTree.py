from __future__ import annotations
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QWidget

from .CheckableTreeItem import CheckableTreeItem


class CheckableTreeIterator:

    def __init__(self, tree:CheckableTree):
        self.__index:int = 0
        self.__tree:CheckableTree = tree
        self.__len:int = tree.topLevelItemCount()

    def __next__(self)->str:
        if self.__index >= self.__len:
            raise StopIteration()
        
        result:str = self.__tree.topLevelItem(self.__index)
        self.__index += 1
        return result

class CheckableTree(QTreeWidget):

    itemsChanged = Signal(list)

    def __init__(self, parent:QWidget=None)->None:
        QTreeWidget.__init__(self, parent)
        self.itemChanged.connect(self.__on_item_changed)
        self.itemClicked.connect(self.__on_item_clicked)

    def __getitem__(self, name:str)->CheckableTreeItem:
        for i in range(self.topLevelItemCount()):
            item:CheckableTreeItem = self.topLevelItem(i)
            if item.text(0) == name:
                return item
            
        item = CheckableTreeItem(name)
        self.addTopLevelItem(item)
        return item
    
    def __setitem__(self, name:str, item:CheckableTreeItem)->None:
        for i in range(self.topLevelItemCount()):
            if self.topLevelItem(i).text(0) == name:
                self.takeTopLevelItem(i)
                self.insertTopLevelItem(i, item)
                return
                
        self.addTopLevelItem(item)

    def __delitem__(self, name:str)->None:
        for i in range(self.topLevelItemCount()):
            item:CheckableTreeItem = self.topLevelItem(i)
            if item.text(0) == name:
                self.takeTopLevelItem(i)
                return

    def __contains__(self, name:str)->bool:
        for i in range(self.topLevelItemCount()):
            item:CheckableTreeItem = self.topLevelItem(i)
            if item.text(0) == name:
                return True
            
        return False
    
    def __iter__(self)->CheckableTreeIterator:
        return CheckableTreeIterator(self)

    def __len__(self)->int:
        return self.topLevelItemCount()

    def _recursively_check_descendants(self, item:CheckableTreeItem, column:int, state:int)->List[CheckableTreeItem]:
        result = []
        for i in range(item.childCount()):
            if item.child(i).checkState(column) == state:
                continue

            result.append(item.child(i))
            item.child(i).setCheckState(column, state)
            result.extend(self._recursively_check_descendants(item.child(i), column, state))

        return result

    def _recursively_check_ancestors(self, item:CheckableTreeItem, column:int)->List[CheckableTreeItem]:
        result = []

        if item is None:
            return result
        
        all_checked:bool = True
        all_unchecked:bool = True
        for i in range(item.childCount()):
            if item.child(i).checkState(column) != Qt.CheckState.Checked:
                all_checked = False
            
            if item.child(i).checkState(column) != Qt.CheckState.Unchecked:
                all_unchecked = False

            if not all_checked and not all_unchecked:
                break

        target_state:Optional[Qt.CheckState] = None
        if all_checked:
            target_state = Qt.CheckState.Checked
        elif all_unchecked:
            target_state = Qt.CheckState.Unchecked
        else:
            target_state = Qt.CheckState.PartiallyChecked

        if item.checkState(column) == target_state:
            return result
        
        item.setCheckState(column, target_state)
        result.append(item)
        result.extend(self._recursively_check_ancestors(item.parent(), column))
        return result

    def __on_item_changed(self, item:CheckableTreeItem, column:int)->None:
        changed_items = []
        old_block_signals = self.signalsBlocked()
        self.blockSignals(True)
        changed_items.extend(self._recursively_check_descendants(item, column, item.checkState(column)))
        changed_items.extend(self._recursively_check_ancestors(item.parent(), column))
        self.blockSignals(old_block_signals)
        if changed_items:
            self.itemsChanged.emit(changed_items)

        self.setCurrentItem(item)

    def __on_item_clicked(self, item:CheckableTreeItem):
        if self.currentItem() is item:
            self.currentItemChanged.emit(item, item)