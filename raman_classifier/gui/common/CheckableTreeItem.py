from __future__ import annotations

from typing import List, Union, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidgetItem


class CheckableTreeItemIterator:

    def __init__(self, item:CheckableTreeItem):
        self.__item:CheckableTreeItem = item
        self.__index:int = 0

    def __next__(self)->str:
        if self.__index >= self.__item.childCount():
            raise StopIteration()
        
        result:str = self.__item.child(self.__index)
        self.__index += 1
        return result

class CheckableTreeItem(QTreeWidgetItem):

    def __init__(self, names:Union[str, List[str]]):
        if isinstance(names, str):
            names = [names]

        QTreeWidgetItem.__init__(self, names)

        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(0, Qt.CheckState.Unchecked)

        self.__user_data:Dict[Any, Any] = {}

    def __getitem__(self, name:str)->CheckableTreeItem:
        for i in range(self.childCount()):
            child:CheckableTreeItem = self.child(i)
            if child.text(0) == name:
                return child
            
        child:CheckableTreeItem = CheckableTreeItem(name)
        self.addChild(child)
        return child
    
    def __setitem__(self, name:str, item:CheckableTreeItem)->None:
        for i in range(self.childCount()):
            child:CheckableTreeItem = self.child(i)
            if child.text(0) == name:
                self.removeChild(child)
                self.insertChild(i, item)
                return
            
        self.addChild(item)

    def removeChild(self, child):
        QTreeWidgetItem.removeChild(self, child)
        for column in range(self.columnCount()):
            self.treeWidget()._recursively_check_ancestors(self, column)

    def __delitem__(self, name:str)->None:
        for i in range(self.childCount()):
            child:CheckableTreeItem = self.child(i)
            if child.text(0) == name:
                self.removeChild(child)
                return
            
    def __len__(self)->int:
        return self.childCount()
    
    def __contains__(self, name:str)->bool:
        for i in range(self.childCount()):
            child:CheckableTreeItem = self.child(i)
            if child.text(0) == name:
                return True
            
        return False
    
    def __iter__(self)->CheckableTreeItemIterator:
        return CheckableTreeItemIterator(self)
    
    @property
    def userData(self)->Dict[Any, Any]:
        return self.__user_data