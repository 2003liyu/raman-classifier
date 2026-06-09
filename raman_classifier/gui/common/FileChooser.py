import os
from typing import List

from PySide6.QtWidgets import QFileDialog, QWidget

from ...common.utils import cat, echo


class FileChooser:

    def __init__(self, parent:QWidget=None, history_file_name:str="file_chooser_history.txt"):
        self.__parent:QWidget = parent
        self.__history_file_name:str = os.path.abspath(history_file_name).replace("\\", "/")
        self.__last_folder:str = ""

        if os.path.isfile(history_file_name):
            self.__last_folder:str = cat(history_file_name)
            if not os.path.isdir(self.__last_folder):
                self.__last_folder:str = ""

    def openFile(self, caption:str="打开文件", dir:str="", filter:str="", selected_filter:str="", options:QFileDialog.Option=QFileDialog.Option())->str:
        if not dir:
            dir = self.__last_folder

        file_path = QFileDialog.getOpenFileName(self.__parent, caption, dir, filter, selected_filter, options)[0]
        if file_path:
            file_path = os.path.abspath(file_path).replace("\\", "/")
            self.__last_folder = os.path.dirname(file_path)
            echo(self.__last_folder, self.__history_file_name)

        return file_path
    
    def openFiles(self, caption:str="打开文件", dir:str="", filter:str="", selected_filter:str="", options:QFileDialog.Option=QFileDialog.Option())->List[str]:
        if not dir:
            dir = self.__last_folder

        file_paths = QFileDialog.getOpenFileNames(self.__parent, caption, dir, filter, selected_filter, options)[0]
        if file_paths:
            file_paths = [os.path.abspath(file_path).replace("\\", "/") for file_path in file_paths]
            self.__last_folder = os.path.dirname(file_paths[0])
            echo(self.__last_folder, self.__history_file_name)

        return file_paths
    
    def openFolder(self, caption:str="选择文件夹", dir:str="", options:QFileDialog.Option=QFileDialog.Option.ShowDirsOnly)->str:
        if not dir:
            dir = self.__last_folder

        folder_path = QFileDialog.getExistingDirectory(self.__parent, caption, dir, options)
        if folder_path:
            folder_path = os.path.abspath(folder_path).replace("\\", "/")
            self.__last_folder = os.path.dirname(folder_path)
            echo(self.__last_folder, self.__history_file_name)

        return folder_path
    
    def saveFile(self, caption:str="保存文件", dir:str="", filter:str="", selected_filter:str="", options:QFileDialog.Option=QFileDialog.Option())->str:
        if not dir:
            dir = self.__last_folder

        file_path = QFileDialog.getSaveFileName(self.__parent, caption, dir, filter, selected_filter, options)[0]
        if file_path:
            file_path = os.path.abspath(file_path).replace("\\", "/")
            self.__last_folder = os.path.dirname(file_path)
            echo(self.__last_folder, self.__history_file_name)

        return file_path

    @property
    def last_folder(self)->str:
        return self.__last_folder