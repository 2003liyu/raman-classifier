from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .DataManager import DataManager


class ProgressStage:

    def __init__(self, name:str, total_progress:float, data_manager:DataManager):
        self.__data_manager:DataManager = data_manager
        self.__name:str = name
        self.__total_progress:float = total_progress
        self.__progress:float = 0

    @property
    def progress(self)->float:
        return self.__progress
    
    @progress.setter
    def progress(self, progress:float)->None:
        self.__progress = progress
        self.__data_manager.progress = self.__start_progress + self.__total_progress * progress

    def __enter__(self)->ProgressStage:
        self.__start_progress:float = self.__data_manager.progress
        self.__data_manager.progress_stage = self.__name
        return self

    def __exit__(self, exc_type, exc_value, traceback):
         self.__progress = 1
         self.__data_manager.progress = self.__start_progress + self.__total_progress