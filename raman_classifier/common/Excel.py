import os
from typing import Optional, Union, Tuple

import openpyxl


class Excel:

    def __init__(self, file_name:str=""):
        if os.path.isfile(file_name):
            self.workbook = openpyxl.load_workbook(file_name)
            self.file_name = os.path.abspath(file_name).replace("\\", "/")
        else:
            self.workbook = openpyxl.Workbook()
            self.file_name = ""

        self.active_sheet = self.workbook.active
        self.active_sheet_index:int = 0
        self.sheet_names = self.workbook.sheetnames
        self.active_sheet_name:str = self.sheet_names[0]

    def sheet(self, index:Union[int,str])->None:
        if isinstance(index, int):
            self.active_sheet = self.workbook[self.sheet_names[index]]
            self.active_sheet_name = self.sheet_names[index]
        elif isinstance(index, str):
            self.active_sheet = self.workbook[index]
            self.active_sheet_name = index

    def delete_cols(self, col:int, cout:int=1)->None:
        self.workbook.delete_cols(col, cout)

    @property
    def sheet_name(self)->str:
        return self.active_sheet.title
    
    @sheet_name.setter
    def sheet_name(self, name:str)->None:
        self.active_sheet.title = name

    def save(self, file_name:Optional[str]=None)->None:
        if file_name is None:
            file_name = self.file_name

        folder_path = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.workbook.save(file_name)

    def __getitem__(self, index:Tuple[int, int]):
        row = index[0]
        column = index[1]
        return self.active_sheet.cell(row=row+1, column=column+1)
    
    def __setitem__(self, index:Tuple[int, int], value)->None:
        row = index[0]
        column = index[1]
        self.active_sheet.cell(row=row+1, column=column+1, value=value)

    def is_empty(self, row:int, col:int)->bool:
        value = self[row, col].value
        return value is None or (isinstance(value, str) and value == '')

    @property
    def rows(self)->int:
        return self.active_sheet.max_row
    
    @property
    def cols(self)->int:
        return self.active_sheet.max_column