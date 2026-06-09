import os
import struct
import hashlib
import pickle
from typing import Any, Tuple, Union
from charset_normalizer import from_path

import numpy as np


def modify_time(file_path:str)->float:
    if not os.path.isfile(file_path):
        return 0
    
    return os.path.getmtime(file_path)

def md5sum(data_or_path:Union[str, np.ndarray, bytes, bytearray])->str:
    md5_hash = hashlib.md5()

    if isinstance(data_or_path, str):
        chunk_size:int = 4096
        
        with open(data_or_path, "rb") as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break

                md5_hash.update(chunk)
    elif isinstance(data_or_path, (bytes, bytearray)):
        md5_hash.update(data_or_path)

    elif isinstance(data_or_path, np.ndarray):
        md5_hash.update(data_or_path.tobytes())
    
    return md5_hash.hexdigest()

def md5sums(content:str)->str:
    return md5sum(content.encode("utf-8"))

def cat(file_name: str) -> str:
    return str(from_path(file_name).best())
    
def echo(content:str, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "w", encoding='utf-8') as file:
        file.write(content)

def get_npy_shape(npy_file_path:str)->Tuple[int, int]:
    with open(npy_file_path, 'rb') as f:
        magic = f.read(6)
        if magic != b'\x93NUMPY':
            raise ValueError(f"{npy_file_path} is not a valid npy file")
        
        version = struct.unpack('<H', f.read(2))[0]
        if version not in (1, 2):
            raise ValueError(f"not support npy file version {version}")
        
        if version == 1:
            header_len = struct.unpack('<I', f.read(4))[0]
        else:  # version == 2
            header_len = struct.unpack('<Q', f.read(8))[0]
        
        header = f.read(header_len).decode('latin1')
        try:
            from numpy.lib.format import _parse_header
            header_dict = _parse_header(header)
            return header_dict['shape']
        except:
            import re
            shape_match = re.search(r"'shape'\s*:\s*\(([^)]+)\)", header)
            if shape_match:
                shape_str = shape_match.group(1)
                shape = tuple(map(int, shape_str.split(',')))
                return shape
            else:
                raise RuntimeError("failed to analyse npy header")
            
def save_var(var:Any, file_name:str)->None:
    target_folder:str = os.path.dirname(os.path.abspath(file_name))
    if not os.path.isdir(target_folder) or not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(file_name, "wb") as file:
        pickle.dump(var, file)

def load_var(file_name:str)->Any:
    with open(file_name, "rb") as file:
        return pickle.load(file)
