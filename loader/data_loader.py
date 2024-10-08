import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from collections import defaultdict

from .data_loader_interface import DataLoaderInterface


class DataLoader(DataLoaderInterface):
    def __init__(self, 
                 file_system: DataLoaderInterface,
                 *args):
        self.file_system = file_system(*args)

    def load(self, file_path, **kwargs):
        return self.file_system.load(file_path, **kwargs)

    def get_listdir(self, root_dir, data_dir):
        data_file_dir = Path(root_dir) / data_dir
        return [str(data_file_dir / data_file) for data_file in os.listdir(data_file_dir)]


class JsonlLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path, **kwargs):
        data_lst = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                data_lst.append(json.loads(line))
        return data_lst

class JsonInDirLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path_lst, **kwargs):
        for file_path in file_path_lst:
            with open(file_path, 'r') as file:
                try:
                    data_lst = json.load(file)
                except:
                    lines = file.read()
                    data_lst = json.loads(lines)
            yield data_lst

class JsonLoader(DataLoaderInterface):
    def __init__(self, *args):
        self.args = args

    def load(self, file_path, **kwargs):
        with open(file_path, 'r') as file:
            try:
                data_lst = json.load(file)
            except:
                lines = file.read()
                data_lst = json.loads(lines)

        return data_lst


class JsonLoader(DataLoaderInterface):
    def __init__(self, *args):
        assert args[0] in ["json", "jsonl"], print(f"Please input among json, jsonl.")

    def load(self, file_path, **kwargs):
        with open(file_path, 'r') as file:
            try:
                qa_lst = json.load(file)
            except:
                lines = file.read()
                qa_lst = json.loads(lines)

        return qa_lst


