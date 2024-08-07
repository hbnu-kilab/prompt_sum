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
        data_dir = Path(root_dir) / data_dir
        return os.listdir(data_dir)


class JsonLoader(DataLoaderInterface):
    def __init__(self, *args):
        assert args[0] in ["json", "jsonl"], print(f"Please input among json, jsonl.")

        self.file_ext = args[0]
        if args[0] == "json":
            self.flag_line = False
        elif args[0] == "jsonl":
            self.flag_line = True

    def load(self, file_path, **kwargs):
        if self.file_ext == "jsonl":
            return self.load_jsonl(file_path)
        elif self.file_ext == "json":
            if type(file_path) == list:
                yield self.load_json_in_dir(file_path)
            else:
                return self.load_json(file_path)

    def load_json_in_dir(self, file_path_lst):
        for file_path in file_path_lst:
            with open(file_path, 'r') as file:
                try:
                    qa_lst = json.load(file)
                except:
                    lines = file.read()
                    qa_lst = json.loads(lines)
                yield qa_lst

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            try:
                qa_lst = json.load(file)
            except:
                lines = file.read()
                qa_lst = json.loads(lines)

        return qa_lst

    def load_jsonl(self, file_path):
        qa_lst = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                qa_lst.append(json.loads(line))
        return qa_lst
