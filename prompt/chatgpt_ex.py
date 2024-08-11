import transformers
import torch
from pathlib import Path
from tqdm import tqdm
from loader import DataLoader, JsonLoader, JsonInDirLoader, SummaryLoader, SummarySBSCLoader, SummarySDSCLoader, SummaryAIHubNewsLoader

from eval import eval
from mk_instruction import *

ROOT_DIR = "/kilab/data/"

data_type = "news"

if data_type == "SBSC":
    # SBSC data
    data_dir = "modu/NIKL_SBSC_2023_v1.0"
    data_loader = DataLoader(JsonInDirLoader, "json")
    sum_loader = SummaryLoader(SummarySBSCLoader)
    data_dir_list = data_loader.get_listdir(ROOT_DIR, data_dir)
    json_lst = list(data_loader.load(data_dir_list))
    src_lst, sum_lst = sum_loader.load(json_lst)
elif data_type == "news":
    # News data
    data_dir = "aihub/summarization/news/news_valid_original.json"
    data_loader = DataLoader(JsonLoader, "json")
    sum_loader = SummaryLoader(SummaryAIHubNewsLoader)
    json_obj = data_loader.load(Path(ROOT_DIR) / data_dir)
    src_lst, sum_lst = sum_loader.load(json_obj)


model_id = "rtzr/ko-gemma-2-9b-it"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)