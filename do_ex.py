from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonLoader, JsonInDirLoader, SummaryLoader, SummarySBSCLoader, SummarySDSCLoader, SummaryAIHubNewsLoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor
from promptor.mk_instruction import mk_inst_for_summary

from eval import eval

ROOT_DIR = "/kilab/data/"

data_type = "news"
model_type = "exaone"

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

if model_type == "gemma2":
    promptor = Promptor(Gemma2Promptor)
elif model_type == "exaone":
    promptor = Promptor(ExaonePromptor)

output_sum_lst = []
for i, (src, sum) in tqdm(enumerate(zip(src_lst, sum_lst)), total=len(src_lst)):
    prev_gold_sum = sum_lst[i-1]
    instruction = mk_inst_for_summary(src, prev_gold_sum)
    output = promptor.do_llm(instruction)
    output_sum_lst.append(output)

    output_sum = output.split("[예제 요약]")[0].replace('\n', ' ')
    rouge_scores, rouge = eval.rouge(output_sum, sum)
    bleu_scores = eval.bleu(output_sum, sum)
    print(f"Rouge scores:\n {rouge_scores}\nRouge: {rouge}")
    print(f"BLEU scores:\n {bleu_scores}")
    
    print(f"Input text: {instruction}")
    print(f"Output summary: {output_sum}")
    print(f"Gold Output summary: {sum}")

