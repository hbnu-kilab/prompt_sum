import transformers
import torch
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
    data_dir = "aihub/news/news_valid_original"
    data_loader = DataLoader(JsonLoader, "json")
    sum_loader = SummaryLoader(SummaryAIHubNewsLoader)
    data_dir_list = data_loader.get_listdir(ROOT_DIR, data_dir)
    json_obj = data_loader.load(data_dir_list)
    src_lst, sum_lst = sum_loader.load(json_obj)


model_id = "rtzr/ko-gemma-2-9b-it"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()


def do_llm(instruction):
    messages = [
        {"role": "user", "content": f"{instruction}"}
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    return outputs, outputs[0]["generated_text"][len(prompt):]

output_sum_lst = []
for i, (src, sum) in enumerate(zip(src_lst, sum_lst)):
    prev_gold_sum = sum_lst[i-1]
    instruction = mk_inst_for_summary(src, prev_gold_sum)
    outputs, output_sum = do_llm(instruction)
    output_sum_lst.append(output_sum)

    output_sum = output_sum.split("[예제 요약]")[0].replace('\n', ' ')
    rouge_scores, rouge = eval.rouge(output_sum, sum)
    bleu_scores = eval.bleu(output_sum, sum)
    print(f"Rouge scores:\n {rouge_scores}\nRouge: {rouge}")
    print(f"BLEU scores:\n {bleu_scores}")
    print(output_sum)

