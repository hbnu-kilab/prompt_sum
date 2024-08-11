import transformers
import torch
from tqdm import tqdm
from loader import DataLoader, JsonLoader, JsonInDirLoader
from eval import eval
from mk_instruction import *

data_loader = DataLoader(JsonInDirLoader, "json")
root_dir = "/kilab/data/"
modu_dir = "modu/NIKL_SBSC_2023_v1.0"
data_dir_list = data_loader.get_listdir(root_dir, modu_dir)

json_lst = list(data_loader.load(data_dir_list))

src_lst, sum_lst = [], []


for json_doc in tqdm(json_lst, total=len(json_lst), desc="load json"):
    for doc in json_doc['document']:
        for issue_sum in doc['SC']['issue_summary']:
            topic = issue_sum['issue']['topic']
            ab_summary = issue_sum['summary']['abstract']['form']
            ref_lst = issue_sum['summary']['abstract']['reference']

            src_sents = []
            for ref_id in ref_lst:
                for sent in doc['sentence']:
                    if ref_id == sent['id']:
                        src_sents.append(sent['form'])
                        break

            src_lst.append(' '.join(src_sents))
            sum_lst.append(ab_summary)
        
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

    rouge = eval.rouge(output_sum, sum)
    print(output_sum)

