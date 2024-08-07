import transformers
import torch

from loader import DataLoader, JsonLoader


data_loader = DataLoader(JsonLoader, "json")
root_dir = "/kilab/data/"
modu_dir = "modu/NIKL_SBSC_2023_v1.0"
data_dir_list = data_loader.get_listdir(root_dir, modu_dir)

json_lst = list(data_loader.load(data_dir_list))

src_lst, sum_lst = [], []

for json_doc in json_lst:
    for doc in json_doc['document']:
        src = ''
        for sent in doc['sentence']:
            src += sent['form']
        
        src_lst.append(src)
        sum_lst.append(doc['SC']['main_summary'])

model_id = "rtzr/ko-gemma-2-9b-it"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


pipeline.model.eval()
instruction = "서울의 유명한 관광 코스를 만들어줄래?"

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

print(outputs[0]["generated_text"][len(prompt):])

