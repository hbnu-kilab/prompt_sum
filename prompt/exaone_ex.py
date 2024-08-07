import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..loader import DataLoader, JsonLoader

model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

data_loader = DataLoader(JsonLoader)
root_dir = "/kilab/data/"
modu_dir = "modu/NIKL_SBSC_2023_v1.0"
data_dir_list = data_loader.get_listdir(root_dir, modu_dir)

json_lst = []
for data_dir in data_dir_list:
    json_lst += [data_loader.load(data_dir)]

# Choose your prompt
prompt = "Explain who you are"  # English example
prompt = "너의 소원을 말해봐"   # Korean example

messages = [
    {"role": "system", 
     "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": prompt}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

output = model.generate(
    input_ids.to("cuda"),
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128
)
print(tokenizer.decode(output[0]))
