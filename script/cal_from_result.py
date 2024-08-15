import evaluate
from tqdm import tqdm
metric = evaluate.combine(["bleu", "rouge", "meteor"])

file_path = "result_gpt4o-mini"

with open(file_path, 'r') as f:
    lines = f.readlines()
    sum_lst, output_sum_lst = [], []
    for line in tqdm(lines, total=len(lines)):
        if "Output summary: " in line[:13]:
            output_sum_lst.append(line)
        elif "Gold Output summary:" in line:
            sum_lst.append(line)
    
metric.add_batch(predictions=output_sum_lst, references=sum_lst)
eval_metric = metric.compute()
print({
    "FINAL // bleu": eval_metric["bleu"]*100,
    "eval_rouge1": eval_metric["rouge1"]*100,
    "eval_rouge2": eval_metric["rouge2"]*100,
    "eval_rougeL": eval_metric["rougeL"]*100,
    "eval_rougeLsum": eval_metric["rougeLsum"]*100,
    "meteor": eval_metric["meteor"]*100,
})