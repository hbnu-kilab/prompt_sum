from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonLoader, JsonInDirLoader, SummaryLoader, SummarySBSCLoader, SummarySDSCLoader, SummaryAIHubNewsLoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor
from promptor.mk_instruction import mk_inst_for_summary, mk_inst_for_summary_w_1shot, mk_inst_for_counterfactual_summary, mk_inst_for_summary_w_cda, mk_inst_for_counterfactual_summary_en, mk_inst_for_summary_w_cda_en

from transformers import AutoTokenizer
from eval import eval
from eval.clean_text import postprocess_text, clean_data_ko
import evaluate

metric = evaluate.combine(["bleu", "rouge", "meteor"])
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

ROOT_DIR = "/kilab/data/"

data_type = "law"
model_type = "gpt-4o-mini"
do_cda = False
"""
rtzr/ko-gemma-2-9b-it
carrotter/ko-gemma-2b-it-sft
"""
nshot = 0

def init_data():
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

        src_lst = src_lst[:len(src_lst)//5]
        sum_lst = sum_lst[:len(sum_lst)//5]
    elif data_type == "law":
        data_dir = "aihub/summarization/law/valid_original.json"
        data_loader = DataLoader(JsonLoader, "json")
        sum_loader = SummaryLoader(SummaryAIHubNewsLoader)
        json_obj = data_loader.load(Path(ROOT_DIR) / data_dir)
        src_lst, sum_lst = sum_loader.load(json_obj)

    return src_lst, sum_lst

def init_model():
    if model_type == "gemma2":
        model_id = "carrotter/ko-gemma-2b-it-sft"
        # model_id = "rtzr/ko-gemma-2-9b-it"
        promptor = Promptor(Gemma2Promptor, model_id)
    elif model_type == "exaone":
        model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        promptor = Promptor(ExaonePromptor, model_id)
    elif model_type in ["gpt-4o-mini", "gpt-4-turbo"]:
        model_id = model_type
        promptor = Promptor(ChatGPTPromptor, model_id)

    return promptor

def baseline(model_type, src_lst, sum_lst, metric, promptor):
    with open(f"./result/pred_{model_type}", 'w') as pf, open(f"./result/gold_{model_type}", 'w') as gf:
        tokenized_output_sum_lst, tokenized_sum_lst = [], []

        for i, (src, sum) in tqdm(enumerate(zip(src_lst, sum_lst)), total=len(src_lst)):
            prev_gold_sum = sum_lst[i-1]
            if nshot == 0:
                instruction = mk_inst_for_summary(src)
            elif nshot == 1:
                instruction = mk_inst_for_summary_w_1shot(src, prev_gold_sum)
            
            output_sum = promptor.do_llm(instruction)

            if nshot == 0:
                output_sum = output_sum.split("[요약]")[-1].replace('\n', ' ')
            elif nshot == 1:
                output_sum = output_sum.split("[예제 요약]")[-1].replace('\n', ' ')

            output_sum = clean_data_ko(output_sum)
            output_sum, sum = postprocess_text(output_sum, sum)

            tokenized_output_sum = ' '.join(tokenizer.tokenize(output_sum))
            tokenized_sum = ' '.join(tokenizer.tokenize(sum))

            tokenized_output_sum_lst.append(tokenized_output_sum)
            tokenized_sum_lst.append(tokenized_sum)
            
            metric.add_batch(predictions=[tokenized_output_sum], references=[tokenized_sum])
            try:
                eval_metric = metric.compute()
            except ZeroDivisionError as e:
                print("Error: Cannot divide by zero")

            print({
                "bleu": eval_metric["bleu"]*100,
                "eval_rouge1": eval_metric["rouge1"]*100,
                "eval_rouge2": eval_metric["rouge2"]*100,
                "eval_rougeL": eval_metric["rougeL"]*100,
                "eval_rougeLsum": eval_metric["rougeLsum"]*100,
                "meteor": eval_metric["meteor"]*100,
            })
    
            print(f"Input text: {instruction}")
            print(f"Output summary: {output_sum}")
            pf.write(f"{output_sum}\n")
            gf.write(f"{sum}\n")
            print(f"Gold Output summary: {sum}\n\n\n")

    metric.add_batch(predictions=tokenized_output_sum_lst, references=tokenized_sum_lst)
    eval_metric = metric.compute()
    print({
        "FINAL // bleu": eval_metric["bleu"]*100,
        "eval_rouge1": eval_metric["rouge1"]*100,
        "eval_rouge2": eval_metric["rouge2"]*100,
        "eval_rougeL": eval_metric["rougeL"]*100,
        "eval_rougeLsum": eval_metric["rougeLsum"]*100,
        "meteor": eval_metric["meteor"]*100,
    })



def sum_w_cda(model_type, src_lst, sum_lst, metric, promptor):
    with open(f"./result/pred_w_cda_{model_type}", 'w') as pf, open(f"./result/gold_w_cda_{model_type}", 'w') as gf, open(f"./result/counterfactual_w_cda_{model_type}", 'w') as cf:
        tokenized_output_sum_lst, tokenized_sum_lst = [], []
    
        for i, (src, sum) in tqdm(enumerate(zip(src_lst, sum_lst)), total=len(src_lst)):
            prev_gold_sum = sum_lst[i-1]
            
            counterfactual_instruction = mk_inst_for_counterfactual_summary_en(src)
            counterfactual_sum = promptor.do_llm(counterfactual_instruction)
            
            # counterfactual_sum = counterfactual_sum.split("[Counterfactual Summary]")[-1].replace('\n', ' ')
            counterfactual_sum = clean_data_ko(counterfactual_sum)


            if nshot == 0:
                instruction = mk_inst_for_summary_w_cda_en(src, counterfactual_sum)
            elif nshot == 1:
                instruction = mk_inst_for_summary_w_1shot(src, prev_gold_sum)
            
            output_sum = promptor.do_llm(instruction)

            if nshot == 0:
                output_sum = output_sum.split("[요약]")[-1].replace('\n', ' ')
            elif nshot == 1:
                output_sum = output_sum.split("[예제 요약]")[-1].replace('\n', ' ')

            output_sum = clean_data_ko(output_sum)
            output_sum, sum = postprocess_text(output_sum, sum)

            tokenized_output_sum = ' '.join(tokenizer.tokenize(output_sum))
            tokenized_sum = ' '.join(tokenizer.tokenize(sum))

            tokenized_output_sum_lst.append(tokenized_output_sum)
            tokenized_sum_lst.append(tokenized_sum)
            metric.add_batch(predictions=[tokenized_output_sum], references=[tokenized_sum])
            try:
                eval_metric = metric.compute()
            except ZeroDivisionError as e:
                print("Error: Cannot divide by zero")

            print({
                "bleu": eval_metric["bleu"]*100,
                "eval_rouge1": eval_metric["rouge1"]*100,
                "eval_rouge2": eval_metric["rouge2"]*100,
                "eval_rougeL": eval_metric["rougeL"]*100,
                "eval_rougeLsum": eval_metric["rougeLsum"]*100,
                "meteor": eval_metric["meteor"]*100,
            })
    
            print(f"Input text: {instruction}")
            print(f"Output summary: {output_sum}")
            print(f"Gold Output summary: {sum}")
            print(f"Counterfactual summary: {counterfactual_sum}\n\n\n")
            pf.write(f"{output_sum}\n")
            gf.write(f"{sum}\n")
            cf.write(f"{counterfactual_sum}\n")

    metric.add_batch(predictions=tokenized_output_sum_lst, references=tokenized_sum_lst)
    eval_metric = metric.compute()
    print({
        "FINAL // bleu": eval_metric["bleu"]*100,
        "eval_rouge1": eval_metric["rouge1"]*100,
        "eval_rouge2": eval_metric["rouge2"]*100,
        "eval_rougeL": eval_metric["rougeL"]*100,
        "eval_rougeLsum": eval_metric["rougeLsum"]*100,
        "meteor": eval_metric["meteor"]*100,
    })

# src_lst, sum_lst = init_data()
# promptor = init_model()

# if do_cda:
#     sum_w_cda(model_type, src_lst, sum_lst, metric, promptor)
# else:
#     baseline(model_type, src_lst, sum_lst, metric, promptor)