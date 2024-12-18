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

def baseline(model_type, src_lst, sum_lst, sum_range, metric, inst_maker, promptor):
    with open(f"./result/pred_{model_type}", 'w') as pf, open(f"./result/gold_{model_type}", 'w') as gf:
        tokenized_output_sum_lst, tokenized_sum_lst = [], []
        scores_dict = {}
        total_len = len(src_lst)

        for i, (src, sum) in tqdm(enumerate(zip(src_lst, sum_lst)), total=total_len):
            prev_gold_sum = sum_lst[i-1]
            if nshot == 0:
                instruction = inst_maker(src, sum_range)
            elif nshot == 1:
                instruction = inst_maker(src, prev_gold_sum)
            
            output_sum = "I'm sorry"
            while "I'm sorry" in output_sum or "죄송하지만 현재 작업을" in output_sum \
                or "죄송해, 내가 널 이해하지 못했어" in output_sum or "죄송하지만 이 요청은 내부 정책에" in output_sum \
                    or "죄송해요, 미완성된 원문은 도움을" in output_sum or "죄송하지만 글자 수가 너무 많아서" in output_sum \
                    or "죄송하지만 주어진 텍스트를 기반으로" in output_sum: 
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
            
            score_dict = gather_rouge(tokenized_sum, tokenized_output_sum, scores_dict, metric)

            '''
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
            '''

            print(score_dict)

            print()
            print(f"Input text: {instruction}")
            print(f"Output summary: {output_sum}")
            pf.write(f"{output_sum}\n")
            gf.write(f"{sum}\n")
            print(f"Gold Output summary: {sum}\n\n\n")

        avg_rouge(scores_dict, total_len)
        print_rouge(scores_dict)
    '''
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
    '''

def avg_rouge(scores_dict, total_len):
    for k, v in scores_dict.items():
        for k_el, v_el in v.items():
            scores_dict[k][k_el] = scores_dict[k][k_el] / total_len * 100

def print_rouge(scores_dict):
    for k, v in scores_dict.items():
        print(f"{k}: {v}")

def gather_rouge(ref, pred, scores_dict, metric):
    score_dict = metric.score(ref, pred)

    for k, v in score_dict.items():
        if k in scores_dict:
            scores_dict[k]['precision'] += score_dict[k].precision
            scores_dict[k]['recall'] += score_dict[k].recall
            scores_dict[k]['fmeasure'] += score_dict[k].fmeasure
        else:
            scores_dict[k] = {}
            scores_dict[k]['precision'] = score_dict[k].precision
            scores_dict[k]['recall'] = score_dict[k].recall
            scores_dict[k]['fmeasure'] = score_dict[k].fmeasure
        
    return score_dict

def sum_w_cda(model_type, src_lst, sum_lst, metric, promptor):
    with open(f"./result/pred_w_cda_{model_type}", 'w') as pf, open(f"./result/gold_w_cda_{model_type}", 'w') as gf, open(f"./result/counterfactual_w_cda_{model_type}", 'w') as cf:
        tokenized_output_sum_lst, tokenized_sum_lst = [], []
        scores_dict = {}
    
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
            '''
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
            '''
    
            gather_rouge(sum, output_sum, scores_dict, metric)

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


sum_range="30~200"
def main():
    if nshot == 0:
        inst_maker = mk_inst_for_summary
    elif nshot == 1:
        inst_maker = mk_inst_for_summary_w_1shot

    src_lst, sum_lst = init_data()
    promptor = init_model()

    if do_cda:
        sum_w_cda(model_type, src_lst, sum_lst, metric, promptor)
    else:
        baseline(model_type, src_lst, sum_lst, sum_range, metric, inst_maker, promptor)