import os
from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonInDirLoader, SummaryLoader, SummaryETRILoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor

from promptor.mk_instruction import mk_inst_exsum_meetsum

import torch
import argparse

from do_ex import baseline

import evaluate
from korouge_score import rouge_scorer




def load_data(data_dir):
    # SBSC data
    data_loader = DataLoader(JsonInDirLoader, "json")
    sum_loader = SummaryLoader(SummaryETRILoader)
    data_dir_list = data_loader.get_listdir(data_dir, '')
    json_lst = list(data_loader.load(data_dir_list))
    # ex_sent_lst = sum_loader.load(json_lst)
    # dialog_lst = sum_loader.load(json_lst, function_name="load_dialog")   # second augmentation

    return data_dir_list, json_lst

def load_model(args):
    if args.model_type == "gemma2":
        model_id = "carrotter/ko-gemma-2b-it-sft"
        # model_id = "rtzr/ko-gemma-2-9b-it"
        promptor = Promptor(Gemma2Promptor, model_id)
    elif args.model_type == "exaone":
        model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        promptor = Promptor(ExaonePromptor, model_id)
    elif args.model_type in ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"]:
        model_id = args.model_type
        promptor = Promptor(ChatGPTPromptor, model_id)

    return promptor

def postpro_ex_sum(aug_data):
    tmp_aug = aug_data.split(': ')[-1].strip()
    try:
        if tmp_aug[-1] == '.': tmp_aug = tmp_aug[:-1]
        if "[결과 id 리스트]:" in tmp_aug: tmp_aug = tmp_aug.split("[결과 id 리스트]: ")[-1]

        if tmp_aug[0] == '[' and tmp_aug[-1] != ']': tmp_aug += ']'
        elif tmp_aug[0] != '[' and tmp_aug[-1] == ']': tmp_aug = '[' + tmp_aug

        aug_ids = eval(tmp_aug)
    except:
        print(aug_data)
        aug_ids = []

    if type(aug_ids) == tuple: aug_ids = list(aug_ids)
    if 0 in aug_ids:
        del aug_ids[aug_ids.index(0)]
    
    return aug_ids



def ex_eval(output_doc_ids, oracle_doc_ids):
    # accuracy
    # indexes = torch.tensor(output_doc_ids)
    indexes = [torch.tensor(o_el) for o_el in output_doc_ids]
    target = [torch.tensor(o_el) for o_el in oracle_doc_ids]
    
    # macro f1
    hits = [torch.isin(idx, tgt).sum().item() for idx, tgt in zip(indexes, target)]
    recalls = [h/len(tgt) for h, tgt in zip(hits, target)]
    precs = [h/len(idx) for h, idx in zip(hits, indexes)]
    f1s = [2*(prec*rec) / (prec+rec) if (prec+rec) > 0 else 0 for prec, rec in zip(precs, recalls)]

    rec = sum(recalls) / len(recalls) * 100
    pre = sum(precs) / len(precs) * 100
    f1 = sum(f1s) / len(f1s) * 100

    c_cnt, o_cnt = 0, 0
    for ordata, outdata in zip(oracle_doc_ids, output_doc_ids):
        all_ids = ordata + outdata
        c_cnt += len(all_ids) - len(set(all_ids))
        o_cnt += len(outdata)

    avg_score = c_cnt/o_cnt * 100
    
    
    print(f"Score: [ACC] {avg_score:.2f}, [PREC] {pre:.2f}, [REC] {rec:.2f}, [F1] {f1:.2f}")
    # print(f"Retrieval Score: [Hits] {hr:.4f}, [RPEC] {rp:.4f}, [RREC] {r2:.4f}, [MAP] {map:.4f}, [MRR] {mrr:.4f}, [NDGC] {ndcg:.4f}")
    print(f"output doc ids: {output_doc_ids[0]}")
    print(f"oracle doc ids: {oracle_doc_ids[0]}")
    print("-------------------------")


def do_eval_meeting_summary(args, promptor, json_lst, sum_type='total_summary'):
    aug_ids_lst, ex_ids_lst = [], []
    for i, ori in tqdm(enumerate(json_lst), total=len(json_lst)):
        # make dialogue with sent_id
        dialogue = ori['dialogue']
        dialog_str = ' '.join([f'[{dial.get("sentence_id")}] {dial.get("sentence")}' for dial in dialogue])
        
        # gold data
        total_summary = ori[sum_type][0]
        if sum_type == "total_summary":
            topic_type = "total_topic"
            sentence_ids = 'total_sentence_ids'
        elif sum_type == "topic_summary":
            topic_type = "topic"
            sentence_ids = 'topic_sentence_ids'


        total_topic = total_summary[topic_type]
        ex_ids = total_summary[sentence_ids] if sentence_ids in total_summary else total_summary['speaker_sentence_ids']
        topic_cot = promptor.do_llm(f"Let's think step by step for the {total_topic}, 결과는 한국어로 출력해줘.")
        topic_input = f'Topic: {total_topic}, Sub-topics: {topic_cot}, 이와 관련있는 문장을 모두 찾으시오.'
        instruction = mk_inst_exsum_meetsum(dialog_str, topic_input, len(dialogue), int(len(dialogue)*0.3))
            
        aug_data = promptor.do_llm(instruction)

        first_aug_ids = postpro_ex_sum(aug_data)
        if first_aug_ids == []: 
            print(aug_data)
            continue

        ###
        step = 3
        new_dialogue = []
        try:
            for a_id in range(0, len(first_aug_ids), step):
                # aug_sent_range = range(aug_ids[a_id], aug_ids[a_id+step])
                end_id = a_id+step-1
                new_dialogue += dialogue[first_aug_ids[a_id]-1:first_aug_ids[end_id if end_id < len(first_aug_ids)-1 else len(first_aug_ids)-1]-1]

            new_dialog_str = ' '.join([f'[{dial.get("sentence_id")}] {dial.get("sentence")}' for dial in new_dialogue])

            instruction = mk_inst_exsum_meetsum(new_dialog_str, topic_input, new_dialog_str.count('['), 20)

            aug_data = promptor.do_llm(instruction)

            sec_aug_ids = postpro_ex_sum(aug_data)
            if sec_aug_ids == []: 
                print(aug_data)
                sec_aug_ids = first_aug_ids
        except:
            sec_aug_ids = first_aug_ids

        ######
        aug_ids_lst.append(sec_aug_ids)
        ex_ids_lst.append(ex_ids)

    ex_eval(aug_ids_lst, ex_ids_lst)

    return aug_ids_lst, ex_ids_lst


def abstractive_summary(json_lst, aug_ids_lst, ex_ids_lst, sum_type="total_summary"):
    src_lst, sum_lst = [], []
    if sum_type == "total_summary":
        asum_type = "total_asummary"
    elif sum_type == "topic_summary":
        asum_type = "topic_asummary"

    for i, (ori, aug_ids, ex_ids) in tqdm(enumerate(zip(json_lst, aug_ids_lst, ex_ids_lst)), total=len(json_lst)):
        # make dialogue with sent_id
        dialogue = ori['dialogue']
        dialogue_dict = {v['sentence_id']: v for v in dialogue}
        total_asummary = ori[sum_type][0][asum_type]

        ex_dial_str = ' '.join([dialogue_dict[ex_id].get("sentence").replace('n/', '').replace('o/', '').strip()
                                 for ex_id in ex_ids])

        src_lst.append(ex_dial_str)
        sum_lst.append(total_asummary)
        
    return src_lst, sum_lst



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/kilab/data/etri", dest="root_dir") 
    parser.add_argument("-dt", "--data_types", nargs='+', default=["timbel", "datamaker-2023-all"], dest="data_types", help="--data_types timbel datamaker-2023-all", type=str) 
    parser.add_argument("-st", "--summary_types", default="total_summary", dest="summary_types", help="--summary_types topic_summary", type=str) 
    parser.add_argument("-d", "--data_dir", default="summarization/ko", dest="data_dir")
    parser.add_argument("-s", "--save_dir", default="./result/etri", dest="save_dir") 
    parser.add_argument("-m", "--model_type", default="gpt-4o-mini", dest="model_type", help="model_type: [gpt-4o-mini, gpt-4-turbo, gemma2, exaone]")
    # parser.add_argument("-cda", "--do_cda", dest="do_cda", action="store_true")
    args = parser.parse_args()

    sum_type = args.summary_types
    promptor = load_model(args)

    # metric = evaluate.combine(["bleu", "rouge", "meteor"])
    metric = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    # tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    sum_range = "200~400"

    for data_type in args.data_types:
        data_path = Path(args.root_dir) / args.data_dir / data_type / "test"
        data_dir_list, json_lst  = load_data(data_path)

        aug_ids_lst, ex_ids_lst = do_eval_meeting_summary(args, promptor, json_lst, sum_type)

        src_lst, sum_lst = abstractive_summary(json_lst, aug_ids_lst, ex_ids_lst, sum_type)

        baseline(args.model_type, src_lst, sum_lst, sum_range, metric, promptor)

if __name__ == "__main__":
    main()
    # python do_etri_aug.py -dt timbel datamaker-2023-all --model_type gpt-4o-mini