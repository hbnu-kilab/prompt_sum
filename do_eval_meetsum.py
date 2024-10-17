import os
from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonInDirLoader, SummaryLoader, SummaryETRILoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor

from promptor.mk_instruction import mk_inst_exsum_meetsum

import torch
import argparse
import json
from copy import deepcopy


def load_data(data_dir):
    # SBSC data
    data_loader = DataLoader(JsonInDirLoader, "json")
    sum_loader = SummaryLoader(SummaryETRILoader)
    data_dir_list = data_loader.get_listdir(data_dir, '')
    json_lst = list(data_loader.load(data_dir_list))
    # ex_sent_lst = sum_loader.load(json_lst)
    # dialog_lst = sum_loader.load(json_lst, function_name="load_dialog")   # second augmentation

    return data_dir_list, json_lst

def load_mode(args):
    if args.model_type == "gemma2":
        model_id = "carrotter/ko-gemma-2b-it-sft"
        # model_id = "rtzr/ko-gemma-2-9b-it"
        promptor = Promptor(Gemma2Promptor, model_id)
    elif args.model_type == "exaone":
        model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        promptor = Promptor(ExaonePromptor, model_id)
    elif args.model_type in ["gpt-4o-mini", "gpt-4-turbo"]:
        model_id = args.model_type
        promptor = Promptor(ChatGPTPromptor, model_id)

    return promptor


def do_eval_meeting_summary(args, promptor, json_lst):
    for i, ori in tqdm(enumerate(json_lst), total=len(json_lst)):
        # make dialogue with sent_id
        dialog_str = ' '.join([f'[{k}] {v.get("sentence")}' for k, v in json_lst['dialogue'].items()])
        
        # gold data
        total_summary = json_lst['total_summary']
        total_topic = total_summary['total_topic']
        ex_ids = total_summary['total_sentence_ids']

        instruction = mk_inst_exsum_meetsum(dialog_str, total_topic)
            
        aug_data = promptor.do_llm(instruction)

        tmp_aug = aug_data.split(': ')[-1].strip()
        try:
            if tmp_aug[-1] == '.': tmp_aug = tmp_aug[:-1]
            
            if tmp_aug[0] == '[' and tmp_aug[-1] != ']': tmp_aug += ']'
            elif tmp_aug[0] != '[' and tmp_aug[-1] == ']': tmp_aug = '[' + tmp_aug

            aug_ids = eval(tmp_aug)
        except:
            print(aug_data)

        if type(aug_ids) == tuple: aug_ids = list(aug_ids)
        if 0 in aug_ids:
            del aug_ids[aug_ids.index(0)]

def eval(output_doc_ids, oracle_doc_ids):
    # accuracy
    indexes = torch.tensor(output_doc_ids)
    target = torch.tensor(oracle_doc_ids)
    # target = [torch.tensor(o_el) for o_el in oracle_doc_ids]
    
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
    
    # torchmetric
    preds = torch.cat(output_doc_scores).to("cpu")
    target_mask = torch.cat([torch.isin(idx, gold).long().unsqueeze(0) for idx, gold in zip(indexes, target)])
    rp = self.rp(preds, target_mask, indexes=indexes)
    r2 = self.r2(preds, target_mask, indexes=indexes)
    hr = self.hr(preds, target_mask, indexes=indexes)
    map = self.map(preds, target_mask, indexes=indexes)
    mrr = self.mrr(preds, target_mask, indexes=indexes)
    ndcg = self.ndcg(preds, target_mask, indexes=indexes)

    print(f"Score: [ACC] {avg_score:.2f}, [PREC] {pre:.2f}, [REC] {rec:.2f}, [F1] {f1:.2f}")
    print(f"Retrieval Score: [Hits] {hr:.4f}, [RPEC] {rp:.4f}, [RREC] {r2:.4f}, [MAP] {map:.4f}, [MRR] {mrr:.4f}, [NDGC] {ndcg:.4f}")
    print(f"output doc ids: {output_doc_ids[0]}")
    print(f"oracle doc ids: {oracle_doc_ids[0]}")
    print("-------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/kilab/data/etri", dest="root_dir") 
    parser.add_argument("-dt", "--data_types", nargs='+', default=["timbel", "datamaker-2023-all"], dest="data_types", help="--data_types timbel datamaker-2023-all", type=str) 
    parser.add_argument("-d", "--data_dir", default="summarization/ko", dest="data_dir")
    parser.add_argument("-s", "--save_dir", default="./result/etri", dest="save_dir") 
    parser.add_argument("-m", "--model_type", default="gpt-4o-mini", dest="model_type", help="model_type: [gpt-4o-mini, gpt-4-turbo, gemma2, exaone]")
    # parser.add_argument("-cda", "--do_cda", dest="do_cda", action="store_true")
    args = parser.parse_args()

    promptor = load_mode(args)

    for data_type in args.data_types:
        data_path = Path(args.root_dir) / args.data_dir / data_type / "test"
        data_dir_list, json_lst, ex_sent_lst = load_data(data_path)

        do_eval_meeting_summary(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type)


if __name__ == "__main__":
    main()
    # python do_etri_aug.py -dt timbel datamaker-2023-all --model_type gpt-4o-mini