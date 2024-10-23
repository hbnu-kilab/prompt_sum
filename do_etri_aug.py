import os
from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonInDirLoader, SummaryLoader, SummaryETRILoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor

from promptor.mk_instruction import mk_inst_etri_augmentation, mk_inst_exsum_wo_noise, mk_inst_get_exsum

import argparse
import json
from copy import deepcopy


def load_data(data_dir):
    # SBSC data
    data_loader = DataLoader(JsonInDirLoader, "json")
    sum_loader = SummaryLoader(SummaryETRILoader)
    data_dir_list = data_loader.get_listdir(data_dir, '')
    json_lst = list(data_loader.load(data_dir_list))
    ex_sent_lst = sum_loader.load(json_lst, function_name="load_total_ex")   # first augmentation
    dialog_lst = sum_loader.load(json_lst, function_name="load_dialog")   # second augmentation

    return data_dir_list, json_lst, ex_sent_lst, dialog_lst

def load_mode(args):
    if args.model_type == "gemma2":
        model_id = "carrotter/ko-gemma-2b-it-sft"
        # model_id = "rtzr/ko-gemma-2-9b-it"
        promptor = Promptor(Gemma2Promptor, model_id)
    elif args.model_type == "exaone":
        model_id = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        promptor = Promptor(ExaonePromptor, model_id)
    elif args.model_type in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
        model_id = args.model_type
        promptor = Promptor(ChatGPTPromptor, model_id)

    return promptor

def aug_for_extracted_dialgoue(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type):
    save_path = Path(args.save_dir)

    with open(f"{save_path/data_type}.log", 'w') as pf:
        for i, (d_dir, ori, ext_lst) in tqdm(enumerate(zip(data_dir_list, json_lst, ex_sent_lst)), total=len(data_dir_list)):
            copy_ori = deepcopy(ori)

            title, file_ext = os.path.splitext(d_dir.split('/')[-1])
            for exts in ext_lst:
                new_ext_dict = {'입력 문장에 치환 가능 명사만 <>로 감싸기':[],
                                '문장 구조 변경': [],
                                '구어체 변형': [],
                                '어순 변형': [],
                                '감정 강조': [],
                                '부정 표현 추가': [],
                                '질문형으로 변형': [],
                                '감정 추가': [],
                                '디테일 추가': [],
                                '상반된 상황 표현': [],
                                '피동형 사용': [],
                                '무작위성 도입': [],
                                '비유적 표현 추가': [],
                                '반어법 사용': [],
                                '주어를 강조': [],
                                '상황 설명 추가': [],
                                '시제 변경': [],
                                '복합문으로 변형': [],
                                '간결한 표현으로 축약': [],
                                '강조 표현 사용': [],
                                '유머 추가': [],
                                '청중에게 질문하는 방식': []}
                for ext in tqdm(exts, total=len(exts), desc=f"Ex sentence: {title}"):
                    ex_sent = ext["sentence"]
                    instruction = mk_inst_etri_augmentation(ex_sent)
                    
                    aug_data = promptor.do_llm(instruction)
                    # output_sum = clean_data_ko(aug_data)

                    print(f"### FILE NAME: {title}")
                    print(f"Input text: {ex_sent}\n")
                    print(f"Augmented data: {aug_data}\n")

                    for a_d in aug_data.split('\n'):
                        copy_ext = deepcopy(ext)
                        a_d = a_d.strip()
                        if len(a_d) == 0: continue 

                        if a_d[0] == '[':
                            aug_data = a_d.split(']')

                            aug_type = aug_data[0].replace('[', '')

                            if len(aug_data[1:]) > 1:
                                aug_data = " ".join(aug_data[1:])
                            else:
                                aug_data = aug_data[1].strip()
                        
                            copy_ext["sentence"] = aug_data
                            if aug_type in new_ext_dict:
                                new_ext_dict[aug_type].append(copy_ext)
                            else:
                                print(f"ERR, key not in dictionary. AUG_TYPE: {aug_type}, AUG_DATA: {aug_data}")

                # save augmented data
                for aug_type, aug_data_lst in new_ext_dict.items():
                    for aug_data in aug_data_lst:
                        idx = aug_data["sentence_id"]-1
                        copy_ori["dialogue"][idx] = aug_data

                    if aug_type == "입력 문장에 치환 가능 명사만 <>로 감싸기":
                        aug_type = "동의어표시"
                    with open(f"./{save_path/data_type}/{title}.{aug_type}{file_ext}", 'w') as of:
                        json.dump(copy_ori, of, indent=4, ensure_ascii=False)
    
def aug_dialogue_by_llm_ext(args, promptor, data_dir_list, json_lst, ex_sent_lst, dialog_lst, data_type):
    # all text
    # total extract sentence 
    # LLM -> total extract sentence와 관련있는 sentence만 출력
    # LLM의 결과와 total extract sentence랑 합쳐서 gen_all_text 생성
    # gen_all_text -> total summary 저장

    save_path = Path(args.save_dir)
    ori_sent_cnt, diff_sent_cnt = 0, 0
    aug_id_err = 0

    with open(f"{save_path/data_type}.log", 'w') as pf:
        for i, (d_dir, ori, ext_lst, dialog_dict) in tqdm(enumerate(zip(data_dir_list, json_lst, ex_sent_lst, dialog_lst)), total=len(data_dir_list)):
            if len(ext_lst) == 0: continue

            title, file_ext = os.path.splitext(d_dir.split('/')[-1])
            # make dialogue with sent_id
            ori_sent_cnt += len(dialog_dict)
            dialog_str = ' '.join([f'[{k}] {v.get("sentence")}' for k, v in dialog_dict.items()])
            try:
                ex_ids = [ex["sentence_id"] for ex in ext_lst[0]]
            except:
                print(f"ERR: {ext_lst[0]}")
                continue

            instruction = mk_inst_exsum_wo_noise(dialog_str, ex_ids)
                
            aug_data = promptor.do_llm(instruction)

            tmp_aug = aug_data.split(': ')[-1].strip()
            try:
                if tmp_aug[-1] == '.': tmp_aug = tmp_aug[:-1]
                if "[결과 id 리스트]:" in tmp_aug: tmp_aug = tmp_aug.split("[결과 id 리스트]: ")[-1]
                
                if tmp_aug[0] == '[' and tmp_aug[-1] != ']': tmp_aug += ']'
                elif tmp_aug[0] != '[' and tmp_aug[-1] == ']': tmp_aug = '[' + tmp_aug

                aug_ids = eval(tmp_aug)
            except:
                print(aug_data)
                aug_id_err += 1
                continue

            if type(aug_ids) == tuple: aug_ids = list(aug_ids)
            if 0 in aug_ids:
                del aug_ids[aug_ids.index(0)]

            merged_ids = sorted(set(aug_ids + ex_ids))
            diff_sent_cnt += len(merged_ids)

            merged_id_dict = {v:k for k, v in enumerate(merged_ids)}
            ori["total_summary"][0]["total_sentence_ids"] = [merged_id_dict[ex_id] for ex_id in ex_ids]
            if "speaker_sentence_ids" in ori["total_summary"][0]:
                ori["total_summary"][0]["speaker_sentence_ids"] \
                    = [merged_id_dict[ex_id] for ex_id 
                        in ori["total_summary"][0]["speaker_sentence_ids"] 
                        if ex_id in merged_id_dict]

            # aug_dial_lst = [{dialog_dict[mid]} for mid in merged_ids]
            aug_dial_lst = []
            no_merged_id = 0
            for mid in merged_ids:
                if mid in dialog_dict:
                    dialog_dict[mid]['sentence_id'] = merged_id_dict[mid]
                    aug_dial_lst.append(dialog_dict[mid])
                else: no_merged_id += 1

            ret_dict = {"dialog": aug_dial_lst, "total_summary": ori["total_summary"]}
            if 'metadata' in ori:
                ret_dict.update('metadata', ori['metadata'])


            with open(f"./{save_path/data_type}/{title}.wo_noise{file_ext}", 'w') as of:
                json.dump(ret_dict, of, indent=4, ensure_ascii=False)


    print(f"Num of no merged id in dialogue: {no_merged_id}")
    print(f"Num of augmentation format error: {aug_id_err}")
    print(f"Reduction ratio: {diff_sent_cnt / ori_sent_cnt:.4f}")


def reset_ex_ids(args, promptor, data_dir_list, json_lst, dialog_lst, sum_type, data_type):
    save_path = Path(args.save_dir)
    ori_sent_cnt, diff_sent_cnt = 0, 0
    aug_id_err = 0

    
    for d_dir, ori, dialog_dict in tqdm(zip(data_dir_list, json_lst, dialog_lst), total=len(json_lst), desc="json iter"):
        title, file_ext = os.path.splitext(d_dir.split('/')[-1])

        ret_dict = {'metadata': ori['metadata'], "dialog": dialog_lst,}

        ori_sent_cnt += len(dialog_dict)
        dialog_str = ' '.join([f'[{k}] {v.get("sentence")}' for k, v in dialog_dict.items()])
        
        for sum_type in ["total_summary", "topic_summary"]:
            if sum_type == "total_summary":
                asum_type = "total_asummary"
                sent_id_type = "total_sentence_ids"
            elif sum_type == "topic_summary":
                asum_type = "topic_asummary"
                sent_id_type = "topic_sentence_ids"

            topic_sum_lst = ori[sum_type]
            for topic_sum in topic_sum_lst:
                a_sum = topic_sum[asum_type]
                sent_ids = topic_sum[sent_id_type]
                topic = topic_sum["topic"]

                instruction = mk_inst_get_exsum(dialog_str, topic, a_sum, sent_ids)

                while True:                
                    aug_data = promptor.do_llm(instruction)

                    tmp_aug = aug_data.split(': ')[-1].strip()
                    try:
                        if tmp_aug[-1] == '.': tmp_aug = tmp_aug[:-1]
                        if "[결과 id 리스트]:" in tmp_aug: tmp_aug = tmp_aug.split("[결과 id 리스트]: ")[-1]
                        
                        if tmp_aug[0] == '[' and tmp_aug[-1] != ']': tmp_aug += ']'
                        elif tmp_aug[0] != '[' and tmp_aug[-1] == ']': tmp_aug = '[' + tmp_aug

                        aug_ids = eval(tmp_aug)
                        break
                    except:
                        continue


                if type(aug_ids) == tuple: aug_ids = list(aug_ids)
                if 0 in aug_ids:
                    del aug_ids[aug_ids.index(0)]

                # merged_id_dict = {v:k for k, v in enumerate(aug_ids)}
                topic_sum[sent_id_type] = aug_ids
                if sum_type == "total_summary":
                    if "speaker_sentence_ids" in ori["total_summary"][0]:
                        ori["total_summary"][0]["speaker_sentence_ids"] = aug_ids
                
            ret_dict.update(sum_type, ori[sum_type])

        # ret_dict = {'metadata': ori['metadata'], "dialog": dialog_lst, sum_type: ori[sum_type]}

        with open(f"./{save_path/data_type}/{title}.reset_eid{file_ext}", 'w') as of:
            json.dump(ret_dict, of, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/kilab/data/etri", dest="root_dir") 
    parser.add_argument("-dt", "--data_types", nargs='+', default=["timbel", "datamaker-2023-all"], dest="data_types", help="--data_types timbel datamaker-2023-all", type=str) 
    parser.add_argument("-d", "--data_dir", default="summarization/ko", dest="data_dir")
    parser.add_argument("-s", "--save_dir", default="./result/etri", dest="save_dir") 
    parser.add_argument("-m", "--model_type", default="gpt-4o-mini", dest="model_type", help="model_type: [gpt-4o-mini, gpt-4-turbo, gemma2, exaone]")
    parser.add_argument("-at", "--augmentation_type", default="style_transfer", dest="augmentation_type", help="augmentation_type: [style_transfer, filter_noise, all]")
    parser.add_argument("-st", "--summary_types", default="total_summary", dest="summary_types", help="--summary_types topic_summary", type=str) 
    # parser.add_argument("-cda", "--do_cda", dest="do_cda", action="store_true")
    args = parser.parse_args()

    promptor = load_mode(args)
    sum_type = args.summary_types

    for data_type in args.data_types:
        data_path = Path(args.root_dir) / args.data_dir / data_type / "train"
        data_dir_list, json_lst, ex_sent_lst, dialog_lst = load_data(data_path)

        if args.augmentation_type == "style_transfer":
            aug_for_extracted_dialgoue(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type)
        elif args.augmentation_type == "filter_noise":
            aug_dialogue_by_llm_ext(args, promptor, data_dir_list, json_lst, ex_sent_lst, dialog_lst, data_type)
        elif args.augmentation_type == "reset_eid":
            reset_ex_ids(args, promptor, data_dir_list, json_lst, dialog_lst, sum_type, data_type)
        elif args.augmentation_type == "all":
            aug_for_extracted_dialgoue(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type)
            aug_dialogue_by_llm_ext(args, promptor, data_dir_list, json_lst, ex_sent_lst, dialog_lst, data_type)


if __name__ == "__main__":
    main()
    # python do_etri_aug.py -dt timbel datamaker-2023-all --model_type gpt-4o-mini