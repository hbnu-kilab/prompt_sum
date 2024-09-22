import os
from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonInDirLoader, SummaryLoader, SummaryETRILoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor

from promptor.mk_instruction import mk_inst_etri_augmentation

import argparse
import json
from copy import deepcopy


def load_data(data_dir):
    # SBSC data
    data_loader = DataLoader(JsonInDirLoader, "json")
    sum_loader = SummaryLoader(SummaryETRILoader)
    data_dir_list = data_loader.get_listdir(data_dir, '')
    json_lst = list(data_loader.load(data_dir_list))
    ex_sent_lst = sum_loader.load(json_lst)

    return data_dir_list, json_lst, ex_sent_lst

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

def aug_for_extracted_dialgoue(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type):
    save_path = Path(args.save_dir)

    with open(f"{save_path/data_type}.log", 'w') as pf:
        for i, (d_dir, ori, ext_lst) in tqdm(enumerate(zip(data_dir_list, json_lst, ex_sent_lst)), total=len(data_dir_list)):
            copy_ori = deepcopy(ori)

            title, file_ext = os.path.splitext(d_dir.split('/')[-1])
            for exts in ext_lst:
                new_ext_dict = {'문장 안 동의어 <s> 표기':[],
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
                for ext in tqdm(exts, total=len(exts), desc="Extracted sentence"):
                    ex_sent = ext["sentence"]
                    instruction = mk_inst_etri_augmentation(ex_sent)
                    
                    aug_data = promptor.do_llm(instruction)
                    # output_sum = clean_data_ko(aug_data)

                    print(f"Input text: {ex_sent}")
                    print(f"Augmented data: {aug_data}")

                    for a_d in aug_data.split('\n'):
                        a_d = a_d.strip()
                        if len(a_d) == 0: continue 

                        if a_d[0] == '[':
                            aug_data = a_d.split(']')

                            aug_type = aug_data[0].replace('[', '')

                            if len(aug_data[1:]) > 1:
                                aug_data = " ".join(aug_data[1:])
                            else:
                                aug_data = aug_data[1].strip()
                        
                            ext["sentence"] = aug_data
                            if aug_type in new_ext_dict:
                                new_ext_dict[aug_type].append(ext)
                            else:
                                print(f"ERR, key not in dictionary. AUG_TYPE: {aug_type}, AUG_DATA: {aug_data}")

                # save augmented data
                for aug_type, v in new_ext_dict.items():
                    idx = v["sentence_id"]-1
                    copy_ori["dialogue"][idx] = v

                    with open(f"{save_path/data_type}/{title}.{aug_type}{file_ext}") as of:
                        json.dump(copy_ori, of, indent=4, ensure_ascii=False)
    

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
        data_path = Path(args.root_dir) / args.data_dir / data_type / "train"
        data_dir_list, json_lst, ex_sent_lst = load_data(data_path)

        aug_for_extracted_dialgoue(args, promptor, data_dir_list, json_lst, ex_sent_lst, data_type)


if __name__ == "__main__":
    main()