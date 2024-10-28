import os
from tqdm import tqdm
from pathlib import Path
from loader import DataLoader, JsonInDirLoader, SummaryLoader, SummaryETRILoader
from promptor import Promptor, ExaonePromptor, Gemma2Promptor, ChatGPTPromptor

from promptor.mk_instruction import mk_inst_exsum_meetsum, mk_inst_for_meeting_summary, \
                                    mk_inst_exsum_w_exids, mk_inst_for_summary, \
                                    mk_inst_for_meeting_summary_new, mk_inst_for_meeting_summary_v5

import torch
import argparse
from transformers import AutoTokenizer

from eval.clean_text import postprocess_text, clean_data_ko

# import evaluate
from korouge_score import rouge_scorer
from copy import deepcopy

import sys
sys.path.append('/home/parkce/git-hubs/')
sys.path.append('/home/parkce/git-hubs/multidyle')
import json
from multidyle.test_multi_dyle import test as multidyle_test
from multidyle.config import Config 

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")


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


def mk_topic(promptor, ori, use_cot, sum_type='total_summary'):
    topic_input_lst, ori_topic_lst = [], []
    if sum_type == "total_summary":
        topic_type = "total_topic"
    elif sum_type == "topic_summary":
        topic_type = "topic"

    for summary in tqdm(ori[sum_type], total=len(ori[sum_type]), desc="mk_topic"):
        topic = summary[topic_type]
        if use_cot:
            topic_cot = promptor.do_llm(f"Let's think step by step for the {topic}, 결과는 한국어로 출력해줘.")
            topic_input = f'Topic: {topic}, Sub-topics: {topic_cot}, 이와 관련있는 문장을 모두 찾으시오.'
        else:
            topic_input = topic

        topic_input_lst.append(topic_input)
        ori_topic_lst.append(topic)

    return topic_input_lst, ori_topic_lst


def get_gold_ex_sum(ori, sum_type='total_summary'):
    # gold data
    total_summary = ori[sum_type][0]
    if sum_type == "total_summary":
        sentence_ids = 'total_sentence_ids'
    elif sum_type == "topic_summary":
        sentence_ids = 'topic_sentence_ids'

    gold_ids_lst = []
    for total_summary in tqdm(ori[sum_type], total=len(ori[sum_type]), desc="get_gold_ex_sum"):
        gold_ids = total_summary[sentence_ids] if sentence_ids in total_summary else total_summary['speaker_sentence_ids']
        gold_ids_lst.append(gold_ids)

    return gold_ids_lst

def do_ext_sum(promptor, ori, topics, multidyle_ex_id=None):
    aug_ids_lst = []
    # for i, (ori, topics) in tqdm(enumerate(zip(json_lst, topic_lst)), total=len(json_lst)):
        # make dialogue with sent_id
    dialogue = ori['dialogue']
    dialog_str = ' '.join([f'[{dial.get("sentence_id")}] {dial.get("sentence")}' for dial in dialogue])
    
    for topic_input in tqdm(topics, total=len(topics), desc="do_ext_sum"):
        # make instruction
        if multidyle_ex_id:
            instruction = mk_inst_exsum_w_exids(dialog_str, topic_input, len(dialogue), int(len(dialogue)*0.3), multidyle_ex_id)
        else:
            instruction = mk_inst_exsum_meetsum(dialog_str, topic_input, len(dialogue), int(len(dialogue)*0.3))
            
        # extractive summary using llm
        aug_data = "I'm sorry"
        cnt = 0
        while "I'm sorry" in aug_data or "죄송" in aug_data or 'Topic]과 관련된 문장' in aug_data:
            aug_data = promptor.do_llm(instruction)
            cnt += 1
            if cnt > 3: break

        first_aug_ids = postpro_ex_sum(aug_data)
        if first_aug_ids == []: 
            if multidyle_ex_id:
                first_aug_ids = multidyle_ex_id
            else:
                first_aug_ids = [fid for fid in range(0, len(dialogue), int(len(dialogue)*0.1))]
            print(aug_data)

        ret_aug_ids = first_aug_ids

        ###
        flag = True
        if flag:
            step = 3
            new_dialogue = []
            try:
                for a_id in range(0, len(first_aug_ids), step):
                    # aug_sent_range = range(aug_ids[a_id], aug_ids[a_id+step])
                    end_id = a_id+step-1
                    new_dialogue += dialogue[first_aug_ids[a_id]-1:first_aug_ids[end_id if end_id < len(first_aug_ids)-1 else len(first_aug_ids)-1]-1]

                new_dialog_str = ' '.join([f'[{dial.get("sentence_id")}] {dial.get("sentence")}' for dial in new_dialogue])

                instruction = mk_inst_exsum_meetsum(new_dialog_str, topic_input, new_dialog_str.count('['), 20)

                aug_data = "I'm sorry"
                cnt = 0
                while "I'm sorry" in aug_data or "죄송" in aug_data or 'Topic]과 관련된 문장' in aug_data:
                    aug_data = promptor.do_llm(instruction)
                    cnt += 1
                    if cnt > 3: break

                sec_aug_ids = postpro_ex_sum(aug_data)
                if sec_aug_ids == []: 
                    print(aug_data)
                    sec_aug_ids = first_aug_ids
            except:
                sec_aug_ids = first_aug_ids
            
            ret_aug_ids = sec_aug_ids
        ######

        aug_ids_lst.append(ret_aug_ids)

    return aug_ids_lst


def get_gold_asum(ori, sum_type="total_summary"):
    if sum_type == "total_summary":
        asum_type = "total_asummary"
    elif sum_type == "topic_summary":
        asum_type = "topic_asummary"

    gold_sum_lst = [t_sum[asum_type] for t_sum in ori[sum_type]]
    tokenized_sum_lst = [' '.join(tokenizer.tokenize(sum)) for sum in gold_sum_lst]

    return gold_sum_lst, tokenized_sum_lst

def mk_src_with_exids(ori, aug_ids_lst):
    src_lst = []

    dialogue = ori['dialogue']
    dialogue_dict = {v['sentence_id']: v for v in dialogue}
    
    for aug_ids in aug_ids_lst:
        ex_dial_str = ' '.join([dialogue_dict[ex_id].get("sentence").replace('n/', '').replace('o/', '').strip()
                                for ex_id in aug_ids if ex_id in dialogue_dict])
        src_lst.append(ex_dial_str)
        
    return src_lst

def do_abs_sum(src_lst, topic_lst, summary_sample, sum_range, inst_maker, promptor):
    output_sum_lst, tokenized_output_sum_lst = [], []
    total_len = len(src_lst)

    for i, (src, topic) in tqdm(enumerate(zip(src_lst, topic_lst)), total=total_len, desc="do_abs_sum"):
        instruction = inst_maker(src, topic, summary_sample, sum_range)
        
        output_sum = "I'm sorry"
        cnt = 0
        while "None" in output_sum or "I'm sorry" in output_sum or "죄송하지만 현재 작업을" in output_sum \
            or "죄송해, 내가 널 이해하지 못했어" in output_sum or "죄송하지만 이 요청은 내부 정책에" in output_sum \
                or "죄송해요, 미완성된 원문은 도움을" in output_sum or "죄송하지만 글자 수가 너무 많아서" in output_sum \
                or "죄송하지만 주어진 텍스트를 기반으로" in output_sum or "미안해" in output_sum: 
            output_sum = promptor.do_llm(instruction)
            cnt += 1
            if cnt > 3: break

        output_sum = output_sum.split("[요약]")[-1].replace('\n', ' ')
        output_sum = clean_data_ko(output_sum)

        output_sum_lst.append(output_sum)
        tokenized_output_sum = ' '.join(tokenizer.tokenize(output_sum))
        tokenized_output_sum_lst.append(tokenized_output_sum)

    return output_sum_lst, tokenized_output_sum_lst


def save_sum_result(ret_obj, output_sum_lst, sum_type, save_dir, data_path):
    # title, file_ext = os.path.splitext(data_path.split('/')[-1])
    title = data_path.split('/')[-1]

    if sum_type == "total_summary":
        asum_type = "total_asummary"
    elif sum_type == "topic_summary":
        asum_type = "topic_asummary"

    for sum_obj, output_sum in zip(ret_obj[sum_type], output_sum_lst):
        sum_obj[asum_type] = output_sum

    with open(save_dir / (f'{title}'), 'w') as of:
        json.dump(ret_obj, of, indent=4, ensure_ascii=False)


def do_tokenization_sum(sum_text, tok_method="bpe"):
    if tok_method in ["bpe"]:
        tokenized_sum_text = ' '.join(tokenizer.tokenize(sum_text)).replace('##', '')

    return tokenized_sum_text


def calc_asum_score(args, metric):
    sum_types = args.summary_types

    scores_dict = {}
    total_len = 0
    for data_type in args.data_types:
        for data_phase in args.data_phases:
            gold_path = Path(args.root_dir) / args.data_dir / data_type / data_phase
            gold_dir_list, gold_json_lst  = load_data(gold_path)
            g_title_lst = [os.path.splitext(g_path.split('/')[-1])[0] for g_path in gold_dir_list]

            scores_dict_json = {}
            inst_len = 0
            for sum_type in sum_types:
                pred_path = Path(f'./{Path(args.save_dir)/data_type}') / f'{data_phase}' / sum_type / args.pipeline_method
                pred_dir_list, pred_json_lst  = load_data(pred_path)
                p_title_dict = {os.path.splitext(g_path.split('/')[-1])[0]:g_i for g_i, g_path in enumerate(pred_dir_list)}
                
                pred_json_lst = [pred_json_lst[p_title_dict[g_t]] for g_t in g_title_lst]

                if sum_type == "total_summary":
                    asum_type = "total_asummary"
                    topic_type = "total_topic"
                elif sum_type == "topic_summary":
                    asum_type = "topic_asummary"
                    topic_type = "topic"

                for pred_dict, gold_dict in tqdm(zip(pred_json_lst, gold_json_lst), total=len(gold_json_lst), desc="eval results"):
                    for pred_sums, gold_sums in zip(pred_dict[sum_type], gold_dict[sum_type]):
                        total_len += 1
                        inst_len += 1
                        gold_tok = do_tokenization_sum(gold_sums[asum_type])
                        pred_tok = do_tokenization_sum(pred_sums[asum_type])
                        score_dict = gather_rouge(pred_tok, gold_tok, scores_dict, metric)
                        _ = gather_rouge(pred_tok, gold_tok, scores_dict_json, metric)

                        print(f"Topic: {gold_sums[topic_type]}")
                        print(f"Output summary: {pred_sums[asum_type]}")
                        print(f"Gold Output summary: {gold_sums[asum_type]}\n\n\n")
                        print(score_dict)

            print("JSON SCORE:")
            avg_rouge(scores_dict_json, inst_len)
            print_rouge(scores_dict_json)

    print("ALL Abstractive Summary Score:")
    avg_rouge(scores_dict, total_len)
    print_rouge(scores_dict)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/kilab/data/etri", dest="root_dir") 
    parser.add_argument("-dt", "--data_types", nargs='+', default=["timbel", "datamaker-2023-all"], dest="data_types", help="--data_types timbel datamaker-2023-all", type=str) 
    parser.add_argument("-dp", "--data_phases", nargs='+', default=["train", "val", "test"], dest="data_phases", help="--data_phases train val test", type=str) 
    parser.add_argument("-st", "--summary_types", nargs='+', default=["total_summary"], dest="summary_types", help="--summary_types topic_summary", type=str) 
    parser.add_argument("-d", "--data_dir", default="summarization/ko", dest="data_dir")
    parser.add_argument("-s", "--save_dir", default="./result/sum_eval", dest="save_dir") 
    parser.add_argument("-m", "--model_type", default="gpt-4o-mini", dest="model_type", help="model_type: [gpt-4o-mini, gpt-4-turbo, gemma2, exaone]")
    # parser.add_argument("-cda", "--do_cda", dest="do_cda", action="store_true")
    parser.add_argument("-pm", "--pipeline_method", default="only_llm", dest="pipeline_method", help="model_type: [only_llm: llm e-sum -> llm a-sum, only_encoder: roberta -> llm a-sum, util_llm: roberta -> llm e-sum -> llm a-sum, merge_exs: reberta + llm e-sum -> llm a-sum, only_gen: only a-sum]")
    args = parser.parse_args()

    data_dir = args.data_dir
    save_path = Path(args.save_dir)
    sum_types = args.summary_types
    promptor = load_model(args)
    # inst_maker = mk_inst_for_meeting_summary
    inst_maker = mk_inst_for_meeting_summary_v5
    # mk_inst_for_summary

    # metric = evaluate.combine(["bleu", "rouge", "meteor"])
    metric = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])

    summary_sample = "AI와 챗GPT와 같은 기술은 직업 변화와 교육 방향성에 영향을 미친다. 챗GPT의 발달로 일정한 패턴으로 움직이는 직군이나 부가가치가 높은 일이 먼저 대체될 것이다. 챗GPT에 대한 기대가 높지만 현재는 현실적인 한계로 인해 부정적인 시각도 많다. 인공지능도 경험과 데이터 축적으로 인간의 판단과 유사한 정확도를 갖출 수 있을 것이며 점차 인공지능에 대한 인식이 긍정적으로 변화할 것이다. 기술이 진화됨에 따라 그에 맞는 능력을 갖추고 윤리적인 부분들을 바탕으로 최대한 활용할 수 있는 방향으로 나아가야 한다."

    scores_dict = {}
    all_aug_ids_lst, all_gold_ids_lst = [], []
    all_multidyle_ex_ids_lst = []
    total_len = 0
    for data_type in args.data_types:
        for data_phase in args.data_phases:
            data_path = Path(args.root_dir) / args.data_dir / data_type / data_phase
            data_dir_list, json_lst  = load_data(data_path)

            for sum_type in sum_types:
                if sum_type == "total_summary":
                    if data_type == "timbel":
                        sum_range = "300~500"
                    elif data_type == "datamaker":
                        sum_range = "50~200"
                elif sum_type == "topic_summary":
                    sum_range = "200~300"

                if args.pipeline_method in ['util_llm', 'merge_exs', 'only_encoder']:
                    multidyle_config = Config()
                    # use multidyle encoder
                    multidyle_data_type = data_type.split('-')[0]
                    if False:
                        multidyle_config.retriever_name_or_path = "klue/roberta-large"
                        multidyle_config.eval_model_dir = '/kilab/models/summarization/multidyle/encoder/epochs_1--val_26.3946'
                    else:
                        multidyle_config.retriever_name_or_path = "klue/roberta-small"
                        multidyle_config.eval_model_dir = '/kilab/models/summarization/multidyle/encoder/epochs_4--val_28.6667'

                    multidyle_config.test_type = multidyle_data_type
                    multidyle_config.dataset = [f'/kilab/data/etri/{data_dir}/{multidyle_data_type}/']
                    
                    for ex_data_dir in multidyle_config.dataset:
                        if os.path.exists(ex_data_dir):
                            for f_name in os.scandir(ex_data_dir):
                                if "_val" in f_name.path or "_test" in f_name.path:
                                    os.remove(f_name.path)

                    if sum_type == "total_summary":
                        multidyle_config.data_type = f"{multidyle_data_type}-onlytotal"
                    elif sum_type == "topic_summary":
                        multidyle_config.data_type = f"{multidyle_data_type}-onlytopic"
                    elif len(sum_types) > 1 and "topic_summary" in sum_types and "total_summary" in sum_types:
                        multidyle_config.data_type = f"{multidyle_data_type}-no_speaker"
                    multidyle_ex_ids = multidyle_test(multidyle_config, file_names=data_dir_list)

                    multidyle_ex_ids = [sorted(inner_lst) for inner_lst in multidyle_ex_ids]
                    all_multidyle_ex_ids_lst += multidyle_ex_ids


                # Abstractive summary
                if sum_type == "topic_summary":
                    use_cot = False
                else: use_cot = True


                # save_files
                save_dir = Path(f'./{save_path/data_type}') / f'{data_phase}' / sum_type / args.pipeline_method
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                else:
                    for f_name in os.scandir(save_dir):
                        os.remove(f_name.path)

                # do abstractive summary
                i = 0
                for j_i, json_obj in tqdm(enumerate(json_lst), total=len(json_lst), desc="json loop"):
                    scores_dict_json = {}
                    ret_obj = deepcopy(json_obj)

                    # get topic or make topic-CoT
                    topic_input_lst, ori_topic_lst = mk_topic(promptor, json_obj, use_cot, sum_type)

                    # get gold extractive and abstractive summary
                    gold_ids_lst = get_gold_ex_sum(json_obj, sum_type)
                    gold_sum_lst, tokenized_gold_sum_lst = get_gold_asum(json_obj, sum_type)

                    # do extractive summarization
                    if args.pipeline_method in ['util_llm']:
                        aug_ids_lst = do_ext_sum(promptor, json_obj, topic_input_lst, multidyle_ex_ids[i])
                    elif args.pipeline_method == 'merge_exs':
                        aug_ids_lst = do_ext_sum(promptor, json_obj, topic_input_lst, multidyle_ex_ids[i])
                        m_j = 0
                        ex_ids_lst = []
                        for aug_ids in aug_ids_lst:
                            ex_ids_lst.append(list(set(multidyle_ex_ids[m_j] + aug_ids)))
                        # ex_ids_lst = [list(set(n1 + n2)) for n1, n2 in zip(multidyle_ex_ids[i], aug_ids_lst)]
                        # mk_inst_exsum_wo_noise(sent, ex_ids_lst)
                        aug_ids_lst = ex_ids_lst
                    elif args.pipeline_method in ['only_encoder']:
                        aug_ids_lst = [multidyle_ex_ids[i]]
                    elif args.pipeline_method in ['only_gen']:
                        pass
                    else:
                        aug_ids_lst = do_ext_sum(promptor, json_obj, topic_input_lst)

                    # make srouce with extractive summary ids
                    if args.pipeline_method in ['only_gen']:
                        dialog_str = ' '.join([f'[{dial.get("sentence_id")}] {dial.get("sentence")}' for dial in json_obj['dialogue']])
                        src_lst = [dialog_str] * len(topic_input_lst)
                    else:
                        src_lst = mk_src_with_exids(json_obj, aug_ids_lst)

                    # do abstractive summarization
                    output_sum_lst, tokenized_output_sum_lst = do_abs_sum(src_lst, topic_input_lst, summary_sample, sum_range, inst_maker, promptor)
                    total_len += len(src_lst)

                    save_sum_result(ret_obj, output_sum_lst, sum_type, save_dir, data_dir_list[j_i])
                    
                    # scoring
                    assert len(src_lst) == len(output_sum_lst)
                    for src, ori_topic, output_sum, gold_sum, \
                        tok_output_sum, tok_gold_sum, aug_ids, gold_ids in zip(src_lst, ori_topic_lst, output_sum_lst, gold_sum_lst, \
                        tokenized_output_sum_lst, tokenized_gold_sum_lst, \
                            aug_ids_lst, gold_ids_lst):
                        tok_output_sum = tok_output_sum.replace('##', '')
                        tok_gold_sum = tok_gold_sum.replace('##', '')
                        score_dict = gather_rouge(tok_output_sum, tok_gold_sum, scores_dict, metric)
                        _ = gather_rouge(tok_output_sum, tok_gold_sum, scores_dict_json, metric)

                        # print
                        # evaluation for extractive summary 
                        # if args.pipeline_method not in ['only_gen']:
                        #     print(f"Input text: {src}")
                        print(f"\nTopic: {ori_topic}")
                        print(f"Output summary: {output_sum}")
                        print(f"Gold Output summary: {gold_sum}\n")

                        if args.pipeline_method not in ['only_gen']:
                            ex_eval([aug_ids], [gold_ids])
                        print(score_dict)
                        print()

                    print("JSON SCORE:")
                    if args.pipeline_method not in ['only_gen']:
                        ex_eval(aug_ids_lst, gold_ids_lst)
                    avg_rouge(scores_dict_json, len(src_lst))
                    print_rouge(scores_dict_json)

                    if args.pipeline_method not in ['only_gen']:
                        all_aug_ids_lst += aug_ids_lst
                    all_gold_ids_lst += gold_ids_lst
                    i += 1

    print("ALL SCORE:")
                        
    avg_rouge(scores_dict, total_len)
    print_rouge(scores_dict)

    calc_asum_score(args, metric)
    
    if args.pipeline_method not in ['only_gen']:
        if "encoder" in args.pipeline_method:
            ex_eval(all_multidyle_ex_ids_lst, all_gold_ids_lst)
        else:
            ex_eval(all_aug_ids_lst, all_gold_ids_lst)

if __name__ == "__main__":
    main()
    # python do_etri_aug.py -dt timbel datamaker-2023-all --model_type gpt-4o-mini
    # python do_eval_meetsum.py -dt timbel -st total_summary -pm util_llm -dp test -d summarization/ko_reset