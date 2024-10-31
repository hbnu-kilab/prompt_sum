from do_eval_meetsum import calc_asum_score, argparse
from korouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-rd", "--root_dir", default="/kilab/data/etri", dest="root_dir") 
    parser.add_argument("-dt", "--data_types", nargs='+', default=["timbel", "datamaker-2023-all"], dest="data_types", help="--data_types timbel datamaker-2023-all", type=str) 
    parser.add_argument("-dp", "--data_phases", nargs='+', default=["train", "val", "test"], dest="data_phases", help="--data_phases train val test", type=str) 
    parser.add_argument("-st", "--summary_types", nargs='+', default=["total_summary"], dest="summary_types", help="--summary_types topic_summary", type=str) 
    parser.add_argument("-d", "--data_dir", default="summarization/ko", dest="data_dir")
    parser.add_argument("-s", "--save_dir", default="./result/sum_eval", dest="save_dir") 
    parser.add_argument("-m", "--model_type", default="gpt-4o-mini", dest="model_type", help="model_type: [gpt-4o-mini, gpt-4-turbo, gemma2, exaone]")
    parser.add_argument("-pm", "--pipeline_method", default="only_llm", dest="pipeline_method", help="model_type: [only_llm: llm e-sum -> llm a-sum, only_encoder: roberta -> llm a-sum, util_llm: roberta -> llm e-sum -> llm a-sum, merge_exs: reberta + llm e-sum -> llm a-sum, only_gen: only a-sum]")
    
    args = parser.parse_args()
    metric = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    
    calc_asum_score(args, metric)