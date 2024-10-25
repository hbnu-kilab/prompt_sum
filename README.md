# prompt_sum

# do_eval_meetsum 수행
```
# multidyle의 상위 폴더 경로 (github 폴더 위치)
export PYTHONPATH=$PYTHONPATH:$PWD

# multidyle 폴더 경로
export PYTHONPATH=$PYTHONPATH:$PWD
```

## arguments
```
root_dir: dataset root directory [str]
data_types: type of dataset [List]
summary_types: type of summarization [List]
data_dir: dataset directory name [str] -> ex) {root_dir} / {data_dir}
save_dir: results save directory [str]
model_type: type of LLM model [str]
pipeline_method: summarization pipeline method
    1) only_llm: extractive summary with LLM -> abstractive LLM summary
    2) only_encoder: extractive summary with encoder -> abstractive LLM summary
    3) util_llm: extractive summary with encoder -> extractive summary with LLM -> abstractive LLM summary
    4) merge_exs: extractive summary with encoder + extractive summary with LLM -> abstractive LLM summary
    5) only_gen: only abstractive LLM summary without extractive summary
```

## perform do_eval_meetsum.py
```
python do_eval_meetsum.py -dt timbel -st topic_summary -pm only_gen
```