#!/bin/bash

python scripts/inference-few-shot.py \
    --base_model model_id \
    --dataset_path path_to_test_data \
    --fs_dataset ./prompts/fewshot_file \
    --save_dir ./outputs \
    --save_name {data}_{model}{method}_fol_3shot.json \
    --temperature 0.1 \
    --max_length 1000