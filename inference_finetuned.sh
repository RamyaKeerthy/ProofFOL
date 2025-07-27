#!/bin/bash

python scripts/inference-finetuned.py \
    --base_model finetuned_model_id \
    --dataset_path path_to_test_data \
    --save_dir ./outputs \
    --save_name {data}_{model}{method}_fol_finetuned.json \
    --temperature 0.1 \
    --max_length 1000