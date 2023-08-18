#!/bin/bash
DOMAIN=${1}

if [ $DOMAIN == "politics" ]; then
    python3 -u run_language_modeling.py \
        --output_dir adapter/DAPT_Politics \
        --model_type=bert \
        --model_name_or_path=/mnt/data/oss_beijing/liuhongyi/models/bert-base-uncased \
        --adapter_name DAPT_Politics \
        --do_train \
        --train_data_file data/CrossNER/ner_data/Unlabeled/Politics/politics_integrated.txt \
        --save_strategy no \
        --mlm
elif [ $DOMAIN == "science" ]; then
    python3 -u run_language_modeling.py \
        --output_dir adapter/DAPT_Science \
        --model_type=bert \
        --model_name_or_path=/mnt/data/oss_beijing/liuhongyi/models/bert-base-uncased \
        --adapter_name DAPT_Science \
        --do_train \
        --train_data_file=data/CrossNER/ner_data/Unlabeled/Natural\ Science/science_integrated.txt \
        --save_strategy no \
        --mlm
fi