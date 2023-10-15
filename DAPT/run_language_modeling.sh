dataset=$1

python3 -m torch.distributed.launch --nproc_per_node 4 run_language_modeling.py \
        --output_dir /root/autodl-tmp/models/DAPT_${dataset} \
        --model_type=bert \
        --model_name_or_path=/root/autodl-tmp/models/DAPT_${dataset} \
        --do_train \
        --do_eval \
        --train_data_file=data/${dataset}/train.txt \
        --eval_data_file=data/${dataset}/dev.txt \
        --per_device_train_batch_size=8 \
        --per_device_eval_batch_size=32 \
        --save_strategy steps \
        --save_steps 50 \
        --evaluation_strategy steps \
        --eval_steps 50 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --num_train_epochs 50 \
        --overwrite_output_dir \
        --mlm