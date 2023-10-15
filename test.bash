torchrun --nnodes=1 --nproc_per_node=4 run.py --cfg_file configs/para/transfer_learning_eg_dapt.yaml --model_path models/DAPT_chemdner --tgt_dataset chemdner --tune_src &&
torchrun --nnodes=1 --nproc_per_node=4 run.py --cfg_file configs/para/transfer_learning_eg_dapt.yaml --model_path models/DAPT_bc5cdr --tgt_dataset bc5cdr --tune_src &&
torchrun --nnodes=1 --nproc_per_node=4 run.py --cfg_file configs/para/transfer_learning_eg_dapt.yaml --model_path models/DAPT_drugprot --tgt_dataset drugprot --tune_src
