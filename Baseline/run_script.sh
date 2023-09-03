torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_disc.yaml --tgt_dataset bc5cdr_pse --tune_tgt_ms > results/tgt_bc5cdr.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_disc.yaml --tgt_dataset drugprot_pse --tune_tgt_ms > results/tgt_drugprot.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_eg.yaml --tgt_dataset bc5cdr --method concat-max --two_stage_train > results/src_bc5cdr.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_eg.yaml --tgt_dataset drugprot --method concat-max --two_stage_train > results/src_drugprot.txt
