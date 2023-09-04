# torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning.yaml
# torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_disc.yaml --tgt_dataset chemdner_pse --tune_tgt > results/tgt_chemdner.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_disc.yaml --tgt_dataset bc5cdr_pse --tune_tgt > results/tgt_bc5cdr.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_disc.yaml --tgt_dataset drugprot_pse --tune_tgt > results/tgt_drugprot.txt

torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_eg.yaml --tgt_dataset chemdner --method sentEnc-max --tune_src > results/src_chemdner.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_eg.yaml --tgt_dataset bc5cdr --method sentEnc-max --tune_src > results/src_bc5cdr.txt
torchrun --nnodes=1 --nproc-per-node=4 run.py --cfg_file configs/para/transfer_learning_eg.yaml --tgt_dataset drugprot --method sentEnc-max --tune_src > results/src_drugprot.txt
