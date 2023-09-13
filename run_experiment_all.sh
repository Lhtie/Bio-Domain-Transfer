# src: pc
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset chemdner --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset bc5cdr --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset drugprot --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset chemdner --method sentEnc-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset drugprot --method sentEnc-max --tune_src "$@"

# src: id
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset chemdner --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset bc5cdr --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset drugprot --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset chemdner --method sentEnc-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset drugprot --method sentEnc-max --tune_src "$@"

# src: cg
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset chemdner --method concat-max --tune_src &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset bc5cdr --method concat-max --tune_src &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset drugprot --method concat-max --tune_src &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset chemdner --method sentEnc-max --tune_src &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset drugprot --method sentEnc-max --tune_src "$@"