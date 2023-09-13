echo "pc starting" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset chemdner --method concat-max --tune_src &&
echo "pc2chemdner concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset bc5cdr --method concat-max --tune_src &&
echo "pc2bc5cdr concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset drugprot --method concat-max --tune_src &&
echo "pc2drugprot concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset chemdner --method sentEnc-max --tune_src &&
echo "pc2chemdner sentEnc" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
echo "pc2bc5cdr sentEnc" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets pc --tgt_dataset drugprot --method sentEnc-max --tune_src 
echo "pc2drugprot sentEnc" &&
echo "pc all finished" &&
echo "id starting" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset chemdner --method concat-max --tune_src &&
echo "id2chemdner concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset bc5cdr --method concat-max --tune_src &&
echo "id2bc5cdr concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset drugprot --method concat-max --tune_src &&
echo "id2drugprot concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset chemdner --method sentEnc-max --tune_src &&
echo "id2chemdner sentEnc" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
echo "id2bc5cdr sentEnc" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets id --tgt_dataset drugprot --method sentEnc-max --tune_src &&
echo "id2drugprot sentEnc" &&
echo "id all finished" &&
echo "cg starting" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset chemdner --method concat-max --tune_src &&
echo "cg2chemdner concat" &&
CUDA_VISIBLE_DEVICES=0,1,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset bc5cdr --method concat-max --tune_src &&
echo "cg2bc5cdr concat" &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset drugprot --method concat-max --tune_src &&
echo "cg2drugprot concat" &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset chemdner --method sentEnc-max --tune_src &&
echo "cg2chemdner sentEnc" &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset bc5cdr --method sentEnc-max --tune_src &&
echo "cg2bc5cdr sentEnc" &&
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg.yaml --datasets cg --tgt_dataset drugprot --method sentEnc-max --tune_src &&
echo "cg2drugprot sentEnc" &&
echo "cg all finished" "$@"
