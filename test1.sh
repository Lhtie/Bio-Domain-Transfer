echo "pc starting" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset chemdner_pse --method concat-max --tune_tgt &&
echo "pc2chemdner concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset bc5cdr_pse --method concat-max --tune_tgt &&
echo "pc2bc5cdr concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset drugprot_pse --method concat-max --tune_tgt &&
echo "pc2drugprot concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset chemdner_pse --method sentEnc-max --tune_tgt &&
echo "pc2chemdner sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset bc5cdr_pse --method sentEnc-max --tune_tgt &&
echo "pc2bc5cdr sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets pc --tgt_dataset drugprot_pse --method sentEnc-max --tune_tgt
echo "pc2drugprot sentEnc" &&
echo "pc all finished" &&
echo "id starting" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset chemdner_pse --method concat-max --tune_tgt &&
echo "id2chemdner concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset bc5cdr_pse --method concat-max --tune_tgt &&
echo "id2bc5cdr concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset drugprot_pse --method concat-max --tune_tgt &&
echo "id2drugprot concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset chemdner_pse --method sentEnc-max --tune_tgt &&
echo "id2chemdner sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset bc5cdr_pse --method sentEnc-max --tune_tgt &&
echo "id2bc5cdr sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets id --tgt_dataset drugprot_pse --method sentEnc-max --tune_tgt &&
echo "id2drugprot sentEnc" &&
echo "id all finished" &&
echo "cg starting" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset chemdner_pse --method concat-max --tune_tgt &&
echo "cg2chemdner concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset bc5cdr_pse --method concat-max --tune_tgt &&
echo "cg2bc5cdr concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset drugprot_pse --method concat-max --tune_tgt &&
echo "cg2drugprot concat" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset chemdner_pse --method sentEnc-max --tune_tgt &&
echo "cg2chemdner sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset bc5cdr_pse --method sentEnc-max --tune_tgt &&
echo "cg2bc5cdr sentEnc" &&
CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --nnodes=1 --nproc_per_node=3 run.py --model_path bert-base-uncased --cfg_file configs/para/transfer_learning_eg_disc.yaml --datasets cg --tgt_dataset drugprot_pse --method sentEnc-max --tune_tgt &&
echo "cg2drugprot sentEnc" &&
echo "cg all finished" "$@"