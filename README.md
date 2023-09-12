### Packages

```bash
pip install -r requirements.txt
```



### Datasets

Download from Huggingface Lhtie/Bio-Domain-Transfer

```bash
git lfs install
git clone https://huggingface.co/datasets/Lhtie/Bio-Domain-Transfer
```

Modify `dataset_dir` from `dataConfig/config.py`: directory  path to datasets (eg. `./Bio-Domain-Transfer`)



### Models

Download BERT, SapBERT, S-PubMedBert-MS-MARCO-SCIFACT from Huggingface

```bash
git lfs install
git clone https://huggingface.co/bert-base-uncased
git clone https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext
git clone https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT
```

Modify `sapbert_path`, `sentbert_path`, `bert_path` from `dataConfig/confg.py`: directory path to models respectively



### Run

Running script (Multi-Processing)

```
torchrun --nnodes=1 --nproc_per_node=<# gpus> run.py \
	--cfg_file <configuration file> \
	--model_path <directory path to BERT model> \
	--seed <random seed> \
	--src_dataaset <dataset of source domain> \
	--tgt_dataset <dataset of target domain> \
```

