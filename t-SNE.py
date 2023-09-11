from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt

from utils.config import get_src_dataset, get_tgt_dataset, read_config
from utils.multi_similarity_loss import extract_feat
from utils.kmeans import *
from utils.entity_encoder import EntityEncoder
from dataConfig.chemdner import chemdner_pse, chemdner
from dataConfig.biomedical import biomedical

sapbert_path = "/mnt/data/oss_beijing/liuhongyi/models/SapBERT-from-PubMedBERT-fulltext"
bert_path = "/mnt/data/oss_beijing/liuhongyi/models/bert-base-uncased"
adapter_path = {
    "politics": "adapter/DAPT_Politics",
    "science": "adapter/DAPT_Science"
}

def draw(cfg, model, tokenizer, data, suffix='', is_src=True):
    dataset = data.load(tokenizer)

    cache_file = f"analysis/scatters/.cache/{data.ds_name}"
    if suffix != '':
        cache_file += f"-{suffix}"
    cache_file += ".pt"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            ids, feats = pickle.load(f)
    else:
        train_set = Dataset.from_dict(dataset['training'][:50])
        # train_set = dataset['training']
        train_set.set_format(type='torch')
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=16)
        model.to(cfg.device).eval()
        ids_list, feats_list = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(batch["input_ids"])
                feat = extract_feat(
                    outputs[0], 
                    batched_label_mask=batch["label_mask"],
                    batched_token_id=batch['token_id'],
                    batched_token_label=batch['ner_tags'],
                    pseudo_labels=batch['pse_tags'] if 'pse_tags' in batch else None
                )
                ids, feats, _ = feat.tensorize()
                ids_list.append(ids.detach().cpu().numpy())
                feats_list.append(feats.detach().cpu().numpy())
        ids = np.concatenate(ids_list)
        feats = np.concatenate(feats_list)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((ids, feats), f)
    
    ### ner_tag labels
    id2tag = {}
    for batch in dataset['training']:
        pse_tags = batch['pse_tags'] if 'pse_tags' in batch else torch.zeros_like(batch['ner_tags'])
        for token_id, tag, p_tag in zip(batch['token_id'], batch['ner_tags'], pse_tags):
            if tag > 0:
                id2tag[token_id.item()] = data.labels[tag].split("-")[-1]
            if p_tag > 0:
                id2tag[token_id.item()] = "OOD"
    labels = []
    for id in ids:
        labels.append(id2tag[id])

    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    z = tsne.fit_transform(feats)
    df = pd.DataFrame()
    df["x"] = z[:,0]
    df["y"] = z[:,1]
    title=f"T-SNE projection of {data.ds_name.split('_')[0]} entities ({suffix})"
    plt.clf()
    # plt.figure(figsize=(8, 6))
    color_palette = sns.color_palette("hls", len(set(labels)))
    # if is_src:
    #     color_palette = color_palette[:2]
    # else:
    #     color_palette = color_palette[2:]
    sns_plot = sns.scatterplot(x="x", y="y",  hue=labels,
                palette=color_palette, data=df)
    # indices = np.random.choice(range(len(ids)), 20)
    # for id, x, y in zip(ids[indices], df["x"][indices], df["y"][indices]):
    #     plt.annotate(data.etts[id], (x, y))
    sns_plot.set(title=title)
    # plt.legend(bbox_to_anchor=(1.25, 1), borderaxespad=0)
    sns_plot.figure.savefig(f"analysis/scatters/{data.ds_name}-{suffix}.png", dpi=300)

def draw_emb(cfg, tokenizer, data, suffix=''):
    ids, central_emb = [], []
    for id, (entity, events) in enumerate(data.ett_rel_set.items()):
        if len(events) > 0:
            ids.append(id)
            central_emb.append(np.mean(events, axis=0))
        
    central_emb = np.stack(central_emb)
    k_range = range(10, 20)
    best_k, best_labels, results = chooseBestKforKMeansParallel(central_emb, k_range)
    print(results)
    print(f"Best K: {best_k}")

    ### ner_tag labels
    dataset = data.load(tokenizer)
    id2tag = {}
    for split in ['training', 'development', 'evaluation']:
        for batch in dataset[split]:
            pse_tags = batch['pse_tags'] if 'pse_tags' in batch else torch.zeros_like(batch['ner_tags'])
            for token_id, tag, p_tag in zip(batch['token_id'], batch['ner_tags'], pse_tags):
                if tag > 0:
                    id2tag[token_id.item()] = data.labels[tag].split("-")[-1]
                if p_tag > 0:
                    id2tag[token_id.item()] = "OOD"
    labels = []
    for id in ids:
        if id in id2tag:
            labels.append(id2tag[id])
        else:
            labels.append("UNK")
    # labels = best_labels
    # oridata = biomedical(read_config("configs/para/transfer_learning.yaml"), cfg.MODEL.BACKBONE)
    # for id, label in zip(ids, best_labels):
    #     if label == 10 and id in id2tag:
    #         print(data.etts[id], id2tag[id], '{' + ', '.join([str(x) for x in oridata.ett_rel_set[oridata.etts[id]]]) + '}')

    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    z = tsne.fit_transform(central_emb)
    df = pd.DataFrame()
    df["x"] = z[:,0]
    df["y"] = z[:,1]
    title=f"T-SNE projection of {data.ds_name.split('_')[0]} entity embeddings ({suffix})"
    plt.clf()
    # plt.figure(figsize=(8, 6))
    color_palette = sns.color_palette("hls", len(set(labels)))
    sns_plot = sns.scatterplot(x="x", y="y",  hue=labels,
                palette=color_palette, data=df)
    sns_plot.set(title=title)
    plt.tight_layout()
    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    sns_plot.figure.savefig(f"analysis/scatters/{data.ds_name}-emb-{suffix}.png", 
                        dpi=300, bbox_inches='tight', bbox_extra_artists=[legend])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/para/transfer_learning.yaml")
    args = parser.parse_args()
    cfg = read_config(args.cfg_file)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    src_data = biomedical(cfg, cfg.MODEL.BACKBONE, retain_chem=True)
    # tgt_data = chemdner_pse(read_config("configs/para/transfer_learning.yaml"), cfg.MODEL.BACKBONE)
    tgt_data = chemdner(cfg.MODEL.BACKBONE)
    # src_data = get_src_dataset(cfg)
    # tgt_data = get_tgt_dataset(cfg)

    model_path = cfg.MODEL.PATH
    adapter_name = cfg.ADAPTER.EVAL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # model = AutoAdapterModel.from_pretrained(model_path)
    # draw(cfg, model, tokenizer, src_data, 'BERT', True)
    # draw(cfg, model, tokenizer, tgt_data, 'BERT', False)

    draw_emb(
        cfg, tokenizer,
        biomedical(read_config("configs/para/transfer_learning_eg.yaml"), cfg.MODEL.BACKBONE), 
        suffix='sentEnc'
    )

    # cfg.OUTPUT.ADAPTER_SAVE_DIR = "adapter/para/eg/concat-max/src-lambda=1.2/biomedical/chemdner"
    # model = AutoAdapterModel.from_pretrained(model_path)
    # model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"))
    # model.set_active_adapters([adapter_name])
    # draw(cfg, model, tokenizer, src_data, 'BERT+SRC (EG)', True)
    # draw(cfg, model, tokenizer, tgt_data, 'BERT+SRC (EG)', False)

    # cfg.OUTPUT.ADAPTER_SAVE_DIR = "adapter/para/biomedical_None/chemdner"
    # model = AutoAdapterModel.from_pretrained(model_path)
    # model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name))
    # model.set_active_adapters([adapter_name])
    # draw(cfg, model, tokenizer, tgt_data, 'BERT+SRC+TGT', False)

    # cfg.OUTPUT.ADAPTER_SAVE_DIR = "adapter/para/disc/biomedical_None/chemdner_pse"
    # model = AutoAdapterModel.from_pretrained(model_path)
    # model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name))
    # model.set_active_adapters([adapter_name])
    # draw(cfg, model, tokenizer, tgt_data, 'BERT+SRC+TGT (DISC)', False)