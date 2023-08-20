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
from dataConfig.chemdner_pse import chemdner_pse
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
        train_set.set_format(type='torch', columns=[
            'input_ids', 'token_type_ids', 'attention_mask', 'labels', 'label_mask', 'ner_tags', 'token_id'
        ])
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=16)
        model.to(cfg.device).eval()
        ids_list, feats_list = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(batch["input_ids"])
                feat, _ = extract_feat(
                    outputs[0], 
                    batched_label_mask=batch["label_mask"],
                    batched_token_id=batch['token_id'],
                    batched_token_label=batch['ner_tags'],
                    overlap_labels=[]
                )
                ids, feats, _ = feat.tensorize()
                ids_list.append(ids.detach().cpu().numpy())
                feats_list.append(feats.detach().cpu().numpy())
        ids = np.concatenate(ids_list)
        feats = np.concatenate(feats_list)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((ids, feats), f)

    ### is_chem & clustering labels
    # id2ischem = {}
    # if is_src:
    #     for batch in Dataset.from_dict(dataset['training'][:50]):
    #         for token_id, tag in zip(batch['token_id'], batch['ner_tags']):
    #             if tag > 0 and token_id != -1:
    #                 id2ischem[token_id] = tag in [1, 2]
    # else:
    #     for batch in Dataset.from_dict(dataset['training'][:50]):
    #         for token_id, tag in zip(batch['token_id'], batch['ner_tags']):
    #             if tag > 0 and token_id != -1:
    #                 id2ischem[token_id] = tag < len(data.labels)
    # src_data = get_src_dataset(cfg)
    # tgt_data = get_tgt_dataset(cfg)
    # overlap_labels = list(set(src_data.labels) & set(tgt_data.labels))
    # for batch in Dataset.from_dict(dataset['training'][:50]):
    #     for token_id, tag in zip(batch['token_id'], batch['ner_tags']):
    #         if tag > 0 and token_id != -1:
    #             id2ischem[token_id] = data.labels[tag] in overlap_labels
    # model = EntityEncoder(sapbert_path)
    # model = EntityEncoder(bert_path, cache_dir=".cache/", adapter_path=adapter_path[data.ds_name.split("_")[0].lower()])
    # embs = model.get_embedding(data.etts)
    # central_emb = np.stack(embs.values())
    # k_range = range(2, 20)
    # best_k, best_labels, results = chooseBestKforKMeansParallel(central_emb, k_range)
    # print(results)
    # labels = []
    # for id in ids:
    #     # cluster = best_labels[list(embs.keys()).index(data.etts[id])]
    #     # labels.append(cluster)
    #     if id2ischem[id]:
    #         # labels.append(f"Chemical-Group {cluster}")
    #         labels.append("Chemical")
    #     else:
    #         # labels.append(f"Non-Chemical-Group {cluster}")
    #         labels.append("Non-Chemical")
    
    ### ner_tag labels
    id2tag = {}
    for batch in dataset['training']:
        for token_id, tag in zip(batch['token_id'], batch['ner_tags']):
            if tag > 0:
                if tag < len(data.labels):
                    id2tag[token_id.item()] = data.labels[tag].split("-")[-1]
                else:
                    id2tag[token_id.item()] = "OOD"
    labels = []
    for id in ids:
        labels.append(id2tag[id])
    # feats = []
    # for id in ids:
    #     feats.append(embs[data.etts[id]])
    # feats = np.stack(feats)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/sample.yaml")
    args = parser.parse_args()
    cfg = read_config(args.cfg_file)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    src_data = biomedical(cfg, cfg.MODEL.BACKBONE, retain_chem=True)
    tgt_data = chemdner_pse(read_config("configs/para/transfer_learning.yaml"), cfg.MODEL.BACKBONE)
    # src_data = get_src_dataset(cfg)
    # tgt_data = get_tgt_dataset(cfg)

    model_path = cfg.MODEL.PATH
    adapter_name = cfg.ADAPTER.EVAL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoAdapterModel.from_pretrained(model_path)
    draw(cfg, model, tokenizer, src_data, 'BERT', True)
    draw(cfg, model, tokenizer, tgt_data, 'BERT', False)

    cfg.OUTPUT.ADAPTER_SAVE_DIR = "adapter/para/biomedical_None/chemdner"
    model = AutoAdapterModel.from_pretrained(model_path)
    model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"))
    model.set_active_adapters([adapter_name])
    draw(cfg, model, tokenizer, src_data, 'BERT+SRC', True)
    draw(cfg, model, tokenizer, tgt_data, 'BERT+SRC', False)

    # cfg.OUTPUT.ADAPTER_SAVE_DIR = "adapter/para/ms/entityEnc-max/politics/music"
    # model = AutoAdapterModel.from_pretrained(model_path)
    # model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name + "_inter"))
    # model.set_active_adapters([adapter_name])
    # draw(cfg, model, tokenizer, src_data, 'BERT+SRC (SW)', True)
    # draw(cfg, model, tokenizer, tgt_data, 'BERT+SRC (SW)', False)