import os
import json
import argparse
import evaluate
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from transformers import AutoConfig, AutoAdapterModel, AutoTokenizer
from utils.config import get_tgt_dataset, read_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="configs/sample.yaml")
    args = parser.parse_args()
    cfg = read_config(args.cfg_file)

    data = get_tgt_dataset(cfg)
    if cfg.ADAPTER.EVAL == "None":
        adapter_name = cfg.DATA.TGT_DATASET + "_ner_" + cfg.MODEL.BACKBONE
    else:
        adapter_name = cfg.ADAPTER.EVAL
    head_name = cfg.DATA.TGT_DATASET + f"_ner_{cfg.MODEL.BACKBONE}_head"
    if cfg.TRAIN.TWO_STAGE:
        head_name += "_2stage"

    # load model
    model_name = cfg.MODEL.PATH
    # config = AutoConfig.from_pretrained(model_name, num_label=len(data.labels), id2label=data.id2label, label2id=data.label2id)
    model = AutoAdapterModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load adapter
    model.load_adapter(os.path.join(cfg.OUTPUT.ADAPTER_SAVE_DIR, adapter_name))
    model.set_active_adapters([adapter_name])
    # print(model.adapter_summary())

    # load head
    model.load_head(os.path.join(cfg.OUTPUT.HEAD_SAVE_DIR, head_name))

    # load data
    dataset = data.load(tokenizer)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset["evaluation"], batch_size=cfg.EVAL.BATCH_SIZE)

    # predict
    res_file_name = cfg.DATA.TGT_DATASET + '-' + adapter_name
    res_file = os.path.join(cfg.OUTPUT.RESULT_SAVE_DIR, res_file_name + '.json')
    if not cfg.EVAL.OVERWRITE and os.path.exists(res_file):
        with open(res_file, "r") as f:
            res_dict = json.load(f)
        predictions = res_dict["predictions"]
        references = res_dict["references"]
    else:
        model.to(device).eval()
        predictions, references = [], []
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                preds = model(batch["input_ids"]).logits
                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=2)
            
            for label_mask, pred, ref in zip(batch["label_mask"], preds, batch["labels"]):
                predictions.append([data.id2label[id.item()] for mask, id in zip(label_mask, pred) if mask == 1])
                references.append([data.id2label[id.item()] for mask, id in zip(label_mask, ref) if mask == 1])

        if not os.path.exists(cfg.OUTPUT.RESULT_SAVE_DIR):
            os.makedirs(cfg.OUTPUT.RESULT_SAVE_DIR)
        with open(res_file, "w") as f:
            json.dump(dict(predictions=predictions, references=references), f)

    seqeval = evaluate.load('evaluate-metric/seqeval')
    results = seqeval.compute(predictions=predictions, references=references)
    print(results)