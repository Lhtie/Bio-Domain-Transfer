import os
import json
import argparse
import evaluate
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from transformers import AutoConfig, AutoAdapterModel, AutoTokenizer

from dataConfig.conll2003 import conll2003
from dataConfig.few_nerd import few_nerd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapter_dir = "adapter/"
head_dir = "head/"
res_dir = "results/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--label_offset", type=int, default=0)
    parser.add_argument("--adapter_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", default=False, action="store_true")
    args = parser.parse_args()

    if args.dataset == "conll2003":
        data = conll2003(args.model_name.split('/')[-1], label_offset=args.label_offset)
    elif args.dataset == "few-nerd":
        data = few_nerd(args.model_name.split('/')[-1], label_offset=args.label_offset)
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported")
    if args.adapter_name == None:
        args.adapter_name = args.dataset + "_ner"
    head_name = args.dataset + "_ner_head"

    # load model
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name, num_label=len(data.labels), id2label=data.id2label, label2id=data.label2id)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load adapter
    model.load_adapter(os.path.join(adapter_dir, args.adapter_name))
    model.set_active_adapters([args.adapter_name])
    # print(model.adapter_summary())

    # load head
    model.load_head(os.path.join(head_dir, head_name))

    # load data
    dataset = data.load(tokenizer)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset["test"], batch_size=args.batch_size)

    # predict
    res_file = os.path.join(res_dir, args.dataset + '-' + args.adapter_name + '.json')
    if not args.overwrite and os.path.exists(res_file):
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

        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        with open(res_file, "w") as f:
            json.dump(dict(predictions=predictions, references=references), f)

    seqeval = evaluate.load('evaluate-metric/seqeval')
    results = seqeval.compute(predictions=predictions, references=references)
    print(results)