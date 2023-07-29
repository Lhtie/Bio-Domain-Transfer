from transformers import AutoAdapterModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import argparse

from dataConfig.conll2003 import conll2003
from dataConfig.few_nerd import few_nerd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args, model, tokenizer, dataloader, adapter_name):
    model.to(device)
    model.train_adapter([adapter_name])
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": 1e-5,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=args.lr)

    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(batch["input_ids"])
            # we need to reformat the tensors for the loss function
            # they need to have the shape (N, C) and (N,) where N is the number
            # of tokens and C the number of classes
            predictions = torch.flatten(outputs[0], 0, 1)
            expected = torch.flatten(batch["labels"].long(), 0, 1)
            label_mask = torch.flatten(batch["label_mask"].long(), 0, 1)
            
            loss = (loss_fn(predictions, expected) * label_mask).mean()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            if i % 10000 == 0:
                print(f"loss: {loss}")
                
    model.save_adapter(f'adapter/{adapter_name}', adapter_name)
    model.save_head(f"head/{adapter_name + '_head'}", adapter_name + '_head')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--label_offset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    if args.dataset == "conll2003":
        data = conll2003(args.model_name.split('/')[-1], label_offset=args.label_offset)
    elif args.dataset == "few-nerd":
        data = few_nerd(args.model_name.split('/')[-1], label_offset=args.label_offset)
    else:
        raise NotImplementedError(f"dataset {args.dataset} is not supported")
    
    # load model
    model_name = args.model_name
    adapter_name = args.dataset + "_ner"
    config = AutoConfig.from_pretrained(model_name, num_label=len(data.labels), id2label=data.id2label, label2id=data.label2id)
    model = AutoAdapterModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # add adapter
    model.add_adapter(adapter_name)
    model.add_tagging_head(adapter_name + "_head", num_labels=len(data.labels), id2label=data.id2label)

    # load data
    dataset = data.load(tokenizer)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset["train"], batch_size=args.batch_size)
    
    print(model)
    # train
    train(args, model, tokenizer, dataloader, adapter_name)