import os
import numpy as np
import pickle
import json
import torch
import datasets
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoAdapterModel
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class EntityEncoder:
    def __init__(
        self, 
        model_path="cambridgeltl/SapBERT-from-PubMedBERT-fulltext", 
        batch_size=128, 
        cache_dir=".cache/",
        adapter_path=None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if adapter_path is not None:
            self.model = AutoAdapterModel.from_pretrained(model_path)
            self.model.load_adapter(adapter_path)
            with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
                config = json.load(f)
            self.model.set_active_adapters([config["name"]])
        else:
            self.model = AutoModel.from_pretrained(model_path)
        self.batch_size = batch_size
        self.lock = torch.multiprocessing.Lock()
        self.cache_dir = cache_dir
        file_name = model_path.split('/')[-1]
        if adapter_path is not None:
            file_name += f"-{adapter_path.split('/')[-1]}"
        file_name += "-cached_embs.pt"
        self.cache_file = os.path.join(self.cache_dir, file_name)
        os.makedirs(cache_dir, exist_ok=True)

        if dist.is_initialized():
            self.device = torch.device("cuda", dist.get_rank())
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def get_dataloader(self, entities):
        dataset = datasets.Dataset.from_dict(dict(entities=entities, ids=range(len(entities))))

        def tokenize(batch):
            return self.tokenizer.batch_encode_plus(batch["entities"],
                                                    padding="max_length", 
                                                    max_length=50, 
                                                    truncation=True,
                                                    return_tensors="pt")

        dataset = dataset.map(tokenize, batched=True, batch_size=self.batch_size)
        dataset.set_format(type="torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'ids'])
        # if not dist.is_initialized():
        #     sampler = None
        #     world_size = 1
        # else:
        #     sampler = DistributedSampler(dataset, shuffle=False)
        #     world_size = dist.get_world_size()
        # dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, shuffle=False,
        #                                          batch_size=self.batch_size // world_size)
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=self.batch_size)
        return dataloader

    def run(self, data_loader):
        all_embs = {}
        for batch in tqdm(data_loader):
            batch_inputs = {k: v.to(self.device) for k, v in batch.items() if k != "ids"}
            with torch.no_grad():
                cls_rep = self.model(**batch_inputs)[0][:,0,:] # use CLS representation as the embedding
                for key, val in zip(batch["ids"], cls_rep.cpu().detach().numpy()):
                    all_embs[key.cpu().detach().item()] = val
        return all_embs

    def get_embedding(self, entities):
        with self.lock:
            if not os.path.exists(self.cache_file):
                embs = {}
                with open(self.cache_file, "wb") as f:
                    pickle.dump(embs, f)
            else:
                with open(self.cache_file, "rb") as f:
                    embs = pickle.load(f)
        
        needed = [e for e in entities if e not in embs]
        if len(needed) > 0:
            dataloader = self.get_dataloader(needed)
            all_embs = self.run(dataloader)

            with self.lock:
                with open(self.cache_file, "rb") as f:
                    embs = pickle.load(f)
                for key, val in all_embs.items():
                    embs[needed[key]] = val
                with open(self.cache_file, "wb") as f:
                    pickle.dump(embs, f)

        # if dist.is_initialized():
        #     dist.barrier()
        # with self.lock:
        #     with open(self.cache_file, "rb") as f:
        #         embs = pickle.load(f)
        return {e: embs[e] for e in entities}
        