import torch
import torch.nn as nn
import os
import json
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoConfig, AutoAdapterModel
from transformers.activations import get_activation
from transformers.adapters.configuration import AdapterConfig

import sys
sys.path.append('ZipIt/')
from graphs.base_graph import BIGGraph, NodeType
from model_merger import ModelMerge
from utils import set_seed

device = 'cuda' if torch.cuda.is_available() else 'cpu'
adapter_dir = 'adapter/'

class AdapterModeling(nn.Module):

    def __init__(
        self, 
        adapter_name,
        input_size,
        config: AdapterConfig
    ):
        super(AdapterModeling, self).__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.down_sample = self.input_size // config["reduction_factor"]
        self.config = config

        self.ff_down = nn.Linear(self.input_size, self.down_sample)
        self.non_linearity = get_activation(config["non_linearity"].lower())
        self.ff_up = nn.Linear(self.down_sample, self.input_size)
        self.residual_merge_l = nn.Linear(self.input_size, self.input_size, bias=False)
        self.unmerge_l = nn.Linear(self.input_size, self.input_size, bias=False)

        if config["init_weights"] == "bert":
            self.ff_down.apply(self.init_bert_weights)
            self.ff_up.apply(self.init_bert_weights)
        elif config["init_weights"] == "mam_adapter":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.ff_down[0].weight, a=math.sqrt(5))
                nn.init.zeros_(self.ff_up.weight)
                nn.init.zeros_(self.ff_down[0].bias)
                nn.init.zeros_(self.ff_up.bias)
        else:
            raise ValueError("Unknown init_weights type: {}".format(config["init_weights"]))

    def forward(self, x):
        assert self.input_size * 2 == x.shape[-1]
        residual_input = x[:, :, :self.input_size]
        x = x[:, :, self.input_size:]

        down = self.ff_down(x)
        down = self.non_linearity(down)

        up = self.ff_up(down)
        output = up + self.residual_merge_l(residual_input)
        return self.unmerge_l(output)

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class AdapterGraph(BIGGraph):
    
    def __init__(self, model):
        super().__init__(model)

    def graphify(self):
        input_node = self.create_node(node_type=NodeType.INPUT)
        residual_input = self.add_nodes_from_sequence('', ['residual_merge_l'], input_node, sep='')

        input_node = self.add_nodes_from_sequence('', 
            ['ff_down', 'non_linearity', NodeType.PREFIX], input_node, sep='')
        input_node = self.add_nodes_from_sequence('',
            ['ff_up', NodeType.SUM], input_node, sep='')
        self.add_directed_edge(residual_input, input_node)

        output_node = self.add_nodes_from_sequence('', 
            [NodeType.PREFIX, 'unmerge_l', NodeType.OUTPUT], input_node, sep='')

        return self

def convert_state_dict(raw_state_dict, layer):
    state_dict = {}
    for key, value in raw_state_dict.items():
        attrs = key.split('.')
        if int(attrs[3]) == layer:
            if "down" in attrs[7]:
                state_dict[f"ff_down.{attrs[-1]}"] = value
            elif "up" in attrs[7]:
                state_dict[f"ff_up.{attrs[-1]}"] = value
            else:
                state_dict[f"{attrs[7]}.{attrs[-1]}"] = value
    return state_dict

def convert_state_dict(raw_state_dict, new_state_dict, layer, reverse=False):
    for key, value in raw_state_dict.items():
        attrs = key.split('.')
        if int(attrs[3]) == layer:
            if "down" in attrs[7]:
                if reverse:
                    raw_state_dict[key] = new_state_dict[f"ff_down.{attrs[-1]}"]
                else:
                    new_state_dict[f"ff_down.{attrs[-1]}"] = value
            elif "up" in attrs[7]:
                if reverse:
                    raw_state_dict[key] = new_state_dict[f"ff_up.{attrs[-1]}"]
                else:
                    new_state_dict[f"ff_up.{attrs[-1]}"] = value
            else:
                if reverse:
                    raw_state_dict[key] = new_state_dict[f"{attrs[7]}.{attrs[-1]}"]
                else:
                    new_state_dict[f"{attrs[7]}.{attrs[-1]}"] = value

if __name__ == "__main__":
    tasks = ['conll2003_ner', 'few-nerd_ner']
    num_layers = 6
    seq_len = 512
    num_seq = 1024
    batch_size = 64

    adapters = {}
    raw_state_dicts = {}
    name_dict = {}
    for task in tasks:
        adapter_task_dir = os.path.join(adapter_dir, task)
        with open(os.path.join(adapter_task_dir, "adapter_config.json"), "r") as json_file:
            config = json.load(json_file)
        name = config["name"]
        hidden_size = config["hidden_size"]
        
        name_dict[task] = name
        adapters[name] = []
        raw_state_dicts[name] = torch.load(os.path.join(adapter_task_dir, "pytorch_adapter.bin"), map_location=device)
        
        for layer in range(num_layers):
            adapter = AdapterModeling(name, hidden_size, AdapterConfig.from_dict(config["config"]))
            state_dict = {}
            convert_state_dict(raw_state_dicts[name], state_dict, layer)
            adapter.load_state_dict(state_dict)
            adapter.to(device)

            adapters[name].append(adapter)

    merged_state_dict = deepcopy(raw_state_dicts[name_dict[tasks[0]]])
    for layer in range(num_layers):
        graphs = []
        for task in tasks:
            name = name_dict[task]
            graphs.append(AdapterGraph(adapters[name][layer]).graphify())
        if layer == 0:
            for idx, graph in enumerate(graphs):
                graph.draw(save_path=f"figs/adaptergraph_{idx}.jpg")

        set_seed(42)
        data_x = torch.rand(num_seq, seq_len, hidden_size * 2)
        data_y = torch.rand(num_seq)
        dataset = TensorDataset(data_x, data_y)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        merger = ModelMerge(*graphs, device=device)
        model = deepcopy(adapters[name_dict[tasks[0]]][layer])
        merger.transform(model, dataloader)

        if layer == 0:
            for idx, graph in enumerate(graphs):
                graph.draw(save_path=f"figs/adaptergraph_merged_{idx}.jpg")
        
        state_dict = merger.merged_model.state_dict()
        convert_state_dict(merged_state_dict, state_dict, layer, reverse=True)

    merged_dir = os.path.join(adapter_dir, "merged")
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    torch.save(merged_state_dict, os.path.join(merged_dir, "pytorch_adapter.bin")) 