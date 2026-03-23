from typing import List, Dict, Iterable
from pathlib import Path
import copy
import json
import torch
import torch.nn as nn


def get_all_conv_layer_names(model: nn.Module) -> list[str]:
    conv_layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer_names.append(name)

    return conv_layer_names

def get_layer_by_string(model, path):
    current_layer = model
    for part in path.split("."):
        if part.isdigit():
            current_layer = current_layer[int(part)]
        else:
            current_layer = getattr(current_layer, part)
    return current_layer

def compute_constant_baseline_loss(Y: torch.Tensor) -> float:
    mean_pred = Y.mean(dim=0, keepdim=True)
    return float(torch.mean((Y - mean_pred) ** 2).item())

def discover_conv2d_layers(model: nn.Module, only_decoder: bool = False) -> List[str]:
    result = []

    root_module = model.decoder if only_decoder and hasattr(model, "decoder") else model

    for name, module in root_module.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            result.append(name)

    return result

def dump_dict(dictx : Dict, dict_path: Path):
    with open(dict_path, 'w') as json_file:
        json.dump(dictx, json_file, indent=4)

def load_dict(dict_path: Path):
    with open(dict_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_first_dataloader_image(dataloader: Iterable):
    batch = next(iter(dataloader))
    return batch["image"][0].unsqueeze(0)

def clone_model(model: torch.nn):
    cloned_model = copy.deepcopy(model)
    cloned_model = cloned_model.cpu()
    return cloned_model