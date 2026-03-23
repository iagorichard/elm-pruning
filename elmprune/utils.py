from typing import List
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
