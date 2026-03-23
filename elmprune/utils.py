
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

