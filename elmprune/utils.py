from typing import List, Dict, Iterable
from pathlib import Path
import re
import shutil
import copy
import json
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

TIMESTAMP_RE = re.compile(r"^\d{8}$")


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

def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).cpu().eval()

def get_first_dataloader_image(dataloader: Iterable) -> torch.Tensor:
    batch = next(iter(dataloader))
    x = batch["image"]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x[:1].cpu()

def find_best_val_loss_files(root: str) -> list[Path]:
    root_path = Path(root).resolve()
    return sorted(root_path.rglob("best_val_loss.pth"))


def mirror_copy_files(src_root: str, dst_root: str) -> list[Path]:
    src_root_path = Path(src_root).resolve()
    dst_root_path = Path(dst_root).resolve()

    copied_files = []

    for src_file in find_best_val_loss_files(src_root_path):
        rel_path = src_file.relative_to(src_root_path)
        dst_file = dst_root_path / rel_path

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

        copied_files.append(dst_file)
    return copied_files


def extract_file_info(file_path: Path, root: Path) -> dict:
    rel_parts = file_path.relative_to(root).parts

    info = {
        "full_path": str(file_path),
        "relative_path": str(file_path.relative_to(root)),
        "timestamp": None,
        "backbone": None,
        "model": None,
        "fold": None,
    }

    if len(rel_parts) >= 5:
        info["timestamp"] = rel_parts[-5]
        info["backbone"] = rel_parts[-4]
        info["model"] = rel_parts[-3]
        info["fold"] = rel_parts[-2]

    return info


def collect_infos(root: str) -> list[dict]:
    root_path = Path(root).resolve()
    files = find_best_val_loss_files(root_path)

    infos = [extract_file_info(file_path, root_path) for file_path in files]
    return infos

def get_model(model_name, backbone):
    if model_name == "Unet":
        model = smp.Unet(
            encoder_name=backbone,  
            encoder_weights="imagenet",    
            in_channels=3,                 
            classes=3                      
        )
    elif model_name == "FPN":
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
    elif model_name == "DeepLabV3":
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
    elif model_name == "MAnet":
        model = smp.MAnet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
    elif model_name == "PAN":
        model = smp.PAN(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
    else:
        raise ValueError(f"Model {model_name} and backbone {backbone} not supported.")
    
    return model

def get_and_load_model(model_name, backbone_name, checkpoints_path):
    model = get_model(model_name, backbone_name)
    model.load_state_dict(torch.load(checkpoints_path, map_location=torch.device('cpu'))['model_state_dict'])
    return model

def save_pruned_model(pruned_model, input_example, path_out, importances_type, prune_percentage):
    traced_model = torch.jit.trace(pruned_model, input_example)
    pruned_prefix = "pruned_"
    file_out = Path(path_out / (f"pruned_{importances_type}_{int(prune_percentage * 100)}.pt"))
    traced_model.save(file_out)

def load_pruned_model(model_filepath):
    return torch.jit.load(model_filepath)

def get_val_dataloader_fold(dataloaders, fold_str):
    folder_num_search = re.search(r'\d+', fold_str)
    folder_num = int(folder_num_search.group())
    folder_id = folder_num - 1
    return dataloaders[folder_id][1]

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_name_to_module(model: nn.Module) -> Dict[str, nn.Module]:
    return {name: module for name, module in model.named_modules()}

def rank_normalize(values: List[float]) -> List[float]:
    if len(values) <= 1:
        return [1.0 for _ in values]
    x = torch.tensor(values, dtype=torch.float32)
    order = torch.argsort(torch.argsort(x))
    return ((order.float() + 1.0) / float(len(values))).tolist()