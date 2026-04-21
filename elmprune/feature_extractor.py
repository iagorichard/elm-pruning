import sys
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from .utils import get_layer_by_string
from .importance_processor_config import ImportanceProcessorConfig


class FeatureExtractor:
    
    def __init__(self, config: ImportanceProcessorConfig, model: nn.Module, dataloader: Iterable, layer_names: List[str]):
        self.config = config 
        self.model = model
        self.dataloader = dataloader
        self.layer_names = layer_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_feature_and_targets(self)-> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        if self.config.feature_type == "segmentation":
            target_extractor = self.__segmentation_mask_histogram_target_extractor()
        else:
            target_extractor = self.__default_logits_gap_target_extractor

        return self.__process_features_and_targets(target_extractor)
    
    def __process_features_and_targets(self, target_extractor: Callable[[Any, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        print("[FeatureExtractor] Extracting features for ELM...")
        
        self.model.eval()
        self.model = self.model.to(self.device)
        storage_device = "cpu"

        hooks = []
        feature_storage: Dict[str, List[torch.Tensor]] = {layer_name: [] for layer_name in self.layer_names}
        target_storage: List[torch.Tensor] = []

        current_batch_features: Dict[str, torch.Tensor] = {}

        def make_hook(layer_name: str):
            def hook_fn(module, inputs, output):
                out = self.__extract_first_tensor(output)

                if out.dim() == 4:
                    # [N, C, H, W] -> [N, C]
                    pooled = out.mean(dim=(2, 3))
                elif out.dim() == 3:
                    pooled = out.mean(dim=2)
                elif out.dim() == 2:
                    pooled = out
                else:
                    pooled = out.flatten(start_dim=1)

                current_batch_features[layer_name] = pooled.detach().to(storage_device)
            return hook_fn

        # Register hooks
        for layer_name in self.layer_names:
            layer = get_layer_by_string(self.model, layer_name)
            hooks.append(layer.register_forward_hook(make_hook(layer_name)))

        try:
            with torch.no_grad():
                total_batches = None
                if hasattr(self.dataloader, "__len__"):
                    total_batches = len(self.dataloader)
                    if self.config.max_batches is not None:
                        total_batches = min(total_batches, self.config.max_batches)

                progress_bar = tqdm(
                    enumerate(self.dataloader),
                    total=total_batches,
                    desc="Collecting features",
                    dynamic_ncols=True,
                    file=sys.stdout,
                    position=1, 
                    leave=False
                )

                for batch_idx, batch in progress_bar:
                    if self.config.max_batches is not None and batch_idx >= self.config.max_batches:
                        break

                    inputs, targets = self.__unpack_batch(batch)

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device) if targets is not None else None

                    current_batch_features.clear()

                    model_output = self.model(inputs)
                    y = target_extractor(model_output, inputs, targets).to(storage_device)

                    for layer_name in self.layer_names:
                        if layer_name not in current_batch_features:
                            raise RuntimeError(f"No activation captured for layer '{layer_name}'.")

                        feature_storage[layer_name].append(current_batch_features[layer_name])

                    target_storage.append(y)

        finally:
            for hook in hooks:
                hook.remove()

        features_by_layer = {
            layer_name: torch.cat(feature_storage[layer_name], dim=0)
            for layer_name in self.layer_names
        }
        targets = torch.cat(target_storage, dim=0)

        return features_by_layer, targets

    def __default_logits_gap_target_extractor(self, model_output: Any) -> torch.Tensor:
        """
        Default target extractor:
        converts model output into [N, D] using a GAP-like reduction.
        Very practical because it does not assume a specific task label encoding.
        """
        out = self.__extract_first_tensor(model_output)

        if out.dim() == 4:
            # Typical segmentation logits: [N, C, H, W] -> [N, C]
            return out.mean(dim=(2, 3)).detach()
        if out.dim() == 3:
            return out.mean(dim=2).detach()
        if out.dim() == 2:
            return out.detach()

        return out.flatten(start_dim=1).detach()

    def __segmentation_mask_histogram_target_extractor(self) -> Callable[[Any, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Returns a target extractor that uses the GT mask distribution per image.
        For each image, target becomes [p(class0), p(class1), ..., p(classK)].

        This is more task-aware for segmentation.
        """
        def _extractor(model_output: Any, input_batch: torch.Tensor, target_batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            if target_batch is None:
                raise ValueError("Target batch is required for mask histogram target extractor.")

            y = target_batch

            if y.dim() == 4 and y.shape[1] == 1:
                y = y.squeeze(1)

            if y.dim() != 3:
                raise ValueError(f"Expected target mask with shape [N, H, W] or [N, 1, H, W], got {tuple(y.shape)}")

            y = y.long()
            histograms = []

            for class_idx in range(self.config.num_classes):
                class_ratio = (y == class_idx).float().mean(dim=(1, 2))
                histograms.append(class_ratio)

            return torch.stack(histograms, dim=1).detach()

        return _extractor
    
    def __extract_first_tensor(self, data: Any) -> torch.Tensor:
        if torch.is_tensor(data):
            return data

        if isinstance(data, dict):
            if "out" in data and torch.is_tensor(data["out"]):
                return data["out"]

            for value in data.values():
                if torch.is_tensor(value):
                    return value

        if isinstance(data, (list, tuple)):
            for value in data:
                if torch.is_tensor(value):
                    return value

        raise TypeError("Could not extract a torch.Tensor from model output / hook output.")

    def __unpack_batch(self, batch: Any) -> Tuple[torch.torch.Tensor, Optional[torch.torch.Tensor]]:
        if torch.is_tensor(batch):
            return batch, None

        if isinstance(batch, dict):
            if "image" not in batch:
                raise KeyError("Batch dict must contain key 'image'.")

            x = batch["image"]
            y = batch.get("mask", None)

            if not torch.is_tensor(x):
                raise TypeError("Batch['image'] is not a torch.Tensor.")

            if y is not None and not torch.is_tensor(y):
                y = None

            return x, y

        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Empty batch received.")

            x = batch[0]
            y = batch[1] if len(batch) > 1 else None

            if not torch.is_tensor(x):
                raise TypeError("Batch input is not a torch.Tensor.")

            if y is not None and not torch.is_tensor(y):
                y = None

            return x, y

        raise TypeError(f"Unsupported batch type: {type(batch)}")