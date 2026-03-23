from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm
import torch
from torch import nn
from .utils import get_layer_by_string, get_all_conv_layer_names
from .elm import ELMRegressor

@dataclass
class ImportanceProcessorConfig:
    hidden_dim: int = 128
    hidden_dim_per_filter: int = 16
    reg_lambda: float = 1e-3
    activation: str = "tanh"  # tanh | relu | sigmoid
    max_batches: Optional[int] = None
    eps: float = 1e-8
    seed: int = 42
    use_double_for_solver: bool = True
    feature_type = "segmentation" # segmention | logits
    num_classes = 3 # if segmentation
    layer_names = "" # to get importance for all layers, or list (str) to specify the layer names


Tensor = torch.Tensor

class ELMImportanceProcessor:

    def __init__(self, config: ImportanceProcessorConfig, model: nn.Module, dataloader: Iterable):
        self.config = config 
        self.model = model
        self.dataloader = dataloader
        self.layer_names = get_all_conv_layer_names(model) if config.layer_names == "" else config.layer_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__process_feature_and_targets()

    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------
    
    def __process_feature_and_targets(self):
        
        if self.config.feature_type == "segmentation":
            target_extractor = self.__segmentation_mask_histogram_target_extractor()
        else:
            target_extractor = self.__default_logits_gap_target_extractor

        self.features_by_layer, self.targets = self.__collect_features_and_targets(target_extractor)
    
    def __collect_features_and_targets(self, target_extractor: Callable[[Any, Tensor, Optional[Tensor]], Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        self.model.eval()
        self.model = self.model.to(self.device)
        storage_device = "cpu"

        hooks = []
        feature_storage: Dict[str, List[Tensor]] = {layer_name: [] for layer_name in self.layer_names}
        target_storage: List[Tensor] = []

        current_batch_features: Dict[str, Tensor] = {}

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
                    dynamic_ncols=True
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

    def __default_logits_gap_target_extractor(self, model_output: Any) -> Tensor:
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

    def __segmentation_mask_histogram_target_extractor(self) -> Callable[[Any, Tensor, Optional[Tensor]], Tensor]:
        """
        Returns a target extractor that uses the GT mask distribution per image.
        For each image, target becomes [p(class0), p(class1), ..., p(classK)].

        This is more task-aware for segmentation.
        """
        def _extractor(model_output: Any, input_batch: Tensor, target_batch: Optional[Tensor] = None) -> Tensor:
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
    
    def __extract_first_tensor(self, data: Any) -> Tensor:
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

        raise TypeError("Could not extract a tensor from model output / hook output.")

    def __unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if torch.is_tensor(batch):
            return batch, None

        if isinstance(batch, dict):
            if "image" not in batch:
                raise KeyError("Batch dict must contain key 'image'.")

            x = batch["image"]
            y = batch.get("mask", None)

            if not torch.is_tensor(x):
                raise TypeError("Batch['image'] is not a tensor.")

            if y is not None and not torch.is_tensor(y):
                y = None

            return x, y

        if isinstance(batch, (list, tuple)):
            if len(batch) == 0:
                raise ValueError("Empty batch received.")

            x = batch[0]
            y = batch[1] if len(batch) > 1 else None

            if not torch.is_tensor(x):
                raise TypeError("Batch input is not a tensor.")

            if y is not None and not torch.is_tensor(y):
                y = None

            return x, y

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def compute_elm_global_importances(self) -> Dict[str, List[float]]:
        """
        One single ELM trained with features from all selected layers concatenated.
        Importance of each filter = increase in ELM loss when that feature is neutralized.
        """
        
        if len(self.features_by_layer) == 0:
            raise RuntimeError("No features were collected for ELM global importance.")

        X_parts = [self.features_by_layer[layer_name] for layer_name in self.layer_names]
        X = torch.cat(X_parts, dim=1)
        Y = self.targets

        elm_model = ELMRegressor(
            hidden_dim=self.config.hidden_dim,
            reg_lambda=self.config.reg_lambda,
            activation_name=self.config.activation,
            seed=self.config.seed,
            eps=self.config.eps,
            use_double_for_solver=self.config.use_double_for_solver,
        )

        elm_model.fit(X, Y)
        importances = elm_model.compute_ablation_importance(X, Y)

        result: Dict[str, List[float]] = {}
        offset = 0
        for layer_name in tqdm(self.layer_names, desc="ELM global feature ranking processing", dynamic_ncols=True):
            channels = self.features_by_layer[layer_name].shape[1]
            result[layer_name] = importances[offset: offset + channels]
            offset += channels

        return result

    def compute_elm_layerwise_importances(self) -> Dict[str, List[float]]:
        """
        One ELM per layer.
        Importance of each filter = increase in ELM loss when that feature is neutralized.
        """
        result: Dict[str, List[float]] = {}
        Y = self.targets.to(self.device)

        for layer_name in tqdm(self.layer_names, desc="ELM layerwise feature ranking processing", dynamic_ncols=True):
            X = self.features_by_layer[layer_name].to(self.device)

            elm_model = ELMRegressor(
                hidden_dim=self.config.hidden_dim,
                reg_lambda=self.config.reg_lambda,
                activation_name=self.config.activation,
                seed=self.config.seed,
                eps=self.config.eps,
                use_double_for_solver=self.config.use_double_for_solver,
            )

            elm_model.fit(X, Y)
            importances = elm_model.compute_ablation_importance(X, Y)
            result[layer_name] = importances

        return result

    def compute_elm_filterwise_importances(self) -> Dict[str, List[float]]:
        """
        One tiny ELM per filter.
        Importance of one filter = how much that single filter alone reduces target reconstruction loss
        compared with a constant baseline.
        """
        result: Dict[str, List[float]] = {}
        Y = self.targets.to(self.device)
        baseline_loss = ELMImportanceProcessor._constant_baseline_loss(Y)

        for layer_name in tqdm(self.layer_names, desc="ELM filterwise feature ranking processing", dynamic_ncols=True):
            X_layer = self.features_by_layer[layer_name].to(self.device)
            layer_importances: List[float] = []

            for filter_idx in range(X_layer.shape[1]):
                X_filter = X_layer[:, filter_idx:filter_idx + 1]

                elm_model = ELMRegressor(
                    hidden_dim=self.config.hidden_dim_per_filter,
                    reg_lambda=self.config.reg_lambda,
                    activation_name=self.config.activation,
                    seed=self.config.seed + filter_idx,
                    eps=self.config.eps,
                    use_double_for_solver=self.config.use_double_for_solver,
                )

                elm_model.fit(X_filter, Y)
                pred = elm_model.predict(X_filter)
                filter_loss = elm_model.calculate_loss(pred, Y).item()

                # Higher reduction => more important
                importance = max(baseline_loss - filter_loss, 0.0)
                layer_importances.append(float(importance))

            result[layer_name] = layer_importances

        return result

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    @staticmethod
    def _constant_baseline_loss(Y: Tensor) -> float:
        mean_pred = Y.mean(dim=0, keepdim=True)
        return float(torch.mean((Y - mean_pred) ** 2).item())

    @staticmethod
    def discover_conv2d_layers(model: nn.Module, only_decoder: bool = False) -> List[str]:
        result = []

        root_module = model.decoder if only_decoder and hasattr(model, "decoder") else model

        for name, module in root_module.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                result.append(name)

        return result