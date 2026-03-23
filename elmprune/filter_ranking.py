import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tqdm.auto import tqdm
import torch
from torch import nn
from .utils import get_layer_by_string

@dataclass
class ELMConfig:
    hidden_dim: int = 128
    hidden_dim_per_filter: int = 16
    reg_lambda: float = 1e-3
    activation: str = "tanh"  # tanh | relu | sigmoid
    max_batches: Optional[int] = None
    eps: float = 1e-8
    seed: int = 42
    use_double_for_solver: bool = True
    storage_device: str = "cpu"  # safer for hooks; fitting still happens on GPU


Tensor = torch.Tensor

class ELMImportanceService:
    """
    Builds pruning importances compatible with:

        {
            "layer_name": [importance_0, importance_1, ...]
        }

    Higher score => more important
    Lower score  => more prunable
    """

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_elm_global_importances(
        model: nn.Module,
        dataloader: Iterable,
        layer_names: List[str],
        device: torch.device,
        config: Optional[ELMConfig] = None,
        target_extractor: Optional[Callable[[Any, Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """
        One single ELM trained with features from all selected layers concatenated.
        Importance of each filter = increase in ELM loss when that feature is neutralized.
        """
        config = config or ELMConfig()
        target_extractor = target_extractor or ELMImportanceService.default_logits_gap_target_extractor

        features_by_layer, targets = ELMImportanceService._collect_features_and_targets(
            model=model,
            dataloader=dataloader,
            layer_names=layer_names,
            device=device,
            config=config,
            target_extractor=target_extractor,
        )

        if len(features_by_layer) == 0:
            raise RuntimeError("No features were collected for ELM global importance.")

        X_parts = [features_by_layer[layer_name] for layer_name in layer_names]
        X = torch.cat(X_parts, dim=1)
        Y = targets

        elm_model = ELMImportanceService._fit_elm_regressor(
            X=X,
            Y=Y,
            hidden_dim=config.hidden_dim,
            reg_lambda=config.reg_lambda,
            activation_name=config.activation,
            seed=config.seed,
            eps=config.eps,
            device=device,
            use_double_for_solver=config.use_double_for_solver,
        )

        importances_flat = ELMImportanceService._compute_ablation_importance(
            elm_model=elm_model,
            X=X.to(device),
            Y=Y.to(device),
        )

        result: Dict[str, List[float]] = {}
        offset = 0
        for layer_name in tqdm(layer_names, desc="ELM global feature ranking processing", dynamic_ncols=True):
            channels = features_by_layer[layer_name].shape[1]
            result[layer_name] = importances_flat[offset: offset + channels]
            offset += channels

        return result

    @staticmethod
    def compute_elm_layerwise_importances(
        model: nn.Module,
        dataloader: Iterable,
        layer_names: List[str],
        device: torch.device,
        config: Optional[ELMConfig] = None,
        target_extractor: Optional[Callable[[Any, Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """
        One ELM per layer.
        Importance of each filter = increase in ELM loss when that feature is neutralized.
        """
        config = config or ELMConfig()
        target_extractor = target_extractor or ELMImportanceService.default_logits_gap_target_extractor

        features_by_layer, targets = ELMImportanceService._collect_features_and_targets(
            model=model,
            dataloader=dataloader,
            layer_names=layer_names,
            device=device,
            config=config,
            target_extractor=target_extractor,
        )

        result: Dict[str, List[float]] = {}
        Y = targets.to(device)

        for layer_name in tqdm(layer_names, desc="ELM layerwise feature ranking processing", dynamic_ncols=True):
            X = features_by_layer[layer_name].to(device)

            elm_model = ELMImportanceService._fit_elm_regressor(
                X=X,
                Y=Y,
                hidden_dim=config.hidden_dim,
                reg_lambda=config.reg_lambda,
                activation_name=config.activation,
                seed=config.seed,
                eps=config.eps,
                device=device,
                use_double_for_solver=config.use_double_for_solver,
            )

            importances = ELMImportanceService._compute_ablation_importance(
                elm_model=elm_model,
                X=X,
                Y=Y,
            )
            result[layer_name] = importances

        return result

    @staticmethod
    def compute_elm_filterwise_importances(
        model: nn.Module,
        dataloader: Iterable,
        layer_names: List[str],
        device: torch.device,
        config: Optional[ELMConfig] = None,
        target_extractor: Optional[Callable[[Any, Tensor, Optional[Tensor]], Tensor]] = None,
    ) -> Dict[str, List[float]]:
        """
        One tiny ELM per filter.
        Importance of one filter = how much that single filter alone reduces target reconstruction loss
        compared with a constant baseline.

        This is the most expensive of the three approaches.
        """
        config = config or ELMConfig()
        target_extractor = target_extractor or ELMImportanceService.default_logits_gap_target_extractor

        features_by_layer, targets = ELMImportanceService._collect_features_and_targets(
            model=model,
            dataloader=dataloader,
            layer_names=layer_names,
            device=device,
            config=config,
            target_extractor=target_extractor,
        )

        Y = targets.to(device)
        baseline_loss = ELMImportanceService._constant_baseline_loss(Y)

        result: Dict[str, List[float]] = {}

        for layer_name in tqdm(layer_names, desc="ELM filterwise feature ranking processing", dynamic_ncols=True):
            X_layer = features_by_layer[layer_name].to(device)
            layer_importances: List[float] = []

            for filter_idx in range(X_layer.shape[1]):
                X_filter = X_layer[:, filter_idx:filter_idx + 1]

                elm_model = ELMImportanceService._fit_elm_regressor(
                    X=X_filter,
                    Y=Y,
                    hidden_dim=config.hidden_dim_per_filter,
                    reg_lambda=config.reg_lambda,
                    activation_name=config.activation,
                    seed=config.seed + filter_idx,
                    eps=config.eps,
                    device=device,
                    use_double_for_solver=config.use_double_for_solver,
                )

                pred = ELMImportanceService._predict_elm(elm_model, X_filter)
                filter_loss = ELMImportanceService._mse(pred, Y).item()

                # Higher reduction => more important
                importance = max(baseline_loss - filter_loss, 0.0)
                layer_importances.append(float(importance))

            result[layer_name] = layer_importances

        return result

    # -------------------------------------------------------------------------
    # Target extractors
    # -------------------------------------------------------------------------

    @staticmethod
    def default_logits_gap_target_extractor(
        model_output: Any,
        input_batch: Tensor,
        target_batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Default target extractor:
        converts model output into [N, D] using a GAP-like reduction.
        Very practical because it does not assume a specific task label encoding.
        """
        out = ELMImportanceService._extract_first_tensor(model_output)

        if out.dim() == 4:
            # Typical segmentation logits: [N, C, H, W] -> [N, C]
            return out.mean(dim=(2, 3)).detach()
        if out.dim() == 3:
            return out.mean(dim=2).detach()
        if out.dim() == 2:
            return out.detach()

        return out.flatten(start_dim=1).detach()

    @staticmethod
    def segmentation_mask_histogram_target_extractor(
        num_classes: int,
    ) -> Callable[[Any, Tensor, Optional[Tensor]], Tensor]:
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

            for class_idx in range(num_classes):
                class_ratio = (y == class_idx).float().mean(dim=(1, 2))
                histograms.append(class_ratio)

            return torch.stack(histograms, dim=1).detach()

        return _extractor

    # -------------------------------------------------------------------------
    # Feature collection
    # -------------------------------------------------------------------------

    @staticmethod
    def _collect_features_and_targets(
        model: nn.Module,
        dataloader: Iterable,
        layer_names: List[str],
        device: torch.device,
        config: ELMConfig,
        target_extractor: Callable[[Any, Tensor, Optional[Tensor]], Tensor],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        model.eval()
        model = model.to(device)

        storage_device = torch.device(config.storage_device)

        hooks = []
        feature_storage: Dict[str, List[Tensor]] = {layer_name: [] for layer_name in layer_names}
        target_storage: List[Tensor] = []

        current_batch_features: Dict[str, Tensor] = {}

        def make_hook(layer_name: str):
            def hook_fn(module, inputs, output):
                out = ELMImportanceService._extract_first_tensor(output)

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
        for layer_name in layer_names:
            layer = get_layer_by_string(model, layer_name)
            hooks.append(layer.register_forward_hook(make_hook(layer_name)))

        try:
            with torch.no_grad():
                total_batches = None
                if hasattr(dataloader, "__len__"):
                    total_batches = len(dataloader)
                    if config.max_batches is not None:
                        total_batches = min(total_batches, config.max_batches)

                progress_bar = tqdm(
                    enumerate(dataloader),
                    total=total_batches,
                    desc="Collecting features",
                    dynamic_ncols=True
                )

                for batch_idx, batch in progress_bar:
                    if config.max_batches is not None and batch_idx >= config.max_batches:
                        break

                    inputs, targets = ELMImportanceService._unpack_batch(batch)

                    inputs = inputs.to(device)
                    targets = targets.to(device) if targets is not None else None

                    current_batch_features.clear()

                    model_output = model(inputs)
                    y = target_extractor(model_output, inputs, targets).to(storage_device)

                    for layer_name in layer_names:
                        if layer_name not in current_batch_features:
                            raise RuntimeError(f"No activation captured for layer '{layer_name}'.")

                        feature_storage[layer_name].append(current_batch_features[layer_name])

                    target_storage.append(y)

        finally:
            for hook in hooks:
                hook.remove()

        features_by_layer = {
            layer_name: torch.cat(feature_storage[layer_name], dim=0)
            for layer_name in layer_names
        }
        targets = torch.cat(target_storage, dim=0)

        return features_by_layer, targets

    # -------------------------------------------------------------------------
    # ELM core
    # -------------------------------------------------------------------------

    @staticmethod
    def _fit_elm_regressor(
        X: Tensor,
        Y: Tensor,
        hidden_dim: int,
        reg_lambda: float,
        activation_name: str,
        seed: int,
        eps: float,
        device: torch.device,
        use_double_for_solver: bool,
    ) -> Dict[str, Tensor]:
        X = X.to(device)
        Y = Y.to(device)

        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        Xn = (X - X_mean) / X_std

        Y_mean = Y.mean(dim=0, keepdim=True)
        Yn = Y - Y_mean

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        in_dim = Xn.shape[1]
        out_dim = Yn.shape[1]

        W = torch.randn((in_dim, hidden_dim), generator=generator, device=device) / math.sqrt(max(in_dim, 1))
        b = torch.randn((hidden_dim,), generator=generator, device=device)

        H = ELMImportanceService._apply_activation(Xn @ W + b, activation_name)

        I = torch.eye(hidden_dim, device=device, dtype=H.dtype)
        lhs = H.T @ H + reg_lambda * I
        rhs = H.T @ Yn

        if use_double_for_solver:
            beta = torch.linalg.solve(lhs.double(), rhs.double()).to(H.dtype)
        else:
            beta = torch.linalg.solve(lhs, rhs)

        return {
            "W": W,
            "b": b,
            "beta": beta,
            "X_mean": X_mean,
            "X_std": X_std,
            "Y_mean": Y_mean,
            "activation_name": activation_name,
        }

    @staticmethod
    def _predict_elm(elm_model: Dict[str, Tensor], X: Tensor) -> Tensor:
        X = X.to(elm_model["W"].device)

        Xn = (X - elm_model["X_mean"]) / elm_model["X_std"]
        H = ELMImportanceService._apply_activation(
            Xn @ elm_model["W"] + elm_model["b"],
            elm_model["activation_name"],
        )
        return H @ elm_model["beta"] + elm_model["Y_mean"]

    @staticmethod
    def _compute_ablation_importance(
        elm_model: Dict[str, Tensor],
        X: Tensor,
        Y: Tensor,
    ) -> List[float]:
        """
        Importance = increase in ELM loss when feature is neutralized to its mean value.
        Since pruning removes less important filters, low scores should be pruned first.
        """
        X = X.to(elm_model["W"].device)
        Y = Y.to(elm_model["W"].device)

        base_pred = ELMImportanceService._predict_elm(elm_model, X)
        base_loss = ELMImportanceService._mse(base_pred, Y).item()

        importances: List[float] = []
        X_work = X.clone()

        for feature_idx in range(X.shape[1]):
            original_column = X_work[:, feature_idx].clone()

            # Neutralize feature by sending it to its mean value
            X_work[:, feature_idx] = elm_model["X_mean"][0, feature_idx]

            ablated_pred = ELMImportanceService._predict_elm(elm_model, X_work)
            ablated_loss = ELMImportanceService._mse(ablated_pred, Y).item()

            importance = max(ablated_loss - base_loss, 0.0)
            importances.append(float(importance))

            X_work[:, feature_idx] = original_column

        return importances

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_first_tensor(data: Any) -> Tensor:
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

    @staticmethod
    def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    @staticmethod
    def _apply_activation(x: Tensor, activation_name: str) -> Tensor:
        activation_name = activation_name.lower()

        if activation_name == "tanh":
            return torch.tanh(x)
        if activation_name == "relu":
            return torch.relu(x)
        if activation_name == "sigmoid":
            return torch.sigmoid(x)

        raise ValueError(f"Unsupported activation: {activation_name}")

    @staticmethod
    def _mse(pred: Tensor, target: Tensor) -> Tensor:
        return torch.mean((pred - target) ** 2)

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