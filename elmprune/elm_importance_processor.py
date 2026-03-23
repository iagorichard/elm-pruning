from typing import Dict, Iterable, List
from tqdm.auto import tqdm
import torch
from torch import nn
from .utils import get_all_conv_layer_names, compute_constant_baseline_loss
from .elm import ELMRegressor
from .importance_processor_config import ImportanceProcessorConfig
from .feature_extractor import FeatureExtractor


class ELMImportanceProcessor:

    def __init__(self, config: ImportanceProcessorConfig, model: nn.Module, dataloader: Iterable):
        self.config = config 
        self.layer_names = get_all_conv_layer_names(model) if config.layer_names == "" else config.layer_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        feature_extractor = FeatureExtractor(config, model, dataloader, self.layer_names)
        self.features_by_layer, self.targets = feature_extractor.extract_feature_and_targets()

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
        baseline_loss = compute_constant_baseline_loss(Y)

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