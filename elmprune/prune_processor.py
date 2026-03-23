from typing import List, Iterable, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch_pruning as tp

from .utils import get_layer_by_string, get_first_dataloader_image, clone_model


class PruneProcessor:

    def __init__(
        self,
        model: nn.Module,
        importances: Dict[str, List[float]],
        percentage: float,
        dataloader: Iterable
    ):
        self.model = clone_model(model)
        self.importances = importances
        self.percentage = percentage
        self.input_example = get_first_dataloader_image(dataloader).to(torch.device("cpu"))

    def execute(self):
        selected_filters_by_layer = self.__select_global_filters_to_prune()

        total_selected = sum(len(idxs) for idxs in selected_filters_by_layer.values())
        print(
            f"Global pruning selection: {total_selected} filters selected "
            f"({self.percentage * 100:.0f}% of candidate filters)."
        )

        for layer_name in self.__get_candidate_layer_names():
            if layer_name not in selected_filters_by_layer:
                continue

            self.__prune_model_layer_by_indices(
                layer_name=layer_name,
                filter_idxs=selected_filters_by_layer[layer_name]
            )

        return self.model

    def __get_candidate_layer_names(self) -> List[str]:
        """
        Assumes the first and last entries in the importance dict
        should not be pruned.
        """
        return list(self.importances.keys())[1:-1]

    def __select_global_filters_to_prune(self) -> Dict[str, List[int]]:
        """
        Select the lowest-importance filters globally across all candidate layers.
        Returns a dict:
            {
                "layer_name": [filter_idx_1, filter_idx_2, ...]
            }
        """
        global_candidates: List[Tuple[float, str, int]] = []

        for layer_name in self.__get_candidate_layer_names():
            layer_importances = self.importances[layer_name]

            for filter_idx, importance in enumerate(layer_importances):
                global_candidates.append((importance, layer_name, filter_idx))

        global_candidates.sort(key=lambda x: x[0], reverse=False)

        desired_quantity = int(len(global_candidates) * self.percentage)
        desired_quantity = min(desired_quantity, len(global_candidates))

        selected = defaultdict(list)

        for _, layer_name, filter_idx in global_candidates[:desired_quantity]:
            selected[layer_name].append(filter_idx)

        for layer_name in selected:
            selected[layer_name] = sorted(set(selected[layer_name]))

        return dict(selected)

    def __prune_model_layer_by_indices(self, layer_name: str, filter_idxs: List[int]):
        try:
            layer = get_layer_by_string(self.model, layer_name)

            if not isinstance(layer, nn.Conv2d):
                print(f"Skipping {layer_name}: layer is not Conv2d.")
                return

            current_out_channels = layer.out_channels

            if current_out_channels <= 1:
                print(f"Skipping {layer_name}: layer has {current_out_channels} output channel(s).")
                return

            valid_filter_idxs = sorted(set(i for i in filter_idxs if i < current_out_channels))

            # Never allow pruning all output channels of the layer
            if len(valid_filter_idxs) >= current_out_channels:
                valid_filter_idxs = valid_filter_idxs[:current_out_channels - 1]

            if not valid_filter_idxs:
                print(f"Skipping {layer_name}: no valid filters to prune.")
                return

            DG = tp.DependencyGraph()
            DG.build_dependency(self.model, example_inputs=self.input_example)

            pruning_plan = DG.get_pruning_plan(
                layer,
                tp.prune_conv,
                idxs=valid_filter_idxs
            )

            pruning_plan.exec()

            self.model.zero_grad()
            with torch.no_grad():
                self.model(self.input_example)

            print(
                f"Prune of layer {layer_name} done with success! "
                f"Pruned filters: {len(valid_filter_idxs)} / {current_out_channels}"
            )

        except Exception as e:
            print(
                f"Exception in trying to prune | Layer {layer_name} "
                f"with selected filters {len(filter_idxs)}: {e}"
            )