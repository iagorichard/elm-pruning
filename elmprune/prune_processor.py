from typing import List, Iterable, Dict
import torch
import torch.nn as nn
import torch_pruning as tp
from .utils import get_layer_by_string, get_first_dataloader_image, clone_model

class PruneProcessor:

    def __init__(self, model: nn.Module, importances: Dict[str, List[float]], percentage: float, dataloader: Iterable):
        self.model = clone_model(model)

        self.importances = importances
        self.percentage = percentage
        self.input_example = get_first_dataloader_image(dataloader)
        self.input_example = self.input_example.to(torch.device("cpu"))

    def execute(self):
        for layer_name in list(self.importances.keys())[1:-1]:
            self.__prune_model_layer_by_importance_percentage(layer_name)
        
        return self.model

    def __prune_model_layer_by_importance_percentage(self, layer_name: str):
        try:
            desired_quantity = int(len(self.importances[layer_name]) * self.percentage)
            layer = get_layer_by_string(self.model, layer_name)
            filter_idxs = self.__get_indices_to_prune_by_quantity(self.importances[layer_name], desired_quantity)
            
            DG = tp.DependencyGraph()
            DG.build_dependency(self.model, example_inputs=self.input_example)

            pruning_plan = DG.get_pruning_plan(layer, tp.prune_conv, idxs=filter_idxs)

            pruning_plan.exec()

            self.model.zero_grad()
            self.model(self.input_example)

            print(f"Prune of the layer {layer_name} with percentage {self.percentage*100:.0f}% done with success!")

        except Exception as e:
            print(f"Exception in trying to prune | Layer {layer_name} with percentage {self.percentage*100:.0f}%: {e}")
        
    def __get_indices_to_prune_by_quantity(self, importances: List, quantity: int) -> List:
        total = len(importances)
        candidate = min(total, quantity)
        indices_sorted = sorted(
            range(total),
            key=lambda i: importances[i],
            reverse=False
        )
        return indices_sorted[:candidate]
