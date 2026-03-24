import copy
import torch
import torch.nn as nn
import torch_pruning as tp
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch_pruning as tp
from .utils import clone_model, build_name_to_module

class PruneProcessor:
    """
    percentage = fração dos canais candidatos globais a remover.
    Ex.: 0.2 => remove os 20% menos importantes entre todos os canais candidatos.
    """

    def __init__(
        self,
        model: nn.Module,
        importances: Dict[str, List[float]],
        percentage: float,
        example_inputs: Any,
        ignore_first_and_last_by_dict_order: bool = True,
        round_to: Optional[int] = None,
        verbose: bool = True,
    ):
        if not (0.0 <= percentage < 1.0):
            raise ValueError("percentage must be in [0.0, 1.0).")

        self.model = clone_model(model)
        self.importances = importances
        self.percentage = percentage
        self.example_inputs = example_inputs
        self.ignore_first_and_last_by_dict_order = ignore_first_and_last_by_dict_order
        self.round_to = round_to
        self.verbose = verbose

    def execute(self) -> nn.Module:
        min_remaining_channels = 3

        name_to_module = build_name_to_module(self.model)
        candidate_layer_names = self.__get_candidate_layer_names()
        selected_by_name = self.__select_global_filter_indices(
            candidate_layer_names,
            name_to_module
        )

        if self.verbose:
            total_selected = sum(len(v) for v in selected_by_name.values())
            total_candidates = sum(
                min(len(self.importances[name]), getattr(name_to_module[name], "out_channels", 0))
                for name in candidate_layer_names
                if isinstance(name_to_module.get(name), nn.Conv2d)
            )
            print(
                f"Selected {total_selected} / {total_candidates} candidate channels "
                f"({self.percentage * 100:.2f}%)."
            )

        ordered_layer_names = [
            name for name, module in self.model.named_modules()
            if name in selected_by_name and isinstance(module, nn.Conv2d)
        ]

        if self.ignore_first_and_last_by_dict_order and len(ordered_layer_names) >= 3:
            ordered_layer_names = ordered_layer_names[1:-1]

        for layer_name in ordered_layer_names:
            current_mapping = build_name_to_module(self.model)
            root_module = current_mapping.get(layer_name)

            if not isinstance(root_module, nn.Conv2d):
                continue

            current_out = getattr(root_module, "out_channels", None)
            if current_out is None or current_out <= min_remaining_channels:
                continue

            selected_idxs = selected_by_name.get(layer_name, [])
            selected_idxs = [i for i in selected_idxs if i < current_out]

            if not selected_idxs:
                continue

            max_prunable = current_out - min_remaining_channels
            if max_prunable <= 0:
                continue

            selected_idxs = selected_idxs[:max_prunable]
            if not selected_idxs:
                continue

            if self.round_to is not None and self.round_to > 1:
                remaining = current_out - len(selected_idxs)
                rounded_remaining = max(
                    self.round_to,
                    (remaining // self.round_to) * self.round_to
                )
                max_prunable = current_out - rounded_remaining
                if max_prunable <= 0:
                    continue
                selected_idxs = selected_idxs[:max_prunable]
                if not selected_idxs:
                    continue

            pruned = False
            try_sizes = []
            n = len(selected_idxs)

            while n > 0:
                try_sizes.append(n)
                n = n // 2

            for try_n in try_sizes:
                trial_model = copy.deepcopy(self.model)
                trial_mapping = build_name_to_module(trial_model)
                trial_root = trial_mapping.get(layer_name)

                if not isinstance(trial_root, nn.Conv2d):
                    continue

                idxs_to_try = selected_idxs[:try_n]

                DG = tp.DependencyGraph().build_dependency(
                    trial_model,
                    example_inputs=self.example_inputs,
                )

                try:
                    group = DG.get_pruning_group(
                        trial_root,
                        tp.prune_conv_out_channels,
                        idxs=idxs_to_try
                    )

                    if not DG.check_pruning_group(group):
                        continue

                    group.prune()

                    trial_model.eval()
                    with torch.no_grad():
                        _ = trial_model(self.example_inputs)

                    self.model = trial_model
                    pruned = True

                    if self.verbose:
                        print(f"Pruned {layer_name}: {try_n} / {current_out}")

                    break

                except Exception:
                    continue

            if not pruned and self.verbose:
                print(f"Skipped {layer_name}: no valid fallback pruning found")

        return self.model

    
    def __get_candidate_layer_names(self) -> List[str]:
        names = list(self.importances.keys())
        if self.ignore_first_and_last_by_dict_order and len(names) >= 3:
            names = names[1:-1]
        return names

    def __select_global_filter_indices(
        self,
        candidate_layer_names: List[str],
        name_to_module: Dict[str, nn.Module],
        ) -> Dict[str, List[int]]:
        global_candidates: List[Tuple[float, str, int]] = []

        for layer_name in candidate_layer_names:
            module = name_to_module.get(layer_name)
            if not isinstance(module, nn.Conv2d):
                continue

            max_valid = min(len(self.importances[layer_name]), module.out_channels)
            for idx in range(max_valid):
                score = float(self.importances[layer_name][idx])
                global_candidates.append((score, layer_name, idx))

        global_candidates.sort(key=lambda x: x[0])  # lowest importance first

        k = int(len(global_candidates) * self.percentage)
        selected = defaultdict(list)

        for _, layer_name, idx in global_candidates[:k]:
            selected[layer_name].append(idx)

        # preserve importance order; just remove duplicates
        for layer_name in selected:
            selected[layer_name] = list(dict.fromkeys(selected[layer_name]))

        return dict(selected)