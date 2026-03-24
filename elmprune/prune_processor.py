from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import torch.nn as nn
import torch_pruning as tp
from .utils import clone_model, get_first_dataloader_image, build_name_to_module

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
        name_to_module = build_name_to_module(self.model)
        candidate_layer_names = self.__get_candidate_layer_names()
        selected_by_name = self.__select_global_filter_indices(candidate_layer_names, name_to_module)

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

        ignored_layers = self.__get_ignored_layers(name_to_module)

        # Build DG once
        DG = tp.DependencyGraph().build_dependency(
            self.model,
            example_inputs=self.example_inputs,
        )

        # Walk all root groups once, sequentially
        for group in DG.get_all_groups(
            ignored_layers=ignored_layers,
            root_module_types=[nn.Conv2d],
        ):
            dep, _ = group[0]
            root_module = dep.target.module

            root_name = self.__find_module_name(name_to_module, root_module)
            if root_name is None:
                continue

            selected_idxs = selected_by_name.get(root_name)
            if not selected_idxs:
                continue

            current_out = getattr(root_module, "out_channels", None)
            if current_out is None or current_out <= 1:
                continue

            valid_idxs = sorted(set(i for i in selected_idxs if i < current_out))

            # never prune all channels
            if len(valid_idxs) >= current_out:
                valid_idxs = valid_idxs[: current_out - 1]

            if not valid_idxs:
                continue

            # optional rounding safeguard
            if self.round_to is not None and self.round_to > 1:
                remaining = current_out - len(valid_idxs)
                rounded_remaining = max(self.round_to, (remaining // self.round_to) * self.round_to)
                max_prunable = current_out - rounded_remaining
                if max_prunable <= 0:
                    continue
                valid_idxs = valid_idxs[:max_prunable]
                if not valid_idxs:
                    continue

            group.prune(idxs=valid_idxs)

            if self.verbose:
                print(f"Pruned {root_name}: {len(valid_idxs)} / {current_out}")

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

        for layer_name in selected:
            selected[layer_name] = sorted(set(selected[layer_name]))

        return dict(selected)

    def __get_ignored_layers(self, name_to_module: Dict[str, nn.Module]) -> List[nn.Module]:
        ignored = []

        # ignore first and last by dict order if requested
        names = list(self.importances.keys())
        if self.ignore_first_and_last_by_dict_order and len(names) >= 2:
            first_name = names[0]
            last_name = names[-1]

            if first_name in name_to_module:
                ignored.append(name_to_module[first_name])
            if last_name in name_to_module:
                ignored.append(name_to_module[last_name])

        return ignored

    def __find_module_name(self, name_to_module: Dict[str, nn.Module], target_module: nn.Module) -> Optional[str]:
        for name, module in name_to_module.items():
            if module is target_module:
                return name
        return None