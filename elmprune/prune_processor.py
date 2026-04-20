import copy
import torch
import torch.nn as nn
import torch_pruning as tp

from collections import defaultdict
from math import ceil, floor
from typing import Dict, List, Any

from .utils import build_name_to_module, count_trainable_params, rank_normalize
from .prune_config import PruneConfig, PruneVerboseLevel


class PruneProcessor:
    def __init__(
        self,
        model: nn.Module,
        importances: Dict[str, List[float]],
        example_inputs: Any,
        config: PruneConfig,
    ):
        self.model = copy.deepcopy(model).to(torch.device("cpu")).eval()
        self.importances_original = importances
        self.importances_live = copy.deepcopy(importances)
        self.example_inputs = self._move_to_cpu(example_inputs)
        self.config = config

        # Mantém rastreabilidade dos índices originais ainda vivos em cada camada.
        self.active_original_indices = {
            layer_name: list(range(len(scores)))
            for layer_name, scores in self.importances_live.items()
        }

        self.base_name_to_module = build_name_to_module(self.model)
        self.base_out_channels = {
            name: module.out_channels
            for name, module in self.base_name_to_module.items()
            if isinstance(module, nn.Conv2d)
        }

        self.protected_layers = self._build_protected_layers()

    def execute(self) -> nn.Module:
        base_params = count_trainable_params(self.model)
        target_params = int(base_params * (1.0 - self.config.target_param_reduction))

        if self.config.verbose.value == PruneVerboseLevel.ALL:
            print(f"[PRUNE] base params   : {base_params}")
            print(f"[PRUNE] target params : {target_params}")

        last_params = base_params

        while True:
            current_params = count_trainable_params(self.model)
            if current_params <= target_params:
                break

            selected_by_name = self._select_indices_for_one_step()
            if not selected_by_name:
                if self.config.verbose.value == PruneVerboseLevel.ALL:
                    print("[PRUNE] no more valid selections.")
                break

            changed = False
            ordered_layer_names = [
                name for name, module in self.model.named_modules()
                if name in selected_by_name and isinstance(module, nn.Conv2d)
            ]

            for layer_name in ordered_layer_names:
                idxs = selected_by_name.get(layer_name, [])
                if not idxs:
                    continue

                trial_model = copy.deepcopy(self.model).to(torch.device("cpu")).eval()
                trial_mapping = build_name_to_module(trial_model)
                trial_root = trial_mapping.get(layer_name)

                if not isinstance(trial_root, nn.Conv2d):
                    continue

                try:
                    DG = tp.DependencyGraph().build_dependency(
                        trial_model,
                        example_inputs=self.example_inputs,
                    )

                    group = DG.get_pruning_group(
                        trial_root,
                        tp.prune_conv_out_channels,
                        idxs=idxs
                    )

                    if not DG.check_pruning_group(group):
                        continue

                    group.prune()

                    with torch.no_grad():
                        _ = trial_model(self.example_inputs)

                    new_params = count_trainable_params(trial_model)

                    if new_params < current_params:
                        pruned_original_idxs = self._map_local_to_original_indices(layer_name, idxs)

                        self.model = trial_model
                        self._remove_pruned_indices_from_live_importance(layer_name, idxs)
                        changed = True

                        if self.config.verbose.value == PruneVerboseLevel.ALL:
                            print(
                                f"[PRUNE] {layer_name}: -{len(idxs)} ch | "
                                f"params {current_params} -> {new_params} | "
                                f"local idxs {idxs} | original idxs {pruned_original_idxs}"
                            )

                        # IMPORTANTE:
                        # Assim que uma poda dá certo, interrompemos a passada atual
                        # para recalcular a seleção com o modelo e as importâncias atualizados.
                        break

                except Exception as ex:
                    if self.config.verbose.value >= PruneVerboseLevel.BASIC.value:
                        print(f"[PRUNE] skipped {layer_name}: {ex}")
                    continue

            new_current_params = count_trainable_params(self.model)

            if not changed or new_current_params >= last_params:
                if self.config.verbose.value == PruneVerboseLevel.ALL:
                    print("[PRUNE] stalled.")
                break

            last_params = new_current_params

            if new_current_params <= target_params:
                break

        if self.config.verbose.value >= PruneVerboseLevel.BASIC.value:
            print("[PRUNE] Prune finished!")

        return self.model

    def _move_to_cpu(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(torch.device("cpu"))
        if isinstance(x, tuple):
            return tuple(self._move_to_cpu(v) for v in x)
        if isinstance(x, list):
            return [self._move_to_cpu(v) for v in x]
        if isinstance(x, dict):
            return {k: self._move_to_cpu(v) for k, v in x.items()}
        return x

    def _build_protected_layers(self) -> set[str]:
        conv_names = [
            name for name, module in self.model.named_modules()
            if isinstance(module, nn.Conv2d)
        ]

        protected = set()
        if conv_names:
            protected.add(conv_names[0])  # first conv

        for name, module in self.model.named_modules():
            lname = name.lower()

            if not isinstance(module, nn.Conv2d):
                continue

            if module.out_channels <= 32:
                protected.add(name)

            if module.out_channels <= 3:
                protected.add(name)

            if any(key in lname for key in [
                "segmentation_head", "classifier", "logits",
                "downsample", "shortcut", "skip", "proj"
            ]):
                protected.add(name)

        return protected

    def _allowed_prunes_for_layer(self, layer_name: str, current_out: int) -> int:
        base_out = self.base_out_channels[layer_name]

        min_keep = max(
            self.config.min_channels_abs,
            ceil(base_out * self.config.min_keep_ratio),
        )

        max_total_prune = floor(base_out * self.config.max_layer_prune_ratio)
        already_pruned = base_out - current_out
        remaining_total_budget = max_total_prune - already_pruned

        step_budget = max(1, floor(current_out * self.config.per_step_layer_ratio))

        allowed = min(
            current_out - min_keep,
            remaining_total_budget,
            step_budget,
        )

        if self.config.round_to and self.config.round_to > 1:
            remaining_after = current_out - allowed
            rounded_remaining = max(
                self.config.round_to,
                (remaining_after // self.config.round_to) * self.config.round_to
            )
            allowed = min(allowed, current_out - rounded_remaining)

        return max(0, allowed)

    def _select_indices_for_one_step(self) -> Dict[str, List[int]]:
        name_to_module = build_name_to_module(self.model)

        candidate_layers = []
        for layer_name, scores in self.importances_live.items():
            module = name_to_module.get(layer_name)
            if not isinstance(module, nn.Conv2d):
                continue
            if layer_name in self.protected_layers:
                continue
            if layer_name not in self.base_out_channels:
                continue
            if len(scores) == 0:
                continue
            candidate_layers.append(layer_name)

        if self.config.selection_scope == "local":
            return self._select_local(candidate_layers, name_to_module)

        return self._select_global(candidate_layers, name_to_module)

    def _select_local(
        self,
        candidate_layers: List[str],
        name_to_module: Dict[str, nn.Module],
    ) -> Dict[str, List[int]]:
        selected = {}

        for layer_name in candidate_layers:
            module = name_to_module[layer_name]
            current_out = module.out_channels
            allowed = self._allowed_prunes_for_layer(layer_name, current_out)
            if allowed <= 0:
                continue

            max_valid = min(current_out, len(self.importances_live[layer_name]))
            scores = self.importances_live[layer_name][:max_valid]

            order = sorted(range(max_valid), key=lambda i: float(scores[i]))
            idxs = order[:allowed]

            if idxs:
                selected[layer_name] = idxs

        return selected

    def _select_global(
        self,
        candidate_layers: List[str],
        name_to_module: Dict[str, nn.Module],
    ) -> Dict[str, List[int]]:
        global_candidates = []

        for layer_name in candidate_layers:
            module = name_to_module[layer_name]
            current_out = module.out_channels
            allowed = self._allowed_prunes_for_layer(layer_name, current_out)
            if allowed <= 0:
                continue

            max_valid = min(current_out, len(self.importances_live[layer_name]))
            scores = self.importances_live[layer_name][:max_valid]

            # Só elm_global deveria cair aqui.
            # Se insistir em usar global com outros modos, normalize por rank.
            if self.config.importance_type != "elm_global":
                scores = rank_normalize(scores)

            order = sorted(range(max_valid), key=lambda i: float(scores[i]))
            for idx in order[:allowed]:
                global_candidates.append((float(scores[idx]), layer_name, idx))

        global_candidates.sort(key=lambda x: x[0])

        # passo global pequeno
        step_global = max(1, int(len(global_candidates) * 0.20))
        chosen = global_candidates[:step_global]

        selected = defaultdict(list)
        for _, layer_name, idx in chosen:
            selected[layer_name].append(idx)

        return dict(selected)

    def _remove_pruned_indices_from_live_importance(self, layer_name: str, pruned_local_idxs: List[int]) -> None:
        if layer_name not in self.importances_live:
            return

        if len(pruned_local_idxs) == 0:
            return

        remove_set = set(pruned_local_idxs)

        self.importances_live[layer_name] = [
            score
            for i, score in enumerate(self.importances_live[layer_name])
            if i not in remove_set
        ]

        if layer_name in self.active_original_indices:
            self.active_original_indices[layer_name] = [
                original_idx
                for i, original_idx in enumerate(self.active_original_indices[layer_name])
                if i not in remove_set
            ]

    def _map_local_to_original_indices(self, layer_name: str, local_idxs: List[int]) -> List[int]:
        if layer_name not in self.active_original_indices:
            return local_idxs

        active = self.active_original_indices[layer_name]
        mapped = []

        for i in local_idxs:
            if 0 <= i < len(active):
                mapped.append(active[i])

        return mapped