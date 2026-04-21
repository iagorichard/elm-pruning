import copy
import torch
import torch.nn as nn
import torch_pruning as tp

from math import ceil, floor
from typing import Dict, List, Any, Optional, Tuple

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
        # maybe in future:
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cpu")
        self.model = self._move_to_device(copy.deepcopy(model))
        self.importances_original = importances
        self.importances_live = copy.deepcopy(importances)
        self.example_inputs = self._minimize_example_inputs(self._move_to_device(example_inputs))
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

        if self.config.verbose.value >= PruneVerboseLevel.BASIC.value:
            print(f"[PruneProcessor] Starting pruning process:\n" 
                  f"[PruneProcessor] - base params   : {base_params};\n"
                  f"[PruneProcessor] - target params : {target_params} (reduction of ~{self.config.target_param_reduction*100}%).")

        last_params = base_params

        while True:
            current_params = count_trainable_params(self.model)
            if current_params <= target_params:
                break

            changed = False
            blocked_candidates = set()

            while True:
                candidate = self._select_best_candidate_for_one_step(blocked_candidates)
                if candidate is None:
                    break

                layer_name, idxs = candidate
                success, new_params = self._try_prune_one(layer_name, idxs, current_params)

                if success:
                    changed = True

                    if self.config.verbose.value == PruneVerboseLevel.ALL:
                        pruned_original_idxs = self._map_local_to_original_indices(layer_name, idxs)
                        print(
                            f"[PruneProcessor] {layer_name}: -{len(idxs)} ch | "
                            f"params {current_params} -> {new_params} | "
                            f"local idxs {idxs} | original idxs {pruned_original_idxs}"
                        )

                    # Após uma poda bem-sucedida, recalcula tudo do zero
                    # com o modelo e importâncias vivas atualizados.
                    break

                blocked_candidates.add((layer_name, tuple(idxs)))

            new_current_params = count_trainable_params(self.model)

            if not changed or new_current_params >= last_params:
                if self.config.verbose.value == PruneVerboseLevel.ALL:
                    print("[PruneProcessor] Stalled status.")
                break

            last_params = new_current_params

            if new_current_params <= target_params:
                break

        if self.config.verbose.value >= PruneVerboseLevel.BASIC.value:
            print(f"[PruneProcessor] Prune finished!\n"
                  f"[PruneProcessor] Final params: {sum(p.numel() for p in self.model.parameters())}")

        return self.model

    def _try_prune_one(
        self,
        layer_name: str,
        idxs: List[int],
        current_params: int,
    ) -> Tuple[bool, int]:
        trial_model = copy.deepcopy(self.model).to(self.device).eval()
        trial_mapping = build_name_to_module(trial_model)
        trial_root = trial_mapping.get(layer_name)

        if not isinstance(trial_root, nn.Conv2d):
            return False, current_params

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
                return False, current_params

            group.prune()

            validate_after_prune = getattr(self.config, "validate_after_prune", True)
            if validate_after_prune:
                with torch.no_grad():
                    _ = trial_model(self.example_inputs)

            new_params = count_trainable_params(trial_model)

            if new_params < current_params:
                self.model = trial_model
                self._remove_pruned_indices_from_live_importance(layer_name, idxs)
                return True, new_params

            return False, current_params

        except Exception as ex:
            if self.config.verbose.value >= PruneVerboseLevel.BASIC.value:
                print(f"[PRUNE] skipped {layer_name}: {ex}")
            return False, current_params

    def _move_to_device(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, tuple):
            return tuple(self._move_to_device(v) for v in x)
        if isinstance(x, list):
            return [self._move_to_device(v) for v in x]
        if isinstance(x, dict):
            return {k: self._move_to_device(v) for k, v in x.items()}
        return x

    def _minimize_example_inputs(self, x: Any) -> Any:
        """
        Tenta reduzir example_inputs para batch 1, o que costuma ajudar
        bastante no custo de build_dependency/forward.
        """
        if torch.is_tensor(x):
            if x.ndim >= 1 and x.shape[0] > 1:
                return x[:1]
            return x

        if isinstance(x, tuple):
            return tuple(self._minimize_example_inputs(v) for v in x)

        if isinstance(x, list):
            return [self._minimize_example_inputs(v) for v in x]

        if isinstance(x, dict):
            return {k: self._minimize_example_inputs(v) for k, v in x.items()}

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

    def _select_best_candidate_for_one_step(
        self,
        blocked_candidates: set[Tuple[str, Tuple[int, ...]]],
    ) -> Optional[Tuple[str, List[int]]]:
        name_to_module = build_name_to_module(self.model)

        best_layer = None
        best_idxs = None
        best_score = None

        for layer_name, scores_full in self.importances_live.items():
            module = name_to_module.get(layer_name)

            if not isinstance(module, nn.Conv2d):
                continue
            if layer_name in self.protected_layers:
                continue
            if layer_name not in self.base_out_channels:
                continue
            if len(scores_full) == 0:
                continue

            current_out = module.out_channels
            allowed = self._allowed_prunes_for_layer(layer_name, current_out)
            if allowed <= 0:
                continue

            max_valid = min(current_out, len(scores_full))
            scores = scores_full[:max_valid]

            if self.config.selection_scope == "global" and self.config.importance_type != "elm_global":
                scores = rank_normalize(scores)

            order = sorted(range(max_valid), key=lambda i: float(scores[i]))
            idxs = order[:allowed]

            if not idxs:
                continue

            candidate_key = (layer_name, tuple(idxs))
            if candidate_key in blocked_candidates:
                continue

            # score médio do bloco candidato: quanto menor, mais descartável
            block_score = float(sum(float(scores[i]) for i in idxs) / len(idxs))

            if best_score is None or block_score < best_score:
                best_score = block_score
                best_layer = layer_name
                best_idxs = idxs

        if best_layer is None:
            return None

        return best_layer, best_idxs

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