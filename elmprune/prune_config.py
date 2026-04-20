from dataclasses import dataclass
from typing import Literal
from enum import Enum

class PruneVerboseLevel(Enum):
    BASIC = 1
    BASIC_ERROR = 2
    ALL = 3

@dataclass
class PruneConfig:
    importance_type: Literal["elm_global", "elm_layerwise", "elm_filterwise"]
    target_param_reduction: float          # ex.: 0.20 = reduzir 20% dos params reais
    selection_scope: Literal["global", "local"]
    min_channels_abs: int = 16
    min_keep_ratio: float = 0.50
    max_layer_prune_ratio: float = 0.35
    per_step_layer_ratio: float = 0.05     # poda pequena por iteração
    round_to: int = 8
    verbose: PruneVerboseLevel = PruneVerboseLevel.BASIC_ERROR