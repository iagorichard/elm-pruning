from dataclasses import dataclass
from math import ceil, floor
from typing import Dict, List, Literal, Optional, Any
from collections import defaultdict
import copy
import torch
import torch.nn as nn
import torch_pruning as tp

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
    verbose: bool = True