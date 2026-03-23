from dataclasses import dataclass
from typing import Optional


@dataclass
class ImportanceProcessorConfig:
    hidden_dim: int = 128
    hidden_dim_per_filter: int = 16
    reg_lambda: float = 1e-3
    activation: str = "tanh"  # tanh | relu | sigmoid
    max_batches: Optional[int] = None
    eps: float = 1e-8
    seed: int = 42
    use_double_for_solver: bool = True
    feature_type = "segmentation" # segmention | logits
    num_classes = 3 # if segmentation
    layer_names = "" # to get importance for all layers, or list (str) to specify the layer names