"""
Chapter 09: Scaling Up

This chapter covers:
- MoE (Mixture of Experts) architecture
- MoE inference mechanics
- Tensor parallelism for large models
- NCCL communication primitives
"""

from .moe_inference import (
    ExpertCache,
    MoEInferenceEngine,
)
from .moe_layer import (
    ExpertLayer,
    MoEConfig,
    MoELayer,
    Router,
)
from .nccl_primitives import (
    AllGatherConfig,
    AllReduceConfig,
    simulate_all_gather,
    simulate_all_reduce,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelConfig,
)

__all__ = [
    "MoEConfig",
    "Router",
    "ExpertLayer",
    "MoELayer",
    "MoEInferenceEngine",
    "ExpertCache",
    "TensorParallelConfig",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "AllReduceConfig",
    "AllGatherConfig",
    "simulate_all_reduce",
    "simulate_all_gather",
]
