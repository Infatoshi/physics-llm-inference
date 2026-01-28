"""
Chapter 06: FlashAttention

This chapter covers:
- The attention memory bottleneck
- Online softmax algorithm
- Tiled attention computation
- FlashAttention implementation
- FlashInfer usage
"""

from .attention_memory import (
    attention_arithmetic_intensity,
    attention_flops,
    attention_memory_bytes,
    naive_attention,
)
from .flash_attention import (
    FlashAttentionConfig,
    flash_attention_forward,
)
from .online_softmax import (
    online_softmax,
    online_softmax_with_output,
    standard_softmax,
)

__all__ = [
    "naive_attention",
    "attention_memory_bytes",
    "attention_flops",
    "attention_arithmetic_intensity",
    "standard_softmax",
    "online_softmax",
    "online_softmax_with_output",
    "flash_attention_forward",
    "FlashAttentionConfig",
]
