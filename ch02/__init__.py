"""
Chapter 02: The Generation Loop

This chapter covers:
- Naive autoregressive generation and why it's O(n^2)
- KV caching: the key insight that makes generation efficient
- Prefill vs Decode phases and their different characteristics
- Memory requirements for KV cache at scale
"""

from .cached_generation import (
    CachedGQA,
    CachedTransformerBlock,
    CachedTransformerModel,
    LayerKVCache,
    cached_generate,
)
from .generation import naive_generate
from .kv_cache import GQAWithCache, KVCache, calculate_kv_cache_size

__all__ = [
    "naive_generate",
    "KVCache",
    "GQAWithCache",
    "calculate_kv_cache_size",
    "LayerKVCache",
    "CachedGQA",
    "CachedTransformerBlock",
    "CachedTransformerModel",
    "cached_generate",
]
