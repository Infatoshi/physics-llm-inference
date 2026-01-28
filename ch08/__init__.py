"""
Chapter 08: Advanced Scheduling

This chapter covers:
- Chunked prefill for long prompts
- Mixed prefill/decode batches
- CUDA graphs for low-latency decode
- Overlap scheduling techniques
"""

from .chunked_prefill import (
    ChunkConfig,
    ChunkedPrefillScheduler,
)
from .cuda_graph import (
    CUDAGraphRunner,
    GraphConfig,
)
from .mixed_batch import (
    MixedBatch,
    MixedBatchScheduler,
)
from .overlap_scheduler import (
    OverlapConfig,
    OverlapScheduler,
)

__all__ = [
    "ChunkedPrefillScheduler",
    "ChunkConfig",
    "MixedBatchScheduler",
    "MixedBatch",
    "CUDAGraphRunner",
    "GraphConfig",
    "OverlapScheduler",
    "OverlapConfig",
]
