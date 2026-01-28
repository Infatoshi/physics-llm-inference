"""
Chapter 05: Kernel Optimization

This chapter covers:
- Memory coalescing patterns
- Shared memory and tiling
- Tensor cores
- Triton programming
- Profiling with ncu
"""

from .memory_coalescing import (
    coalesced_access,
    measure_access_pattern,
    strided_access,
)
from .shared_memory import (
    demonstrate_bank_conflicts,
    tiled_reduce,
)
from .triton_matmul import (
    benchmark_triton_matmul,
    triton_matmul,
)

__all__ = [
    "coalesced_access",
    "strided_access",
    "measure_access_pattern",
    "tiled_reduce",
    "demonstrate_bank_conflicts",
    "triton_matmul",
    "benchmark_triton_matmul",
]
