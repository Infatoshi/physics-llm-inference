"""
Chapter 03: Compute Characteristics

This chapter covers:
- GEMM vs GEMV and their different performance characteristics
- The roofline model for predicting bottlenecks
- Arithmetic intensity and compute vs memory bound regimes
- How batching transforms memory-bound to compute-bound
"""

from .batching_benchmark import (
    benchmark_batched_gemv,
    find_transition_batch_size,
)
from .gemm_benchmark import benchmark_gemm, gemm_bytes, gemm_flops
from .gemv_benchmark import benchmark_gemv, gemv_bytes, gemv_flops
from .roofline import (
    arithmetic_intensity,
    is_compute_bound,
    plot_roofline,
    roofline_throughput,
)

__all__ = [
    "benchmark_gemm",
    "gemm_flops",
    "gemm_bytes",
    "benchmark_gemv",
    "gemv_flops",
    "gemv_bytes",
    "arithmetic_intensity",
    "roofline_throughput",
    "is_compute_bound",
    "plot_roofline",
    "benchmark_batched_gemv",
    "find_transition_batch_size",
]
