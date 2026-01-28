"""
Chapter 04: Kernel Fundamentals

This chapter covers:
- GPU architecture: SMs, warps, threads, memory hierarchy
- Writing basic CUDA kernels
- Calling CUDA from Python via PyTorch extensions
- Profiling with nsys
- Understanding kernel launch overhead
"""

from .gpu_architecture import (
    GPUSpec,
    get_gpu_spec,
    theoretical_occupancy,
    warp_efficiency,
)
from .launch_overhead import (
    compare_many_small_vs_one_large,
    measure_kernel_launch_overhead,
)

__all__ = [
    "GPUSpec",
    "get_gpu_spec",
    "theoretical_occupancy",
    "warp_efficiency",
    "measure_kernel_launch_overhead",
    "compare_many_small_vs_one_large",
]
