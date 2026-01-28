import time
from dataclasses import dataclass

import torch


@dataclass
class TensorCoreResult:
    size: int
    fp16_us: float
    fp32_us: float
    speedup: float
    fp16_tflops: float
    fp32_tflops: float


def tensor_core_info() -> dict:
    return {
        "operation": "D = A * B + C",
        "input_types": ["fp16", "bf16", "tf32", "int8", "fp8"],
        "accumulator_types": ["fp16", "fp32", "int32"],
        "ampere_shape": "8x8x4 or 16x8x16",
        "hopper_shape": "up to 64x256x16",
        "minimum_size": "matrices should be multiples of 8 or 16",
    }


def benchmark_tensor_cores(
    size: int = 4096,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> TensorCoreResult | None:
    if not torch.cuda.is_available():
        return None

    a_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)
    b_fp16 = torch.randn(size, size, device=device, dtype=torch.float16)

    for _ in range(warmup):
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a_fp16, b_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - start) / iterations

    a_fp32 = a_fp16.float()
    b_fp32 = b_fp16.float()

    for _ in range(warmup):
        _ = torch.matmul(a_fp32, b_fp32)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a_fp32, b_fp32)
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / iterations

    flops = 2 * size * size * size

    return TensorCoreResult(
        size=size,
        fp16_us=fp16_time * 1e6,
        fp32_us=fp32_time * 1e6,
        speedup=fp32_time / fp16_time,
        fp16_tflops=flops / fp16_time / 1e12,
        fp32_tflops=flops / fp32_time / 1e12,
    )


def explain_tensor_cores() -> str:
    return """
Tensor Cores

Tensor cores are specialized units for matrix multiply-accumulate.
They perform D = A * B + C on small matrix tiles in a single cycle.

Why Tensor Cores Matter:
- 8-16x higher throughput than CUDA cores for matmul
- Essential for LLM inference performance
- cuBLAS/cuDNN automatically use them for supported dtypes

Requirements for Tensor Core Usage:
1. Data type: fp16, bf16, tf32, int8, or fp8
2. Matrix dimensions: multiples of 8 or 16
3. Alignment: pointers aligned to 16 bytes
4. Operation: matmul or conv2d

Ampere Tensor Cores (A100, RTX 30xx):
- 312 TFLOPS fp16 (A100)
- 35.6 TFLOPS fp16 (RTX 3090)
- 8x8x4 or 16x8x16 matrix tiles

Hopper Tensor Cores (H100):
- 989 TFLOPS fp16
- Larger tiles, more efficient
- FP8 support for 2x throughput

Detecting Tensor Core Usage:
- Profile with ncu or nsight-systems
- Look for sm__pipe_tensor_cycles_active metric
- Or compare fp16 vs fp32 matmul performance (2x+ indicates tensor cores)
"""


def verify_tensor_core_usage(size: int = 4096) -> dict | None:
    if not torch.cuda.is_available():
        return None

    result = benchmark_tensor_cores(size=size, warmup=5, iterations=20)
    if result is None:
        return None

    likely_using_tc = result.speedup > 1.5

    return {
        "size": size,
        "fp16_tflops": result.fp16_tflops,
        "fp32_tflops": result.fp32_tflops,
        "speedup": result.speedup,
        "likely_tensor_cores": likely_using_tc,
        "note": "speedup > 1.5x suggests tensor core usage" if likely_using_tc
                else "speedup < 1.5x suggests CUDA cores only",
    }


if __name__ == "__main__":
    print(explain_tensor_cores())
    print("\nTensor Core Specs:")
    for k, v in tensor_core_info().items():
        print(f"  {k}: {v}")

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Benchmark Results:")
        print("-" * 60)

        for size in [1024, 2048, 4096]:
            result = benchmark_tensor_cores(size=size)
            if result:
                print(f"Size {size}x{size}:")
                print(f"  FP16: {result.fp16_us:.1f} us ({result.fp16_tflops:.1f} TFLOPS)")
                print(f"  FP32: {result.fp32_us:.1f} us ({result.fp32_tflops:.1f} TFLOPS)")
                print(f"  Speedup: {result.speedup:.2f}x")

        print("\nTensor Core Verification:")
        verification = verify_tensor_core_usage()
        if verification:
            for k, v in verification.items():
                print(f"  {k}: {v}")
    else:
        print("\nCUDA not available")
