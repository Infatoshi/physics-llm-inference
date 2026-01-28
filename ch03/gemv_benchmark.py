import time
from dataclasses import dataclass

import torch


@dataclass
class BenchmarkResult:
    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    tflops: float
    memory_gbps: float


def gemv_flops(m: int, k: int) -> int:
    return 2 * m * k


def gemv_bytes(m: int, k: int, dtype: torch.dtype = torch.float16) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return (m * k + k + m) * element_size


def benchmark_gemv(
    m: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> BenchmarkResult:
    weight = torch.randn(m, k, dtype=dtype, device=device)
    x = torch.randn(k, dtype=dtype, device=device)

    for _ in range(warmup):
        torch.mv(weight, x)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mv(weight, x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    times_tensor = torch.tensor(times)
    mean_us = times_tensor.mean().item()
    std_us = times_tensor.std().item()
    min_us = times_tensor.min().item()
    max_us = times_tensor.max().item()

    flops = gemv_flops(m, k)
    tflops = flops / (mean_us * 1e-6) / 1e12

    bytes_moved = gemv_bytes(m, k, dtype)
    memory_gbps = bytes_moved / (mean_us * 1e-6) / 1e9

    return BenchmarkResult(
        mean_us=mean_us,
        std_us=std_us,
        min_us=min_us,
        max_us=max_us,
        tflops=tflops,
        memory_gbps=memory_gbps,
    )


def benchmark_decode_gemv(
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    return benchmark_gemv(hidden_dim, hidden_dim, dtype=dtype)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        print("GEMV Benchmark (Decode-like workloads)")
        print("=" * 60)

        hidden_dims = [2048, 4096, 8192, 16384]

        for hidden in hidden_dims:
            result = benchmark_gemv(hidden, hidden)
            print(f"M={hidden:6d}, K={hidden}: "
                  f"{result.mean_us:8.1f} us, {result.memory_gbps:.1f} GB/s, "
                  f"{result.tflops:.4f} TFLOPS")
