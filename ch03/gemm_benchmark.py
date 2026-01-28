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


def gemm_flops(m: int, n: int, k: int) -> int:
    return 2 * m * n * k


def gemm_bytes(m: int, n: int, k: int, dtype: torch.dtype = torch.float16) -> int:
    element_size = torch.tensor([], dtype=dtype).element_size()
    return (m * k + k * n + m * n) * element_size


def benchmark_gemm(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> BenchmarkResult:
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(k, n, dtype=dtype, device=device)

    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.mm(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    times_tensor = torch.tensor(times)
    mean_us = times_tensor.mean().item()
    std_us = times_tensor.std().item()
    min_us = times_tensor.min().item()
    max_us = times_tensor.max().item()

    flops = gemm_flops(m, n, k)
    tflops = flops / (mean_us * 1e-6) / 1e12

    bytes_moved = gemm_bytes(m, n, k, dtype)
    memory_gbps = bytes_moved / (mean_us * 1e-6) / 1e9

    return BenchmarkResult(
        mean_us=mean_us,
        std_us=std_us,
        min_us=min_us,
        max_us=max_us,
        tflops=tflops,
        memory_gbps=memory_gbps,
    )


def benchmark_prefill_gemm(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float16,
) -> BenchmarkResult:
    m = batch_size * seq_len
    n = hidden_dim
    k = hidden_dim
    return benchmark_gemm(m, n, k, dtype=dtype)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        print("GEMM Benchmark (Prefill-like workloads)")
        print("=" * 60)

        configs = [
            (32, 512, 4096, 4096),
            (32, 1024, 4096, 4096),
            (32, 2048, 4096, 4096),
            (1, 4096, 4096, 4096),
        ]

        for batch, seq, hidden, _ in configs:
            m = batch * seq
            result = benchmark_gemm(m, hidden, hidden)
            print(f"M={m:6d}, N={hidden}, K={hidden}: "
                  f"{result.mean_us:8.1f} us, {result.tflops:.2f} TFLOPS")
