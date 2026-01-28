import time
from dataclasses import dataclass

import torch


@dataclass
class BatchBenchmarkResult:
    batch_size: int
    mean_us: float
    tokens_per_second: float
    tflops: float
    memory_gbps: float


def benchmark_batched_gemv(
    batch_size: int,
    m: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> BatchBenchmarkResult:
    weight = torch.randn(m, k, dtype=dtype, device=device)
    x = torch.randn(batch_size, k, dtype=dtype, device=device)

    for _ in range(warmup):
        x @ weight.T
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        x @ weight.T
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1e6)

    mean_us = sum(times) / len(times)

    flops = 2 * batch_size * m * k
    tflops = flops / (mean_us * 1e-6) / 1e12

    element_size = torch.tensor([], dtype=dtype).element_size()
    bytes_moved = (m * k + batch_size * k + batch_size * m) * element_size
    memory_gbps = bytes_moved / (mean_us * 1e-6) / 1e9

    tokens_per_second = batch_size / (mean_us * 1e-6)

    return BatchBenchmarkResult(
        batch_size=batch_size,
        mean_us=mean_us,
        tokens_per_second=tokens_per_second,
        tflops=tflops,
        memory_gbps=memory_gbps,
    )


def find_transition_batch_size(
    m: int,
    k: int,
    peak_tflops: float,
    memory_bandwidth_gbps: float,
) -> int:
    ridge_point = peak_tflops * 1000 / memory_bandwidth_gbps

    element_size = 2

    batch = 1
    while True:
        flops = 2 * batch * m * k
        bytes_moved = (m * k + batch * k + batch * m) * element_size
        ai = flops / bytes_moved

        if ai >= ridge_point:
            return batch
        batch *= 2
        if batch > 1024:
            return batch


def benchmark_batch_sweep(
    m: int,
    k: int,
    batch_sizes: list[int],
    dtype: torch.dtype = torch.float16,
) -> list[BatchBenchmarkResult]:
    results = []
    for batch in batch_sizes:
        result = benchmark_batched_gemv(batch, m, k, dtype=dtype)
        results.append(result)
    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        hidden = 4096
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        print("Batched GEMV Benchmark (Decode throughput vs batch size)")
        print("=" * 70)
        print(f"{'Batch':>6} {'Time (us)':>12} {'Tokens/s':>12} {'TFLOPS':>10} {'GB/s':>10}")
        print("-" * 70)

        for batch in batch_sizes:
            result = benchmark_batched_gemv(batch, hidden, hidden)
            print(f"{result.batch_size:>6} {result.mean_us:>12.1f} "
                  f"{result.tokens_per_second:>12.0f} "
                  f"{result.tflops:>10.2f} {result.memory_gbps:>10.1f}")

        rtx_3090_tflops = 35.6
        rtx_3090_bandwidth = 936.0
        transition = find_transition_batch_size(
            hidden, hidden, rtx_3090_tflops, rtx_3090_bandwidth
        )
        print()
        print(f"Estimated transition batch size (RTX 3090): {transition}")
