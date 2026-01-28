import time
from dataclasses import dataclass

import torch


@dataclass
class AccessPatternResult:
    pattern_name: str
    time_us: float
    bandwidth_gbps: float
    efficiency: float


def coalesced_access(data: torch.Tensor) -> torch.Tensor:
    return data.clone()


def strided_access(data: torch.Tensor, stride: int = 32) -> torch.Tensor:
    n = data.shape[0]
    indices = torch.arange(0, n, stride, device=data.device)
    return data[indices]


def measure_access_pattern(
    size: int = 1024 * 1024 * 32,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> tuple[AccessPatternResult, AccessPatternResult]:
    data = torch.randn(size, device=device)
    bytes_per_element = 4

    for _ in range(warmup):
        _ = coalesced_access(data)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = coalesced_access(data)
    torch.cuda.synchronize()
    coalesced_time = (time.perf_counter() - start) / iterations

    total_bytes = size * bytes_per_element * 2
    coalesced_bw = total_bytes / coalesced_time / 1e9

    strided_data = torch.randn(size, device=device)
    stride = 32

    for _ in range(warmup):
        _ = strided_access(strided_data, stride)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = strided_access(strided_data, stride)
    torch.cuda.synchronize()
    strided_time = (time.perf_counter() - start) / iterations

    accessed_elements = size // stride
    strided_bytes = accessed_elements * bytes_per_element * 2
    strided_bw = strided_bytes / strided_time / 1e9

    peak_bandwidth = 936.0

    coalesced_result = AccessPatternResult(
        pattern_name="coalesced",
        time_us=coalesced_time * 1e6,
        bandwidth_gbps=coalesced_bw,
        efficiency=min(coalesced_bw / peak_bandwidth, 1.0),
    )

    strided_result = AccessPatternResult(
        pattern_name="strided",
        time_us=strided_time * 1e6,
        bandwidth_gbps=strided_bw,
        efficiency=min(strided_bw / peak_bandwidth, 1.0),
    )

    return coalesced_result, strided_result


def explain_coalescing():
    explanation = """
Memory Coalescing

GPU memory is accessed in transactions of 32, 64, or 128 bytes.
When threads in a warp access consecutive memory addresses,
the hardware combines these into a single transaction.

Coalesced access (good):
  Thread 0 reads address 0
  Thread 1 reads address 4
  Thread 2 reads address 8
  ...
  Thread 31 reads address 124
  -> 1 transaction of 128 bytes

Strided access (bad):
  Thread 0 reads address 0
  Thread 1 reads address 128
  Thread 2 reads address 256
  ...
  Thread 31 reads address 3968
  -> 32 separate transactions

The penalty for non-coalesced access can be 10-32x slower.
"""
    return explanation


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        print(explain_coalescing())
        print("\nBenchmark Results:")
        print("=" * 60)

        coalesced, strided = measure_access_pattern()

        print(f"Coalesced: {coalesced.time_us:.1f} us, {coalesced.bandwidth_gbps:.1f} GB/s")
        print(f"Strided:   {strided.time_us:.1f} us, {strided.bandwidth_gbps:.1f} GB/s")
        print(f"Coalesced efficiency: {coalesced.efficiency:.1%}")
