import time
from dataclasses import dataclass

import torch


@dataclass
class LaunchOverheadResult:
    num_kernels: int
    total_time_us: float
    per_kernel_us: float
    work_time_us: float
    overhead_fraction: float


def measure_kernel_launch_overhead(
    num_kernels: int = 1000,
    tensor_size: int = 1024,
    warmup: int = 10,
    device: str = "cuda",
) -> LaunchOverheadResult:
    x = torch.randn(tensor_size, device=device)
    y = torch.randn(tensor_size, device=device)

    for _ in range(warmup):
        torch.add(x, y)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_kernels):
        z = torch.add(x, y)
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    total_us = total_time * 1e6
    per_kernel_us = total_us / num_kernels

    big_x = torch.randn(tensor_size * num_kernels, device=device)
    big_y = torch.randn(tensor_size * num_kernels, device=device)

    for _ in range(warmup):
        torch.add(big_x, big_y)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    big_z = torch.add(big_x, big_y)
    torch.cuda.synchronize()
    single_kernel_time = time.perf_counter() - start
    work_time_us = single_kernel_time * 1e6

    overhead_fraction = 1.0 - (work_time_us / total_us)

    return LaunchOverheadResult(
        num_kernels=num_kernels,
        total_time_us=total_us,
        per_kernel_us=per_kernel_us,
        work_time_us=work_time_us,
        overhead_fraction=overhead_fraction,
    )


def compare_many_small_vs_one_large(
    num_ops: int = 100,
    sizes: list = None,
    device: str = "cuda",
) -> dict:
    if sizes is None:
        sizes = [1024, 4096, 16384, 65536, 262144]

    results = {}

    for size in sizes:
        x_small = torch.randn(size, device=device)
        y_small = torch.randn(size, device=device)

        for _ in range(10):
            torch.add(x_small, y_small)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_ops):
            z = torch.add(x_small, y_small)
        torch.cuda.synchronize()
        many_small_time = (time.perf_counter() - start) * 1e6

        x_large = torch.randn(size * num_ops, device=device)
        y_large = torch.randn(size * num_ops, device=device)

        for _ in range(10):
            torch.add(x_large, y_large)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        z_large = torch.add(x_large, y_large)
        torch.cuda.synchronize()
        one_large_time = (time.perf_counter() - start) * 1e6

        results[size] = {
            "many_small_us": many_small_time,
            "one_large_us": one_large_time,
            "speedup": many_small_time / one_large_time,
            "overhead_us": many_small_time - one_large_time,
        }

    return results


def demonstrate_fused_vs_unfused(
    batch: int = 32,
    seq: int = 1024,
    hidden: int = 4096,
    device: str = "cuda",
) -> dict:
    x = torch.randn(batch, seq, hidden, device=device, dtype=torch.float16)
    w1 = torch.randn(hidden, hidden, device=device, dtype=torch.float16)
    w2 = torch.randn(hidden, hidden, device=device, dtype=torch.float16)

    for _ in range(10):
        torch.addmm(torch.zeros(hidden, device=device, dtype=torch.float16),
                    x.view(-1, hidden), w1)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        h1 = x @ w1.T
        h2 = torch.relu(h1)
        h3 = h2 @ w2.T
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) * 1e6 / 100

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        h1 = x @ w1.T
        h2 = torch.relu_(h1)
        h3 = h2 @ w2.T
    torch.cuda.synchronize()
    partially_fused_time = (time.perf_counter() - start) * 1e6 / 100

    return {
        "unfused_us": unfused_time,
        "partially_fused_us": partially_fused_time,
        "speedup": unfused_time / partially_fused_time,
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
    else:
        print("Kernel Launch Overhead Analysis")
        print("=" * 60)

        result = measure_kernel_launch_overhead(num_kernels=1000, tensor_size=1024)
        print(f"1000 small kernels: {result.total_time_us:.1f} us total")
        print(f"Per-kernel time: {result.per_kernel_us:.2f} us")
        print(f"Equivalent work time: {result.work_time_us:.1f} us")
        print(f"Overhead fraction: {result.overhead_fraction:.1%}")
        print()

        print("Many Small vs One Large Comparison")
        print("-" * 60)
        comparison = compare_many_small_vs_one_large()
        print(f"{'Size':>10} {'100 small (us)':>15} {'1 large (us)':>15} {'Speedup':>10}")
        for size, data in comparison.items():
            print(f"{size:>10} {data['many_small_us']:>15.1f} "
                  f"{data['one_large_us']:>15.1f} {data['speedup']:>10.1f}x")
        print()

        print("Fused vs Unfused Operations")
        print("-" * 60)
        fused_result = demonstrate_fused_vs_unfused()
        print(f"Unfused (3 kernels): {fused_result['unfused_us']:.1f} us")
        print(f"Partially fused (in-place relu): {fused_result['partially_fused_us']:.1f} us")
        print(f"Speedup: {fused_result['speedup']:.2f}x")
