from dataclasses import dataclass


@dataclass
class HardwareSpec:
    peak_tflops: float
    memory_bandwidth_gbps: float
    name: str


RTX_3090 = HardwareSpec(
    peak_tflops=35.6,
    memory_bandwidth_gbps=936.0,
    name="RTX 3090",
)

RTX_4090 = HardwareSpec(
    peak_tflops=82.6,
    memory_bandwidth_gbps=1008.0,
    name="RTX 4090",
)

A100_80GB = HardwareSpec(
    peak_tflops=312.0,
    memory_bandwidth_gbps=2039.0,
    name="A100 80GB",
)

H100_SXM = HardwareSpec(
    peak_tflops=989.0,
    memory_bandwidth_gbps=3350.0,
    name="H100 SXM",
)


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    return flops / bytes_moved


def roofline_throughput(
    ai: float,
    hw: HardwareSpec,
) -> float:
    memory_bound_throughput = ai * hw.memory_bandwidth_gbps / 1000
    return min(memory_bound_throughput, hw.peak_tflops)


def is_compute_bound(ai: float, hw: HardwareSpec) -> bool:
    ridge_point = hw.peak_tflops * 1000 / hw.memory_bandwidth_gbps
    return ai >= ridge_point


def ridge_point(hw: HardwareSpec) -> float:
    return hw.peak_tflops * 1000 / hw.memory_bandwidth_gbps


def gemm_arithmetic_intensity(m: int, n: int, k: int) -> float:
    flops = 2 * m * n * k
    bytes_moved = (m * k + k * n + m * n) * 2
    return flops / bytes_moved


def gemv_arithmetic_intensity(m: int, k: int) -> float:
    flops = 2 * m * k
    bytes_moved = (m * k + k + m) * 2
    return flops / bytes_moved


def batched_gemv_arithmetic_intensity(batch: int, m: int, k: int) -> float:
    flops = 2 * batch * m * k
    bytes_moved = (m * k + batch * k + batch * m) * 2
    return flops / bytes_moved


def plot_roofline(
    hw: HardwareSpec,
    points: list | None = None,
    save_path: str | None = None,
):
    import matplotlib.pyplot as plt
    import numpy as np

    ai_range = np.logspace(-2, 4, 1000)
    throughput = np.minimum(
        ai_range * hw.memory_bandwidth_gbps / 1000,
        hw.peak_tflops
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai_range, throughput, 'b-', linewidth=2, label='Roofline')

    ridge = ridge_point(hw)
    ax.axvline(x=ridge, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(f'Ridge Point\nAI = {ridge:.1f}',
                xy=(ridge, hw.peak_tflops * 0.8),
                fontsize=10)

    if points:
        for name, ai, measured_tflops in points:
            color = 'green' if is_compute_bound(ai, hw) else 'red'
            ax.scatter([ai], [measured_tflops], s=100, c=color, zorder=5)
            ax.annotate(name, xy=(ai, measured_tflops),
                       xytext=(5, 5), textcoords='offset points')

    ax.fill_between(ai_range[ai_range < ridge],
                    throughput[ai_range < ridge],
                    0.01, alpha=0.2, color='red',
                    label='Memory Bound')
    ax.fill_between(ai_range[ai_range >= ridge],
                    throughput[ai_range >= ridge],
                    0.01, alpha=0.2, color='green',
                    label='Compute Bound')

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
    ax.set_ylabel('Throughput (TFLOPS)')
    ax.set_title(f'Roofline Model - {hw.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.01, 10000])
    ax.set_ylim([0.01, hw.peak_tflops * 1.5])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


if __name__ == "__main__":
    hw = RTX_3090

    print(f"Hardware: {hw.name}")
    print(f"Peak TFLOPS: {hw.peak_tflops}")
    print(f"Memory Bandwidth: {hw.memory_bandwidth_gbps} GB/s")
    print(f"Ridge Point: {ridge_point(hw):.1f} FLOP/Byte")
    print()

    gemm_ai = gemm_arithmetic_intensity(4096, 4096, 4096)
    print(f"GEMM (4096x4096x4096) AI: {gemm_ai:.1f} FLOP/Byte")
    print(f"  Compute bound: {is_compute_bound(gemm_ai, hw)}")
    print()

    gemv_ai = gemv_arithmetic_intensity(4096, 4096)
    print(f"GEMV (4096x4096) AI: {gemv_ai:.2f} FLOP/Byte")
    print(f"  Compute bound: {is_compute_bound(gemv_ai, hw)}")
    print()

    for batch in [1, 4, 16, 64, 256]:
        ai = batched_gemv_arithmetic_intensity(batch, 4096, 4096)
        bound = "compute" if is_compute_bound(ai, hw) else "memory"
        print(f"Batched GEMV (batch={batch:3d}) AI: {ai:.2f} FLOP/Byte ({bound})")
