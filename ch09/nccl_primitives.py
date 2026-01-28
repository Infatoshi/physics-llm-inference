from dataclasses import dataclass


@dataclass
class AllReduceConfig:
    world_size: int = 8
    data_size_mb: float = 1.0
    bandwidth_gbps: float = 600.0
    latency_us: float = 5.0


@dataclass
class AllGatherConfig:
    world_size: int = 8
    data_size_per_gpu_mb: float = 1.0
    bandwidth_gbps: float = 600.0
    latency_us: float = 5.0


def simulate_all_reduce(config: AllReduceConfig) -> dict:
    data_bytes = config.data_size_mb * 1024 * 1024

    transfer_bytes = 2 * data_bytes * (config.world_size - 1) / config.world_size

    bandwidth_bytes_per_us = config.bandwidth_gbps * 1e9 / 8 / 1e6

    transfer_time_us = transfer_bytes / bandwidth_bytes_per_us

    total_time_us = config.latency_us + transfer_time_us

    effective_bandwidth = data_bytes / (total_time_us * 1e-6) / 1e9

    return {
        "data_size_mb": config.data_size_mb,
        "world_size": config.world_size,
        "transfer_bytes": transfer_bytes,
        "latency_us": config.latency_us,
        "transfer_time_us": transfer_time_us,
        "total_time_us": total_time_us,
        "effective_bandwidth_gbps": effective_bandwidth,
        "bandwidth_efficiency": effective_bandwidth / config.bandwidth_gbps,
    }


def simulate_all_gather(config: AllGatherConfig) -> dict:
    data_per_gpu_bytes = config.data_size_per_gpu_mb * 1024 * 1024

    total_data_bytes = data_per_gpu_bytes * config.world_size

    transfer_bytes = data_per_gpu_bytes * (config.world_size - 1)

    bandwidth_bytes_per_us = config.bandwidth_gbps * 1e9 / 8 / 1e6

    transfer_time_us = transfer_bytes / bandwidth_bytes_per_us

    total_time_us = config.latency_us + transfer_time_us

    effective_bandwidth = total_data_bytes / (total_time_us * 1e-6) / 1e9

    return {
        "data_per_gpu_mb": config.data_size_per_gpu_mb,
        "total_data_mb": total_data_bytes / 1024 / 1024,
        "world_size": config.world_size,
        "transfer_time_us": transfer_time_us,
        "total_time_us": total_time_us,
        "effective_bandwidth_gbps": effective_bandwidth,
    }


def compute_ring_all_reduce_time(
    data_size_bytes: int,
    world_size: int,
    bandwidth_gbps: float = 600.0,
    latency_us: float = 5.0,
) -> float:
    chunk_size = data_size_bytes / world_size

    num_steps = 2 * (world_size - 1)

    bandwidth_bytes_per_us = bandwidth_gbps * 1e9 / 8 / 1e6

    per_step_time = latency_us + chunk_size / bandwidth_bytes_per_us

    total_time = num_steps * per_step_time

    return total_time


def compute_communication_overlap_potential(
    compute_time_us: float,
    comm_time_us: float,
) -> dict:
    sequential_time = compute_time_us + comm_time_us

    overlapped_time = max(compute_time_us, comm_time_us)

    overlap_ratio = min(compute_time_us, comm_time_us) / max(compute_time_us, comm_time_us)

    return {
        "compute_time_us": compute_time_us,
        "comm_time_us": comm_time_us,
        "sequential_time_us": sequential_time,
        "overlapped_time_us": overlapped_time,
        "potential_speedup": sequential_time / overlapped_time,
        "overlap_ratio": overlap_ratio,
        "bottleneck": "compute" if compute_time_us >= comm_time_us else "communication",
    }


def explain_nccl() -> str:
    return """
NCCL (NVIDIA Collective Communications Library)

NCCL provides optimized collective operations for multi-GPU:

1. All-Reduce: combine data from all GPUs, result on all GPUs
   - Used for: gradient averaging, tensor parallel reduction
   - Algorithm: ring all-reduce or tree all-reduce
   - Time: O(data_size) bandwidth + O(log N) latency

2. All-Gather: gather data from all GPUs to all GPUs
   - Used for: collecting results, expert parallelism
   - Result: each GPU has concatenation of all inputs
   - Time: O(N * data_size) bandwidth

3. Reduce-Scatter: reduce and scatter to different GPUs
   - Used for: distributed optimizer
   - Each GPU gets 1/N of the reduced result
   - Time: O(data_size) bandwidth

4. All-to-All: each GPU sends different data to each GPU
   - Used for: expert parallelism routing
   - Time: O(data_size) bandwidth

Ring Algorithm (All-Reduce):
  1. Divide data into N chunks
  2. Each GPU sends one chunk to next GPU
  3. After N-1 steps: reduce phase complete
  4. Another N-1 steps: gather phase complete
  Total: 2*(N-1) steps, each sending data_size/N

NVLink vs PCIe:
  - NVLink: 600 GB/s (H100), low latency
  - PCIe 5.0: 64 GB/s, higher latency
  - NVLink enables efficient tensor parallelism
"""


if __name__ == "__main__":
    print(explain_nccl())

    print("\n" + "=" * 60)
    print("All-Reduce Simulation")
    print("-" * 60)

    for world_size in [2, 4, 8]:
        config = AllReduceConfig(
            world_size=world_size,
            data_size_mb=100.0,
            bandwidth_gbps=600.0,
        )
        result = simulate_all_reduce(config)
        print(f"\nWorld size: {world_size}")
        print(f"  Data size: {result['data_size_mb']:.1f} MB")
        print(f"  Total time: {result['total_time_us']:.1f} us")
        print(f"  Effective bandwidth: {result['effective_bandwidth_gbps']:.1f} GB/s")
        print(f"  Efficiency: {result['bandwidth_efficiency']:.1%}")

    print("\n" + "=" * 60)
    print("Communication-Compute Overlap Analysis")
    print("-" * 60)

    for compute_us in [100, 500, 1000]:
        comm_us = 200
        overlap = compute_communication_overlap_potential(compute_us, comm_us)
        print(f"\nCompute: {compute_us} us, Comm: {comm_us} us")
        print(f"  Sequential: {overlap['sequential_time_us']:.0f} us")
        print(f"  Overlapped: {overlap['overlapped_time_us']:.0f} us")
        print(f"  Speedup: {overlap['potential_speedup']:.2f}x")
        print(f"  Bottleneck: {overlap['bottleneck']}")
