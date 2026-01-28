from dataclasses import dataclass

import torch


@dataclass
class GPUSpec:
    name: str
    compute_capability: tuple
    num_sms: int
    max_threads_per_sm: int
    max_threads_per_block: int
    warp_size: int
    shared_memory_per_sm_kb: int
    shared_memory_per_block_kb: int
    registers_per_sm: int
    l2_cache_mb: float
    memory_gb: float
    memory_bandwidth_gbps: float
    peak_fp16_tflops: float
    peak_fp32_tflops: float


AMPERE_SPECS = {
    "RTX 3090": GPUSpec(
        name="RTX 3090",
        compute_capability=(8, 6),
        num_sms=82,
        max_threads_per_sm=1536,
        max_threads_per_block=1024,
        warp_size=32,
        shared_memory_per_sm_kb=100,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        l2_cache_mb=6.0,
        memory_gb=24.0,
        memory_bandwidth_gbps=936.2,
        peak_fp16_tflops=35.6,
        peak_fp32_tflops=35.6,
    ),
    "A100 80GB": GPUSpec(
        name="A100 80GB",
        compute_capability=(8, 0),
        num_sms=108,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        warp_size=32,
        shared_memory_per_sm_kb=164,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        l2_cache_mb=40.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=2039.0,
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
    ),
}

HOPPER_SPECS = {
    "H100 SXM": GPUSpec(
        name="H100 SXM",
        compute_capability=(9, 0),
        num_sms=132,
        max_threads_per_sm=2048,
        max_threads_per_block=1024,
        warp_size=32,
        shared_memory_per_sm_kb=228,
        shared_memory_per_block_kb=48,
        registers_per_sm=65536,
        l2_cache_mb=50.0,
        memory_gb=80.0,
        memory_bandwidth_gbps=3350.0,
        peak_fp16_tflops=989.0,
        peak_fp32_tflops=67.0,
    ),
}


def get_gpu_spec(device: torch.device | None = None) -> GPUSpec | None:
    if not torch.cuda.is_available():
        return None

    if device is None:
        device = torch.device("cuda:0")

    props = torch.cuda.get_device_properties(device)
    name = props.name

    all_specs = {**AMPERE_SPECS, **HOPPER_SPECS}
    for spec_name, spec in all_specs.items():
        if spec_name in name:
            return spec

    return GPUSpec(
        name=name,
        compute_capability=(props.major, props.minor),
        num_sms=props.multi_processor_count,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        max_threads_per_block=props.max_threads_per_block,
        warp_size=props.warp_size,
        shared_memory_per_sm_kb=props.max_shared_memory_per_multiprocessor // 1024,
        shared_memory_per_block_kb=props.max_shared_memory_per_block // 1024,
        registers_per_sm=props.regs_per_multiprocessor,
        l2_cache_mb=props.l2_cache_size / 1024 / 1024,
        memory_gb=props.total_memory / 1024 / 1024 / 1024,
        memory_bandwidth_gbps=0.0,
        peak_fp16_tflops=0.0,
        peak_fp32_tflops=0.0,
    )


def theoretical_occupancy(
    threads_per_block: int,
    registers_per_thread: int,
    shared_memory_per_block: int,
    spec: GPUSpec,
) -> float:
    warps_per_block = (threads_per_block + spec.warp_size - 1) // spec.warp_size

    max_blocks_by_threads = spec.max_threads_per_sm // threads_per_block

    registers_per_block = registers_per_thread * threads_per_block
    max_blocks_by_registers = spec.registers_per_sm // registers_per_block if registers_per_block > 0 else float('inf')

    shared_mem_per_sm = spec.shared_memory_per_sm_kb * 1024
    max_blocks_by_shared = shared_mem_per_sm // shared_memory_per_block if shared_memory_per_block > 0 else float('inf')

    max_blocks = min(max_blocks_by_threads, max_blocks_by_registers, max_blocks_by_shared)
    max_blocks = int(max_blocks)

    active_warps = warps_per_block * max_blocks
    max_warps = spec.max_threads_per_sm // spec.warp_size

    return active_warps / max_warps


def warp_efficiency(active_threads: int, warp_size: int = 32) -> float:
    return active_threads / warp_size


def threads_to_grid_block(
    total_threads: int,
    threads_per_block: int = 256,
) -> tuple:
    num_blocks = (total_threads + threads_per_block - 1) // threads_per_block
    return (num_blocks,), (threads_per_block,)


if __name__ == "__main__":
    spec = get_gpu_spec()
    if spec:
        print(f"GPU: {spec.name}")
        print(f"Compute Capability: {spec.compute_capability}")
        print(f"SMs: {spec.num_sms}")
        print(f"Max threads/SM: {spec.max_threads_per_sm}")
        print(f"Shared memory/SM: {spec.shared_memory_per_sm_kb} KB")
        print(f"Registers/SM: {spec.registers_per_sm}")
        print(f"Memory: {spec.memory_gb:.1f} GB")
        print()

        occ = theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
            spec=spec,
        )
        print(f"Theoretical occupancy (256 threads, 32 regs): {occ:.1%}")
    else:
        print("No CUDA GPU available")
