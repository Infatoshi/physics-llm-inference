from dataclasses import dataclass

import torch


@dataclass
class TileConfig:
    tile_size: int
    num_tiles: int
    shared_memory_bytes: int


def compute_tile_config(n: int, tile_size: int = 32) -> TileConfig:
    num_tiles = (n + tile_size - 1) // tile_size
    shared_memory_bytes = tile_size * 4
    return TileConfig(
        tile_size=tile_size,
        num_tiles=num_tiles,
        shared_memory_bytes=shared_memory_bytes,
    )


def tiled_reduce(data: torch.Tensor, tile_size: int = 256) -> torch.Tensor:
    n = data.shape[0]
    num_tiles = (n + tile_size - 1) // tile_size

    partial_sums = []
    for i in range(num_tiles):
        start = i * tile_size
        end = min(start + tile_size, n)
        partial_sums.append(data[start:end].sum())

    return torch.stack(partial_sums).sum()


def demonstrate_bank_conflicts() -> dict:
    info = {
        "shared_memory_banks": 32,
        "bank_width_bytes": 4,
        "conflict_free_pattern": "consecutive threads access consecutive addresses",
        "conflict_pattern": "threads access addresses that map to same bank",
        "bank_formula": "bank = (address / 4) % 32",
    }
    return info


def explain_shared_memory() -> str:
    explanation = """
Shared Memory

Shared memory is on-chip memory shared by all threads in a block.
It is much faster than global memory (100x lower latency).

Key characteristics:
- 48-164 KB per SM (architecture dependent)
- Organized into 32 banks
- Bank conflicts cause serialization

Usage pattern:
1. Load tile from global memory to shared memory
2. Synchronize threads (__syncthreads)
3. Compute using shared memory
4. Write results back to global memory

Tiled Matrix Multiplication:
- Load a tile of A and B into shared memory
- Compute partial products
- Repeat for all tiles
- Reduces global memory traffic by tile_size factor
"""
    return explanation


def shared_memory_requirements(
    threads_per_block: int,
    elements_per_thread: int,
    dtype_bytes: int = 4,
) -> int:
    return threads_per_block * elements_per_thread * dtype_bytes


def max_blocks_by_shared_memory(
    shared_per_block: int,
    shared_per_sm: int = 100 * 1024,
) -> int:
    if shared_per_block == 0:
        return float('inf')
    return shared_per_sm // shared_per_block


if __name__ == "__main__":
    print(explain_shared_memory())
    print("\nBank Conflict Info:")
    for k, v in demonstrate_bank_conflicts().items():
        print(f"  {k}: {v}")

    print("\nTile Configuration Example:")
    config = compute_tile_config(n=1024, tile_size=32)
    print(f"  Tiles: {config.num_tiles}")
    print(f"  Shared memory: {config.shared_memory_bytes} bytes")

    print("\nShared Memory Occupancy Impact:")
    for smem in [0, 16*1024, 32*1024, 48*1024]:
        max_blocks = max_blocks_by_shared_memory(smem)
        if max_blocks == float('inf'):
            print(f"  {smem//1024:3d} KB -> unlimited blocks")
        else:
            print(f"  {smem//1024:3d} KB -> {max_blocks} blocks/SM")
