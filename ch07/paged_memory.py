from dataclasses import dataclass, field

import torch


@dataclass
class BlockTable:
    request_id: int
    block_indices: list[int] = field(default_factory=list)
    num_tokens: int = 0

    def num_blocks(self) -> int:
        return len(self.block_indices)


class PagedKVCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.free_blocks: set[int] = set(range(num_blocks))
        self.block_tables: dict[int, BlockTable] = {}

        if torch.cuda.is_available() and device == "cuda":
            self.k_cache = torch.zeros(
                (num_blocks, num_layers, block_size, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            self.v_cache = torch.zeros(
                (num_blocks, num_layers, block_size, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
        else:
            self.k_cache = None
            self.v_cache = None

    def allocate_blocks(self, request_id: int, num_tokens: int) -> BlockTable:
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"Not enough free blocks: need {num_blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        allocated = []
        for _ in range(num_blocks_needed):
            block_idx = self.free_blocks.pop()
            allocated.append(block_idx)

        block_table = BlockTable(
            request_id=request_id,
            block_indices=allocated,
            num_tokens=num_tokens,
        )
        self.block_tables[request_id] = block_table

        return block_table

    def extend_blocks(self, request_id: int, new_tokens: int) -> None:
        if request_id not in self.block_tables:
            raise KeyError(f"Request {request_id} not found")

        table = self.block_tables[request_id]
        old_tokens = table.num_tokens
        new_total = old_tokens + new_tokens

        old_blocks = (old_tokens + self.block_size - 1) // self.block_size
        new_blocks = (new_total + self.block_size - 1) // self.block_size
        blocks_needed = new_blocks - old_blocks

        if blocks_needed > len(self.free_blocks):
            raise RuntimeError(
                f"Not enough free blocks for extension: need {blocks_needed}, "
                f"have {len(self.free_blocks)}"
            )

        for _ in range(blocks_needed):
            block_idx = self.free_blocks.pop()
            table.block_indices.append(block_idx)

        table.num_tokens = new_total

    def free_blocks_for_request(self, request_id: int) -> int:
        if request_id not in self.block_tables:
            return 0

        table = self.block_tables.pop(request_id)
        freed = len(table.block_indices)

        for block_idx in table.block_indices:
            self.free_blocks.add(block_idx)

        return freed

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_memory_usage(self) -> dict:
        used_blocks = self.num_blocks - len(self.free_blocks)
        bytes_per_block = (
            2 *
            self.num_layers *
            self.block_size *
            self.num_heads *
            self.head_dim *
            2
        )
        total_bytes = self.num_blocks * bytes_per_block
        used_bytes = used_blocks * bytes_per_block

        return {
            "total_blocks": self.num_blocks,
            "used_blocks": used_blocks,
            "free_blocks": len(self.free_blocks),
            "block_size_tokens": self.block_size,
            "bytes_per_block": bytes_per_block,
            "total_mb": total_bytes / 1024 / 1024,
            "used_mb": used_bytes / 1024 / 1024,
            "utilization": used_blocks / self.num_blocks if self.num_blocks > 0 else 0,
        }


def explain_paged_memory() -> str:
    return """
Paged KV Cache (PagedAttention)

Problem: Variable sequence lengths waste memory
  - Contiguous allocation: must reserve max_seq_len for each request
  - Short sequences waste memory
  - Can't share memory between requests

Solution: Page-based memory management (like OS virtual memory)

Key concepts:
  - Block: fixed-size chunk of KV cache (e.g., 16 tokens)
  - Block table: maps logical positions to physical blocks
  - Free list: pool of available blocks

Allocation:
  1. Request arrives with prompt of N tokens
  2. Allocate ceil(N / block_size) blocks
  3. Create block table mapping [0, 1, 2, ...] -> [physical blocks]

During generation:
  - When current block fills, allocate new block
  - Add to block table
  - No copying, no pre-allocation

Memory efficiency:
  - Only allocate what's needed
  - No fragmentation (all blocks same size)
  - Easy to free (return blocks to pool)

Non-contiguous attention:
  - Attention kernel reads from scattered blocks
  - Uses block table for indirection
  - Small overhead from non-contiguous access
"""


if __name__ == "__main__":
    print(explain_paged_memory())

    print("\n" + "=" * 60)
    print("Paged KV Cache Demo")
    print("-" * 60)

    cache = PagedKVCache(
        num_blocks=100,
        block_size=16,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        device="cpu",
    )

    print("Initial state:")
    print(f"  {cache.get_memory_usage()}")

    table1 = cache.allocate_blocks(request_id=1, num_tokens=50)
    print("\nAllocated for request 1 (50 tokens):")
    print(f"  Blocks: {table1.block_indices}")
    print(f"  Free: {cache.get_num_free_blocks()}")

    table2 = cache.allocate_blocks(request_id=2, num_tokens=100)
    print("\nAllocated for request 2 (100 tokens):")
    print(f"  Blocks: {table2.block_indices}")
    print(f"  Free: {cache.get_num_free_blocks()}")

    cache.extend_blocks(request_id=1, new_tokens=30)
    updated_table = cache.block_tables[1]
    print("\nExtended request 1 by 30 tokens:")
    print(f"  Total tokens: {updated_table.num_tokens}")
    print(f"  Blocks: {updated_table.block_indices}")

    freed = cache.free_blocks_for_request(request_id=2)
    print(f"\nFreed request 2: {freed} blocks returned")
    print(f"  Free: {cache.get_num_free_blocks()}")

    print("\nFinal memory usage:")
    usage = cache.get_memory_usage()
    for k, v in usage.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
