from dataclasses import dataclass

import torch


@dataclass
class FlashAttentionConfig:
    block_q: int = 64
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    if config is None:
        config = FlashAttentionConfig()

    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5

    BLOCK_Q = config.block_q
    BLOCK_K = config.block_k

    output = torch.zeros_like(q)
    row_max = torch.full((B, H, N), float('-inf'), device=q.device, dtype=q.dtype)
    row_sum = torch.zeros((B, H, N), device=q.device, dtype=q.dtype)

    num_q_blocks = (N + BLOCK_Q - 1) // BLOCK_Q
    num_k_blocks = (N + BLOCK_K - 1) // BLOCK_K

    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * BLOCK_Q
        q_end = min(q_start + BLOCK_Q, N)
        q_block = q[:, :, q_start:q_end, :]

        block_output = torch.zeros_like(q_block)
        block_max = torch.full((B, H, q_end - q_start), float('-inf'),
                               device=q.device, dtype=q.dtype)
        block_sum = torch.zeros((B, H, q_end - q_start),
                               device=q.device, dtype=q.dtype)

        for k_block_idx in range(num_k_blocks):
            k_start = k_block_idx * BLOCK_K
            k_end = min(k_start + BLOCK_K, N)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            new_max = torch.maximum(block_max, scores.max(dim=-1).values)

            scale_old = torch.exp(block_max - new_max)
            scale_new = torch.exp(scores - new_max.unsqueeze(-1))

            new_sum = block_sum * scale_old + scale_new.sum(dim=-1)

            block_output = (block_output * block_sum.unsqueeze(-1) * scale_old.unsqueeze(-1) +
                           torch.matmul(scale_new, v_block)) / new_sum.unsqueeze(-1)

            block_max = new_max
            block_sum = new_sum

        output[:, :, q_start:q_end, :] = block_output
        row_max[:, :, q_start:q_end] = block_max
        row_sum[:, :, q_start:q_end] = block_sum

    return output


def flash_attention_memory_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    block_size: int = 64,
    dtype_bytes: int = 2,
) -> dict:
    qkv_bytes = 3 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    output_bytes = batch_size * num_heads * seq_len * head_dim * dtype_bytes

    q_block = block_size * head_dim * dtype_bytes
    k_block = block_size * head_dim * dtype_bytes
    v_block = block_size * head_dim * dtype_bytes
    scores_block = block_size * block_size * dtype_bytes
    o_block = block_size * head_dim * dtype_bytes
    stats_block = block_size * 2 * dtype_bytes

    sram_bytes = q_block + k_block + v_block + scores_block + o_block + stats_block

    return {
        "hbm_bytes": qkv_bytes + output_bytes,
        "hbm_mb": (qkv_bytes + output_bytes) / 1024 / 1024,
        "sram_bytes_per_block": sram_bytes,
        "sram_kb_per_block": sram_bytes / 1024,
        "naive_hbm_bytes": qkv_bytes + output_bytes + batch_size * num_heads * seq_len * seq_len * dtype_bytes,
        "memory_savings": f"{seq_len // block_size}x",
    }


def explain_flash_attention() -> str:
    return """
FlashAttention

FlashAttention avoids materializing the full N x N attention matrix by:
1. Processing Q in tiles (blocks of rows)
2. For each Q tile, iterating through K,V tiles
3. Using online softmax to accumulate results without storing full attention

Algorithm for one Q block:
  Initialize: O = 0, m = -inf, d = 0

  For each K,V block:
    1. Compute S_ij = Q_i @ K_j^T / sqrt(d)
    2. m_new = max(m, rowmax(S_ij))
    3. P_ij = exp(S_ij - m_new)
    4. d_new = d * exp(m - m_new) + rowsum(P_ij)
    5. O_new = (O * d * exp(m - m_new) + P_ij @ V_j) / d_new
    6. Update m = m_new, d = d_new, O = O_new

Memory Analysis:
- Naive: O(N^2) HBM for attention matrix
- Flash: O(N * d) HBM, O(B^2) SRAM per tile

SRAM usage per tile:
- Q block: B_q * d
- K block: B_k * d
- V block: B_k * d
- Scores: B_q * B_k
- Output: B_q * d
- Stats: B_q * 2 (for m and d)

Total SRAM: O(B_q * d + B_k * d + B_q * B_k) per tile
"""


if __name__ == "__main__":
    print(explain_flash_attention())

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Correctness Verification")
        print("-" * 60)

        B, H, N, D = 2, 8, 256, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        from .attention_memory import naive_attention
        naive_out = naive_attention(q, k, v)
        flash_out = flash_attention_forward(q, k, v)

        max_diff = (naive_out - flash_out).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        print(f"Correctness: {'PASS' if max_diff < 0.01 else 'FAIL'}")

        print("\n" + "=" * 60)
        print("Memory Analysis")
        print("-" * 60)

        for seq_len in [512, 1024, 2048, 4096, 8192]:
            mem = flash_attention_memory_bytes(
                batch_size=1,
                num_heads=32,
                seq_len=seq_len,
                head_dim=64,
            )
            print(f"Seq {seq_len:5d}: Flash HBM={mem['hbm_mb']:.1f} MB, "
                  f"Naive HBM={mem['naive_hbm_bytes']/1024/1024:.1f} MB, "
                  f"Savings={mem['memory_savings']}")
    else:
        print("\nCUDA not available for verification")
