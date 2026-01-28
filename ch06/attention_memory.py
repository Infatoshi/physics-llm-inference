from dataclasses import dataclass

import torch


@dataclass
class AttentionMemoryStats:
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    qk_bytes: int
    softmax_bytes: int
    output_bytes: int
    total_bytes: int
    total_mb: float


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)

    return output


def attention_memory_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> AttentionMemoryStats:
    qk_bytes = batch_size * num_heads * seq_len * seq_len * dtype_bytes

    softmax_bytes = batch_size * num_heads * seq_len * seq_len * dtype_bytes

    output_bytes = batch_size * num_heads * seq_len * head_dim * dtype_bytes

    total_bytes = qk_bytes + softmax_bytes + output_bytes

    return AttentionMemoryStats(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        qk_bytes=qk_bytes,
        softmax_bytes=softmax_bytes,
        output_bytes=output_bytes,
        total_bytes=total_bytes,
        total_mb=total_bytes / 1024 / 1024,
    )


def attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
) -> int:
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len

    attn_v_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    return qk_flops + softmax_flops + attn_v_flops


def attention_arithmetic_intensity(
    seq_len: int,
    head_dim: int,
) -> float:
    flops = 4 * seq_len * seq_len * head_dim + 5 * seq_len * seq_len

    bytes_rw = 2 * seq_len * seq_len * 2 + seq_len * head_dim * 2 * 3

    return flops / bytes_rw


def explain_attention_bottleneck() -> str:
    return """
The Attention Memory Bottleneck

Standard attention computes:
1. S = Q @ K^T / sqrt(d)  -> O(N^2 * d) compute, O(N^2) memory
2. P = softmax(S)         -> O(N^2) compute, O(N^2) memory
3. O = P @ V              -> O(N^2 * d) compute, O(N * d) memory

The problem: we must materialize the full N x N attention matrix.

For a 128K context with 32 heads:
- Attention matrix: 32 * 128K * 128K * 2 bytes = 1 TB per layer!
- This must be written to HBM, then read back for softmax, then written again.

Memory bandwidth becomes the bottleneck, not compute.
Arithmetic intensity is low because we do O(N^2) work with O(N^2) memory traffic.

FlashAttention Solution:
- Never materialize the full N x N matrix
- Compute attention in tiles that fit in SRAM
- Use online softmax to avoid multiple passes
- Reduce HBM traffic from O(N^2) to O(N)
"""


@torch.no_grad()
def benchmark_attention_memory(
    seq_lens: list = None,
    head_dim: int = 64,
    num_heads: int = 32,
    batch_size: int = 1,
    device: str = "cuda",
) -> dict:
    if seq_lens is None:
        seq_lens = [512, 1024, 2048, 4096, 8192]

    results = {}

    for seq_len in seq_lens:
        try:
            torch.cuda.reset_peak_memory_stats()

            q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                           device=device, dtype=torch.float16)
            k = torch.randn(batch_size, num_heads, seq_len, head_dim,
                           device=device, dtype=torch.float16)
            v = torch.randn(batch_size, num_heads, seq_len, head_dim,
                           device=device, dtype=torch.float16)

            _ = naive_attention(q, k, v)
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            theoretical = attention_memory_bytes(
                batch_size, num_heads, seq_len, head_dim, dtype_bytes=2
            )

            results[seq_len] = {
                "theoretical_mb": theoretical.total_mb,
                "actual_mb": peak_memory,
                "qk_matrix_mb": theoretical.qk_bytes / 1024 / 1024,
            }

            del q, k, v
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            results[seq_len] = {"oom": True}
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    print(explain_attention_bottleneck())

    print("\n" + "=" * 60)
    print("Memory Scaling by Sequence Length")
    print("-" * 60)

    for seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        stats = attention_memory_bytes(
            batch_size=1,
            num_heads=32,
            seq_len=seq_len,
            head_dim=64,
            dtype_bytes=2,
        )
        ai = attention_arithmetic_intensity(seq_len, 64)
        print(f"Seq {seq_len:6d}: {stats.total_mb:8.1f} MB, AI={ai:.2f} FLOP/byte")

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("Actual Memory Usage (GPU)")
        print("-" * 60)

        results = benchmark_attention_memory()
        for seq_len, data in results.items():
            if "oom" in data:
                print(f"Seq {seq_len}: OOM")
            else:
                print(f"Seq {seq_len}: theoretical={data['theoretical_mb']:.1f} MB, "
                      f"actual={data['actual_mb']:.1f} MB")
