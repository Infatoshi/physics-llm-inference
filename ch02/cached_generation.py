"""
Chapter 02: Generation with KV Cache

The correct approach: compute K,V once during prefill, reuse during decode.

Two phases:
1. Prefill: Process entire prompt, build KV cache (compute-bound)
2. Decode: Generate one token at a time using cache (memory-bound)
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LayerKVCache:
    """KV cache for a single layer."""
    k: torch.Tensor  # (batch, max_seq_len, num_kv_heads, head_dim)
    v: torch.Tensor
    seq_len: int = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new KV and return full cache."""
        new_len = k_new.shape[1]
        self.k[:, self.seq_len:self.seq_len + new_len] = k_new
        self.v[:, self.seq_len:self.seq_len + new_len] = v_new
        self.seq_len += new_len
        return self.k[:, :self.seq_len], self.v[:, :self.seq_len]


class CachedGQA(nn.Module):
    """GQA with proper KV cache integration."""

    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project for new tokens
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k_new = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v_new = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Update cache and get full K, V
        if cache is not None:
            k_full, v_full = cache.update(k_new, v_new)
        else:
            k_full, v_full = k_new, v_new

        # Transpose for attention
        q = q.transpose(1, 2)  # (batch, num_heads, seq, head_dim)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        # Expand K, V for GQA
        k_full = k_full.repeat_interleave(self.num_groups, dim=1)
        v_full = v_full.repeat_interleave(self.num_groups, dim=1)

        # Attention
        full_len = k_full.shape[2]
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        if seq_len > 1:
            # Prefill: mask future tokens within the prompt
            causal_mask = torch.triu(
                torch.ones(seq_len, full_len, device=x.device, dtype=torch.bool),
                diagonal=full_len - seq_len + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_full)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.o_proj(attn_output)


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CachedTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden_dim)
        self.attn = CachedGQA(hidden_dim, num_heads, num_kv_heads)
        self.post_attn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerKVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        h = x + self.attn(self.input_norm(x), cache, start_pos)
        out = h + self.ffn(self.post_attn_norm(h))
        return out


class CachedTransformerModel(nn.Module):
    """Transformer with KV caching for efficient generation."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            CachedTransformerBlock(hidden_dim, num_heads, num_kv_heads, intermediate_dim)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: list[LayerKVCache] | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            x = layer(x, cache, start_pos)

        x = self.norm(x)
        return self.lm_head(x)

    def create_caches(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[LayerKVCache]:
        """Create KV caches for all layers."""
        caches = []
        for _ in range(self.num_layers):
            k = torch.zeros(
                batch_size, max_seq_len, self.num_kv_heads, self.head_dim,
                device=device, dtype=dtype
            )
            v = torch.zeros_like(k)
            caches.append(LayerKVCache(k=k, v=v, seq_len=0))
        return caches


def cached_generate(
    model: CachedTransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Generate with KV caching - the correct way.

    Returns generated tokens and timing breakdown.
    """
    model.eval()
    device = input_ids.device
    dtype = next(model.parameters()).dtype
    batch_size, prompt_len = input_ids.shape

    # Create caches
    max_seq_len = prompt_len + max_new_tokens
    caches = model.create_caches(batch_size, max_seq_len, device, dtype)

    timings = {"prefill_ms": 0, "decode_ms": [], "total_ms": 0}
    generated_tokens = []

    with torch.no_grad():
        # === PREFILL PHASE ===
        # Process entire prompt, populate caches
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(input_ids, caches, start_pos=0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        timings["prefill_ms"] = (time.perf_counter() - start) * 1000

        # Sample first new token
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens.append(next_token)

        # === DECODE PHASE ===
        # Generate one token at a time using cache
        for i in range(max_new_tokens - 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            # Only forward the new token
            logits = model(next_token, caches, start_pos=prompt_len + i)

            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["decode_ms"].append((time.perf_counter() - start) * 1000)

            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token)

    # Combine all generated tokens
    all_tokens = torch.cat([input_ids] + generated_tokens, dim=1)
    timings["total_ms"] = timings["prefill_ms"] + sum(timings["decode_ms"])

    return all_tokens, timings


def compare_generation_methods():
    """Compare naive vs cached generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Create model
    model = CachedTransformerModel(
        vocab_size=1000,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        num_kv_heads=2,
        intermediate_dim=1024,
    ).to(device, dtype)

    # Input
    prompt_len = 50
    max_new_tokens = 50
    input_ids = torch.randint(0, 1000, (1, prompt_len), device=device)

    print(f"Generation Comparison (device={device})")
    print(f"Prompt: {prompt_len} tokens, Generate: {max_new_tokens} tokens")
    print("=" * 60)

    # Cached generation
    tokens, timings = cached_generate(model, input_ids, max_new_tokens)

    print("\nCached Generation:")
    print(f"  Prefill:     {timings['prefill_ms']:.2f} ms ({prompt_len} tokens)")
    print(f"  Decode avg:  {sum(timings['decode_ms'])/len(timings['decode_ms']):.2f} ms/token")
    print(f"  Total:       {timings['total_ms']:.2f} ms")

    # Key insight
    print("\n" + "=" * 60)
    print("\nKey insight:")
    print(f"  Prefill processes {prompt_len} tokens in one pass (GEMM - compute bound)")
    print("  Decode processes 1 token at a time (GEMV - memory bound)")
    print("  Decode time is CONSTANT regardless of sequence length!")


if __name__ == "__main__":
    compare_generation_methods()
