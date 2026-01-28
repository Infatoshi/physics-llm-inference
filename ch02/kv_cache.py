import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int

    @classmethod
    def create(
        cls,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        k_cache = torch.zeros(
            (batch_size, max_seq_len, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros(
            (batch_size, max_seq_len, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        return cls(k_cache=k_cache, v_cache=v_cache, seq_len=0)

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = k.shape[1]
        start = self.seq_len
        end = start + new_tokens
        self.k_cache[:, start:end] = k
        self.v_cache[:, start:end] = v
        self.seq_len = end
        return self.k_cache[:, :end], self.v_cache[:, :end]

    def memory_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2


class GQAWithCache(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
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
        kv_cache: KVCache | None = None,
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        k_full, v_full = (kv_cache.update(k, v) if kv_cache else (k, v)) if use_cache else (k, v)

        q = q.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_full.transpose(1, 2)

        k_full = k_full.repeat_interleave(self.num_groups, dim=1)
        v_full = v_full.repeat_interleave(self.num_groups, dim=1)

        full_seq_len = k_full.shape[2]
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = torch.triu(
            torch.ones(seq_len, full_seq_len, device=x.device, dtype=torch.bool),
            diagonal=full_seq_len - seq_len + 1
        ) if seq_len > 1 else None
        scores = scores.masked_fill(mask, float('-inf')) if mask is not None else scores

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_full)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output, kv_cache


def calculate_kv_cache_size(
    batch_size: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> dict:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    per_token_per_layer = 2 * num_kv_heads * head_dim * bytes_per_element
    per_token = per_token_per_layer * num_layers
    total = per_token * max_seq_len * batch_size
    return {
        "per_token_per_layer_bytes": per_token_per_layer,
        "per_token_bytes": per_token,
        "total_bytes": total,
        "total_mb": total / 1024 / 1024,
        "total_gb": total / 1024 / 1024 / 1024,
    }
