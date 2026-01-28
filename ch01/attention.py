import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores / math.sqrt(q.size(-1))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    seq_len = q.size(1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_fn = causal_attention if causal else naive_attention
        attn_output = attn_fn(q, k, v)
        output = self.o_proj(attn_output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf')) if causal else scores
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        output = self.o_proj(attn_output)
        return output
