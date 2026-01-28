
import torch
import torch.nn as nn

from .ffn import FusedSwiGLUFFN
from .gqa import GroupedQueryAttention


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.self_attn = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=norm_eps)
        self.mlp = FusedSwiGLUFFN(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, causal=causal)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                intermediate_dim=intermediate_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "intermediate_dim": intermediate_dim,
        }

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> dict:
        embed_params = sum(p.numel() for p in self.embed_tokens.parameters())
        layer_params = sum(p.numel() for p in self.layers.parameters())
        norm_params = sum(p.numel() for p in self.norm.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "embed_tokens": embed_params,
            "layers": layer_params,
            "norm": norm_params,
            "lm_head": lm_head_params,
            "total": embed_params + layer_params + norm_params + lm_head_params,
        }


LLAMA_7B_CONFIG = {
    "vocab_size": 32000,
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 32,
    "intermediate_dim": 11008,
}


QWEN3_CONFIG = {
    "vocab_size": 151936,
    "hidden_dim": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "intermediate_dim": 11008,
}
