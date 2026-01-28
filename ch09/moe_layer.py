from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEConfig:
    hidden_dim: int = 4096
    expert_dim: int = 14336
    num_experts: int = 8
    num_experts_per_tok: int = 2
    normalize_expert_weights: bool = True


class Router(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)

        top_weights, top_indices = torch.topk(
            weights, self.config.num_experts_per_tok, dim=-1
        )

        if self.config.normalize_expert_weights:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        return top_weights, top_indices, logits


class ExpertLayer(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, expert_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.router = Router(config)
        self.experts = nn.ModuleList([
            ExpertLayer(config.hidden_dim, config.expert_dim)
            for _ in range(config.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        weights, indices, router_logits = self.router(x_flat)

        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.config.num_experts):
            mask = (indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue

            expert_input = x_flat[mask]
            expert_output = self.experts[expert_idx](expert_input)

            for k in range(self.config.num_experts_per_tok):
                expert_mask = indices[:, k] == expert_idx
                combined_mask = mask & expert_mask
                if combined_mask.any():
                    output[combined_mask] += (
                        weights[combined_mask, k].unsqueeze(-1) *
                        expert_output[expert_mask[mask]]
                    )

        return output.view(batch_size, seq_len, hidden_dim)


def expert_load_balance_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    num_experts_per_tok: int,
) -> torch.Tensor:
    probs = F.softmax(router_logits, dim=-1)
    avg_probs = probs.mean(dim=0)

    top_indices = torch.topk(probs, num_experts_per_tok, dim=-1).indices
    expert_mask = torch.zeros_like(probs).scatter_(-1, top_indices, 1.0)
    tokens_per_expert = expert_mask.mean(dim=0)

    return num_experts * (avg_probs * tokens_per_expert).sum()


def explain_moe() -> str:
    return """
Mixture of Experts (MoE)

MoE replaces dense FFN with sparse expert selection:
  - N experts, each a full FFN
  - Router selects top-k experts per token
  - Only selected experts compute, reducing total FLOPs

Example (Mixtral 8x7B):
  - 8 experts, each ~7B parameters
  - Top-2 routing: 2 experts active per token
  - Effective compute: ~14B params (2x7B)
  - Total params: ~47B

Routing:
  1. Router: linear projection hidden_dim -> num_experts
  2. Softmax over experts
  3. Select top-k by probability
  4. Normalize selected weights to sum to 1
  5. Weighted sum of expert outputs

Load balancing:
  - Want tokens distributed evenly across experts
  - Auxiliary loss encourages balance
  - Prevents expert collapse (one expert gets all tokens)

For inference:
  - Router is cheap (small linear layer)
  - Only compute 2 expert FFNs instead of 1 large one
  - Memory: need all expert weights loaded
  - Can offload unused experts to CPU/NVMe
"""


if __name__ == "__main__":
    print(explain_moe())

    print("\n" + "=" * 60)
    print("MoE Layer Demo")
    print("-" * 60)

    config = MoEConfig(
        hidden_dim=256,
        expert_dim=512,
        num_experts=8,
        num_experts_per_tok=2,
    )

    moe = MoELayer(config)

    x = torch.randn(2, 16, 256)
    output = moe(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts[0].parameters())

    print(f"\nTotal params:        {total_params:,}")
    print(f"Params per expert:   {expert_params:,}")
    print(f"Active params/token: {expert_params * config.num_experts_per_tok:,}")
