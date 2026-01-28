from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TensorParallelConfig:
    world_size: int = 1
    rank: int = 0
    hidden_dim: int = 4096
    intermediate_dim: int = 14336


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int = 1,
        rank: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        assert out_features % world_size == 0
        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition)) if bias else None

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.linear(x, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int = 1,
        rank: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank

        assert in_features % world_size == 0
        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = nn.functional.linear(x, self.weight, None)
        return output


class TensorParallelMLP(nn.Module):
    def __init__(self, config: TensorParallelConfig):
        super().__init__()
        self.config = config

        self.gate_proj = ColumnParallelLinear(
            config.hidden_dim,
            config.intermediate_dim,
            config.world_size,
            config.rank,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_dim,
            config.intermediate_dim,
            config.world_size,
            config.rank,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_dim,
            config.hidden_dim,
            config.world_size,
            config.rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output


def compute_tp_memory_savings(
    hidden_dim: int,
    intermediate_dim: int,
    world_size: int,
    dtype_bytes: int = 2,
) -> dict:
    dense_mlp_params = (
        hidden_dim * intermediate_dim +
        hidden_dim * intermediate_dim +
        intermediate_dim * hidden_dim
    )
    dense_memory = dense_mlp_params * dtype_bytes

    tp_params_per_gpu = dense_mlp_params / world_size
    tp_memory_per_gpu = tp_params_per_gpu * dtype_bytes

    return {
        "dense_params": dense_mlp_params,
        "dense_memory_mb": dense_memory / 1024 / 1024,
        "tp_params_per_gpu": tp_params_per_gpu,
        "tp_memory_per_gpu_mb": tp_memory_per_gpu / 1024 / 1024,
        "memory_reduction": world_size,
    }


def explain_tensor_parallelism() -> str:
    return """
Tensor Parallelism

Goal: Split model across GPUs to fit larger models

Key insight: Matrix multiply can be split along different dimensions

Column Parallel (split output dim):
  Y = XW where W is [in, out]
  Split W into [in, out/N] per GPU
  Each GPU computes partial output
  Result: Y_i has shape [batch, out/N]

Row Parallel (split input dim):
  Y = XW where W is [in, out]
  Split W into [in/N, out] per GPU
  Split X into [batch, in/N] per GPU
  Each GPU computes partial sum
  All-reduce to get final Y

For MLP: gate_proj -> up_proj -> down_proj
  Column parallel for gate_proj, up_proj
  Row parallel for down_proj
  One all-reduce per MLP layer

For Attention: QKV projection -> attention -> output projection
  Column parallel for QKV
  Row parallel for output
  One all-reduce per attention layer

Communication:
  - Forward: one all-reduce per layer
  - Backward: one all-reduce per layer
  - Communication volume: O(batch * seq * hidden)

Trade-offs:
  - Pro: Linear memory reduction with GPU count
  - Con: Communication overhead
  - Con: Requires fast interconnect (NVLink)
"""


if __name__ == "__main__":
    print(explain_tensor_parallelism())

    print("\n" + "=" * 60)
    print("Tensor Parallel MLP Demo")
    print("-" * 60)

    for world_size in [1, 2, 4, 8]:
        savings = compute_tp_memory_savings(
            hidden_dim=4096,
            intermediate_dim=14336,
            world_size=world_size,
        )
        print(f"\nWorld size: {world_size}")
        print(f"  Dense memory:  {savings['dense_memory_mb']:.1f} MB")
        print(f"  Per-GPU memory: {savings['tp_memory_per_gpu_mb']:.1f} MB")

    print("\n" + "-" * 60)
    print("Single GPU simulation:")

    config = TensorParallelConfig(
        world_size=1,
        rank=0,
        hidden_dim=256,
        intermediate_dim=512,
    )

    mlp = TensorParallelMLP(config)
    x = torch.randn(2, 16, 256)
    output = mlp(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
