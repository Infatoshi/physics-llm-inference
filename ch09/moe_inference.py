from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class ExpertUsageStats:
    expert_id: int
    tokens_routed: int = 0
    total_weight: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class ExpertCache:
    def __init__(
        self,
        max_experts_in_memory: int,
        num_total_experts: int,
    ):
        self.max_experts = max_experts_in_memory
        self.num_total_experts = num_total_experts
        self.cached_experts: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.stats: dict[int, ExpertUsageStats] = {
            i: ExpertUsageStats(expert_id=i) for i in range(num_total_experts)
        }

    def get_expert(self, expert_id: int) -> torch.Tensor | None:
        if expert_id in self.cached_experts:
            self.cached_experts.move_to_end(expert_id)
            self.stats[expert_id].cache_hits += 1
            return self.cached_experts[expert_id]

        self.stats[expert_id].cache_misses += 1
        return None

    def add_expert(self, expert_id: int, weights: torch.Tensor) -> int | None:
        evicted = None

        if len(self.cached_experts) >= self.max_experts:
            evicted_id, _ = self.cached_experts.popitem(last=False)
            evicted = evicted_id

        self.cached_experts[expert_id] = weights
        return evicted

    def get_cache_hit_rate(self) -> float:
        total_hits = sum(s.cache_hits for s in self.stats.values())
        total_accesses = total_hits + sum(s.cache_misses for s in self.stats.values())
        return total_hits / total_accesses if total_accesses > 0 else 0.0

    def get_cached_expert_ids(self) -> list[int]:
        return list(self.cached_experts.keys())


@dataclass
class MoEInferenceConfig:
    num_experts: int = 8
    num_experts_per_tok: int = 2
    max_experts_in_gpu: int = 4
    enable_expert_offload: bool = False


class MoEInferenceEngine:
    def __init__(self, config: MoEInferenceConfig):
        self.config = config
        self.expert_cache = ExpertCache(
            max_experts_in_memory=config.max_experts_in_gpu,
            num_total_experts=config.num_experts,
        )
        self.batch_expert_usage: dict[int, int] = {}

    def plan_expert_execution(
        self,
        expert_indices: torch.Tensor,
    ) -> dict[str, list[int]]:
        unique_experts = expert_indices.unique().tolist()

        in_cache = []
        need_load = []

        for expert_id in unique_experts:
            if self.expert_cache.get_expert(expert_id) is not None:
                in_cache.append(expert_id)
            else:
                need_load.append(expert_id)

        return {
            "in_cache": in_cache,
            "need_load": need_load,
            "total_unique": len(unique_experts),
        }

    def update_batch_stats(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> None:
        for expert_id in range(self.config.num_experts):
            mask = expert_indices == expert_id
            count = mask.sum().item()
            if count > 0:
                self.expert_cache.stats[expert_id].tokens_routed += count
                self.expert_cache.stats[expert_id].total_weight += expert_weights[mask].sum().item()

    def get_load_balance_metrics(self) -> dict:
        token_counts = [s.tokens_routed for s in self.expert_cache.stats.values()]
        total_tokens = sum(token_counts)

        if total_tokens == 0:
            return {"balance_ratio": 1.0, "max_load": 0, "min_load": 0}

        expected = total_tokens / self.config.num_experts
        max_load = max(token_counts)
        min_load = min(token_counts)

        balance_ratio = min_load / max_load if max_load > 0 else 1.0

        return {
            "balance_ratio": balance_ratio,
            "max_load": max_load,
            "min_load": min_load,
            "expected": expected,
            "std_dev": torch.tensor(token_counts).float().std().item(),
        }


def explain_moe_inference() -> str:
    return """
MoE Inference Optimization

Challenge: MoE models have many expert weights
  - Mixtral 8x7B: 8 experts, ~47B total params
  - Qwen3-30B-A3B: 128 experts, ~30B total params
  - Cannot fit all experts in GPU memory

Strategies:

1. Expert Parallelism
   - Distribute experts across GPUs
   - All-to-all communication for routing
   - Each GPU has subset of experts

2. Expert Offloading
   - Keep only active experts in GPU
   - Offload others to CPU/NVMe
   - Prefetch based on router predictions

3. Expert Caching (LRU)
   - Cache recently used experts in GPU
   - Evict least recently used
   - Works well when expert usage is skewed

4. Speculative Expert Loading
   - Run router ahead of time
   - Prefetch predicted experts
   - Overlap loading with computation

Batch Processing:
  - Aggregate tokens by expert
  - Process each expert's tokens together
  - Reduces kernel launch overhead
  - Improves memory access patterns

Expert Load Balancing:
  - Unbalanced routing wastes GPU resources
  - Aux loss during training helps
  - Can affect inference throughput
"""


if __name__ == "__main__":
    print(explain_moe_inference())

    print("\n" + "=" * 60)
    print("MoE Inference Engine Demo")
    print("-" * 60)

    config = MoEInferenceConfig(
        num_experts=8,
        num_experts_per_tok=2,
        max_experts_in_gpu=4,
    )
    engine = MoEInferenceEngine(config)

    expert_indices = torch.tensor([
        [0, 1], [0, 2], [1, 3], [2, 3],
        [0, 4], [1, 5], [4, 5], [6, 7],
    ])
    expert_weights = torch.rand_like(expert_indices.float())
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

    plan = engine.plan_expert_execution(expert_indices)
    print("Execution plan:")
    print(f"  In cache:    {plan['in_cache']}")
    print(f"  Need load:   {plan['need_load']}")
    print(f"  Total unique: {plan['total_unique']}")

    engine.update_batch_stats(expert_indices, expert_weights)
    metrics = engine.get_load_balance_metrics()
    print("\nLoad balance metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
