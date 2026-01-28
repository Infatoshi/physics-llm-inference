import pytest
import torch

from .moe_inference import (
    ExpertCache,
    MoEInferenceConfig,
    MoEInferenceEngine,
)
from .moe_layer import (
    ExpertLayer,
    MoEConfig,
    MoELayer,
    Router,
    expert_load_balance_loss,
)
from .nccl_primitives import (
    AllGatherConfig,
    AllReduceConfig,
    compute_communication_overlap_potential,
    compute_ring_all_reduce_time,
    simulate_all_gather,
    simulate_all_reduce,
)
from .tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelConfig,
    TensorParallelMLP,
    compute_tp_memory_savings,
)


class TestMoEConfig:
    def test_default_config(self):
        config = MoEConfig()
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2

    def test_custom_config(self):
        config = MoEConfig(num_experts=16, num_experts_per_tok=4)
        assert config.num_experts == 16
        assert config.num_experts_per_tok == 4


class TestRouter:
    def test_router_output_shape(self):
        config = MoEConfig(hidden_dim=64, num_experts=8, num_experts_per_tok=2)
        router = Router(config)

        x = torch.randn(4, 16, 64)
        x_flat = x.view(-1, 64)
        weights, indices, logits = router(x_flat)

        assert weights.shape == (64, 2)
        assert indices.shape == (64, 2)
        assert logits.shape == (64, 8)

    def test_router_weights_normalized(self):
        config = MoEConfig(hidden_dim=64, num_experts=8, num_experts_per_tok=2)
        router = Router(config)

        x = torch.randn(32, 64)
        weights, _, _ = router(x)

        sums = weights.sum(dim=-1)
        expected = torch.ones(32)
        torch.testing.assert_close(sums, expected, rtol=1e-4, atol=1e-4)


class TestExpertLayer:
    def test_expert_output_shape(self):
        expert = ExpertLayer(hidden_dim=64, expert_dim=256)

        x = torch.randn(8, 64)
        output = expert(x)

        assert output.shape == (8, 64)


class TestMoELayer:
    def test_moe_output_shape(self):
        config = MoEConfig(hidden_dim=64, expert_dim=256, num_experts=4)
        moe = MoELayer(config)

        x = torch.randn(2, 8, 64)
        output = moe(x)

        assert output.shape == (2, 8, 64)


class TestLoadBalanceLoss:
    def test_load_balance_loss_positive(self):
        router_logits = torch.randn(32, 8)
        loss = expert_load_balance_loss(router_logits, num_experts=8, num_experts_per_tok=2)
        assert loss.item() > 0


class TestExpertCache:
    def test_cache_initialization(self):
        cache = ExpertCache(max_experts_in_memory=4, num_total_experts=8)
        assert cache.max_experts == 4
        assert len(cache.stats) == 8

    def test_cache_add_and_get(self):
        cache = ExpertCache(max_experts_in_memory=2, num_total_experts=4)

        weights = torch.randn(64, 256)
        cache.add_expert(0, weights)

        retrieved = cache.get_expert(0)
        assert retrieved is not None
        torch.testing.assert_close(retrieved, weights)

    def test_cache_eviction(self):
        cache = ExpertCache(max_experts_in_memory=2, num_total_experts=4)

        cache.add_expert(0, torch.randn(64, 256))
        cache.add_expert(1, torch.randn(64, 256))

        evicted = cache.add_expert(2, torch.randn(64, 256))

        assert evicted == 0
        assert cache.get_expert(0) is None
        assert cache.get_expert(2) is not None


class TestMoEInferenceEngine:
    def test_plan_expert_execution(self):
        config = MoEInferenceConfig(num_experts=8, max_experts_in_gpu=4)
        engine = MoEInferenceEngine(config)

        expert_indices = torch.tensor([[0, 1], [2, 3], [4, 5]])
        plan = engine.plan_expert_execution(expert_indices)

        assert "in_cache" in plan
        assert "need_load" in plan
        assert plan["total_unique"] == 6


class TestColumnParallelLinear:
    def test_output_shape(self):
        layer = ColumnParallelLinear(
            in_features=256,
            out_features=1024,
            world_size=4,
            rank=0,
        )

        x = torch.randn(8, 256)
        output = layer(x)

        assert output.shape == (8, 256)

    def test_weight_shape(self):
        layer = ColumnParallelLinear(
            in_features=256,
            out_features=1024,
            world_size=4,
            rank=0,
        )

        assert layer.weight.shape == (256, 256)


class TestRowParallelLinear:
    def test_output_shape(self):
        layer = RowParallelLinear(
            in_features=1024,
            out_features=256,
            world_size=4,
            rank=0,
        )

        x = torch.randn(8, 256)
        output = layer(x)

        assert output.shape == (8, 256)


class TestTensorParallelMLP:
    def test_forward_pass(self):
        config = TensorParallelConfig(
            world_size=1,
            rank=0,
            hidden_dim=64,
            intermediate_dim=256,
        )

        mlp = TensorParallelMLP(config)
        x = torch.randn(2, 8, 64)
        output = mlp(x)

        assert output.shape == (2, 8, 64)


class TestTPMemorySavings:
    def test_memory_reduction(self):
        savings = compute_tp_memory_savings(
            hidden_dim=4096,
            intermediate_dim=14336,
            world_size=4,
        )

        assert savings["memory_reduction"] == 4
        assert savings["tp_memory_per_gpu_mb"] < savings["dense_memory_mb"]


class TestAllReduce:
    def test_all_reduce_simulation(self):
        config = AllReduceConfig(
            world_size=8,
            data_size_mb=10.0,
            bandwidth_gbps=600.0,
        )

        result = simulate_all_reduce(config)

        assert result["total_time_us"] > 0
        assert result["effective_bandwidth_gbps"] > 0
        assert result["bandwidth_efficiency"] <= 1.0


class TestAllGather:
    def test_all_gather_simulation(self):
        config = AllGatherConfig(
            world_size=8,
            data_size_per_gpu_mb=10.0,
            bandwidth_gbps=600.0,
        )

        result = simulate_all_gather(config)

        assert result["total_data_mb"] == 80.0
        assert result["total_time_us"] > 0


class TestRingAllReduce:
    def test_ring_time_calculation(self):
        time_us = compute_ring_all_reduce_time(
            data_size_bytes=100 * 1024 * 1024,
            world_size=8,
            bandwidth_gbps=600.0,
        )

        assert time_us > 0


class TestCommunicationOverlap:
    def test_compute_bound(self):
        result = compute_communication_overlap_potential(
            compute_time_us=1000,
            comm_time_us=100,
        )

        assert result["bottleneck"] == "compute"
        assert result["potential_speedup"] > 1.0

    def test_communication_bound(self):
        result = compute_communication_overlap_potential(
            compute_time_us=100,
            comm_time_us=1000,
        )

        assert result["bottleneck"] == "communication"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
