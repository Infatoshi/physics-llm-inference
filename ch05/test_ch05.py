import pytest
import torch

from .memory_coalescing import (
    AccessPatternResult,
    coalesced_access,
    strided_access,
)
from .shared_memory import (
    compute_tile_config,
    demonstrate_bank_conflicts,
    max_blocks_by_shared_memory,
    shared_memory_requirements,
    tiled_reduce,
)
from .tensor_cores import (
    tensor_core_info,
)

try:
    from .triton_matmul import TRITON_AVAILABLE, triton_matmul
except ImportError:
    TRITON_AVAILABLE = False


class TestMemoryCoalescing:
    def test_coalesced_access_shape(self):
        data = torch.randn(1024)
        result = coalesced_access(data)
        assert result.shape == data.shape

    def test_strided_access_shape(self):
        data = torch.randn(1024)
        stride = 32
        result = strided_access(data, stride)
        expected_size = 1024 // stride
        assert result.shape[0] == expected_size

    def test_strided_access_values(self):
        data = torch.arange(128).float()
        result = strided_access(data, stride=4)
        expected = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28]).float()
        torch.testing.assert_close(result[:8], expected)

    def test_access_pattern_result_fields(self):
        result = AccessPatternResult(
            pattern_name="test",
            time_us=100.0,
            bandwidth_gbps=500.0,
            efficiency=0.5,
        )
        assert result.pattern_name == "test"
        assert result.time_us == 100.0
        assert result.bandwidth_gbps == 500.0
        assert result.efficiency == 0.5


class TestSharedMemory:
    def test_tile_config_basic(self):
        config = compute_tile_config(n=1024, tile_size=32)
        assert config.tile_size == 32
        assert config.num_tiles == 32
        assert config.shared_memory_bytes == 128

    def test_tile_config_non_divisible(self):
        config = compute_tile_config(n=1000, tile_size=32)
        assert config.num_tiles == 32

    def test_tiled_reduce_correctness(self):
        data = torch.ones(1024)
        result = tiled_reduce(data, tile_size=256)
        assert result.item() == 1024.0

    def test_tiled_reduce_random(self):
        data = torch.randn(512)
        expected = data.sum()
        result = tiled_reduce(data, tile_size=64)
        torch.testing.assert_close(result, expected)

    def test_bank_conflicts_info(self):
        info = demonstrate_bank_conflicts()
        assert "shared_memory_banks" in info
        assert info["shared_memory_banks"] == 32
        assert "bank_width_bytes" in info

    def test_shared_memory_requirements(self):
        smem = shared_memory_requirements(
            threads_per_block=256,
            elements_per_thread=4,
            dtype_bytes=4,
        )
        assert smem == 256 * 4 * 4

    def test_max_blocks_zero_shared(self):
        max_blocks = max_blocks_by_shared_memory(0)
        assert max_blocks == float('inf')

    def test_max_blocks_by_shared_memory(self):
        max_blocks = max_blocks_by_shared_memory(
            shared_per_block=32 * 1024,
            shared_per_sm=100 * 1024,
        )
        assert max_blocks == 3


class TestTensorCores:
    def test_tensor_core_info_fields(self):
        info = tensor_core_info()
        assert "operation" in info
        assert "input_types" in info
        assert "ampere_shape" in info


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonMatmul:
    def test_triton_matmul_correctness_small(self):
        a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        expected = torch.matmul(a, b)
        result = triton_matmul(a, b, block_m=32, block_n=32, block_k=32)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_triton_matmul_correctness_medium(self):
        a = torch.randn(256, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        expected = torch.matmul(a, b)
        result = triton_matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)

    def test_triton_matmul_non_square(self):
        a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
        b = torch.randn(256, 64, device="cuda", dtype=torch.float16)
        expected = torch.matmul(a, b)
        result = triton_matmul(a, b)
        torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUBenchmarks:
    def test_tensor_core_speedup_exists(self):
        from .tensor_cores import benchmark_tensor_cores
        result = benchmark_tensor_cores(size=1024, warmup=3, iterations=10)
        assert result is not None
        assert result.fp16_us > 0
        assert result.fp32_us > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
