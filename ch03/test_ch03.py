import pytest
import torch

from .batching_benchmark import find_transition_batch_size
from .gemm_benchmark import gemm_bytes, gemm_flops
from .gemv_benchmark import gemv_bytes, gemv_flops
from .roofline import (
    A100_80GB,
    RTX_3090,
    arithmetic_intensity,
    batched_gemv_arithmetic_intensity,
    gemm_arithmetic_intensity,
    gemv_arithmetic_intensity,
    is_compute_bound,
    ridge_point,
)


class TestGEMMCalculations:
    def test_gemm_flops_square(self):
        flops = gemm_flops(1024, 1024, 1024)
        assert flops == 2 * 1024 * 1024 * 1024

    def test_gemm_flops_rectangular(self):
        m, n, k = 512, 1024, 2048
        flops = gemm_flops(m, n, k)
        assert flops == 2 * m * n * k

    def test_gemm_bytes_fp16(self):
        m, n, k = 1024, 1024, 1024
        bytes_moved = gemm_bytes(m, n, k, torch.float16)
        expected = (m * k + k * n + m * n) * 2
        assert bytes_moved == expected

    def test_gemm_bytes_fp32(self):
        m, n, k = 1024, 1024, 1024
        bytes_moved = gemm_bytes(m, n, k, torch.float32)
        expected = (m * k + k * n + m * n) * 4
        assert bytes_moved == expected


class TestGEMVCalculations:
    def test_gemv_flops(self):
        m, k = 4096, 4096
        flops = gemv_flops(m, k)
        assert flops == 2 * m * k

    def test_gemv_bytes_fp16(self):
        m, k = 4096, 4096
        bytes_moved = gemv_bytes(m, k, torch.float16)
        expected = (m * k + k + m) * 2
        assert bytes_moved == expected


class TestArithmeticIntensity:
    def test_basic_calculation(self):
        flops = 1000
        bytes_moved = 100
        ai = arithmetic_intensity(flops, bytes_moved)
        assert ai == 10.0

    def test_gemm_ai_is_high(self):
        ai = gemm_arithmetic_intensity(4096, 4096, 4096)
        assert ai > 100

    def test_gemv_ai_is_low(self):
        ai = gemv_arithmetic_intensity(4096, 4096)
        assert ai < 2

    def test_batched_gemv_increases_ai(self):
        ai_1 = batched_gemv_arithmetic_intensity(1, 4096, 4096)
        ai_16 = batched_gemv_arithmetic_intensity(16, 4096, 4096)
        ai_256 = batched_gemv_arithmetic_intensity(256, 4096, 4096)
        assert ai_1 < ai_16 < ai_256


class TestRooflineModel:
    def test_ridge_point_rtx_3090(self):
        rp = ridge_point(RTX_3090)
        assert 30 < rp < 50

    def test_ridge_point_a100(self):
        rp = ridge_point(A100_80GB)
        assert 100 < rp < 200

    def test_gemm_is_compute_bound(self):
        ai = gemm_arithmetic_intensity(4096, 4096, 4096)
        assert is_compute_bound(ai, RTX_3090)
        assert is_compute_bound(ai, A100_80GB)

    def test_gemv_is_memory_bound(self):
        ai = gemv_arithmetic_intensity(4096, 4096)
        assert not is_compute_bound(ai, RTX_3090)
        assert not is_compute_bound(ai, A100_80GB)


class TestTransitionBatchSize:
    def test_transition_exists(self):
        batch = find_transition_batch_size(
            4096, 4096,
            RTX_3090.peak_tflops,
            RTX_3090.memory_bandwidth_gbps,
        )
        assert batch > 1
        assert batch <= 1024

    def test_small_batch_is_memory_bound(self):
        ai = batched_gemv_arithmetic_intensity(1, 4096, 4096)
        assert not is_compute_bound(ai, RTX_3090)

    def test_large_batch_is_compute_bound(self):
        ai = batched_gemv_arithmetic_intensity(512, 4096, 4096)
        assert is_compute_bound(ai, RTX_3090)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
