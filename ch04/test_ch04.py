import pytest
import torch

from .gpu_architecture import (
    AMPERE_SPECS,
    GPUSpec,
    get_gpu_spec,
    theoretical_occupancy,
    threads_to_grid_block,
    warp_efficiency,
)


class TestGPUSpec:
    def test_rtx_3090_spec_exists(self):
        assert "RTX 3090" in AMPERE_SPECS
        spec = AMPERE_SPECS["RTX 3090"]
        assert spec.num_sms == 82
        assert spec.warp_size == 32

    def test_a100_spec_exists(self):
        assert "A100 80GB" in AMPERE_SPECS
        spec = AMPERE_SPECS["A100 80GB"]
        assert spec.num_sms == 108

    def test_spec_has_required_fields(self):
        spec = AMPERE_SPECS["RTX 3090"]
        assert spec.max_threads_per_sm > 0
        assert spec.max_threads_per_block > 0
        assert spec.shared_memory_per_sm_kb > 0
        assert spec.registers_per_sm > 0


class TestOccupancy:
    def test_full_occupancy(self):
        spec = AMPERE_SPECS["RTX 3090"]
        occ = theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
            spec=spec,
        )
        assert 0 < occ <= 1.0

    def test_more_threads_higher_occupancy(self):
        spec = AMPERE_SPECS["RTX 3090"]
        occ_small = theoretical_occupancy(
            threads_per_block=64,
            registers_per_thread=32,
            shared_memory_per_block=0,
            spec=spec,
        )
        occ_large = theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
            spec=spec,
        )
        assert occ_large >= occ_small

    def test_high_registers_reduces_occupancy(self):
        spec = AMPERE_SPECS["RTX 3090"]
        occ_low_regs = theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=32,
            shared_memory_per_block=0,
            spec=spec,
        )
        occ_high_regs = theoretical_occupancy(
            threads_per_block=256,
            registers_per_thread=128,
            shared_memory_per_block=0,
            spec=spec,
        )
        assert occ_low_regs >= occ_high_regs


class TestWarpEfficiency:
    def test_full_warp(self):
        eff = warp_efficiency(32, 32)
        assert eff == 1.0

    def test_half_warp(self):
        eff = warp_efficiency(16, 32)
        assert eff == 0.5

    def test_single_thread(self):
        eff = warp_efficiency(1, 32)
        assert eff == 1/32


class TestGridBlock:
    def test_exact_fit(self):
        grid, block = threads_to_grid_block(256, 256)
        assert grid == (1,)
        assert block == (256,)

    def test_multiple_blocks(self):
        grid, block = threads_to_grid_block(1024, 256)
        assert grid == (4,)
        assert block == (256,)

    def test_partial_block(self):
        grid, block = threads_to_grid_block(300, 256)
        assert grid == (2,)
        assert block == (256,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUDetection:
    def test_get_gpu_spec_returns_spec(self):
        spec = get_gpu_spec()
        assert spec is not None
        assert isinstance(spec, GPUSpec)
        assert spec.num_sms > 0

    def test_detected_spec_has_warp_size_32(self):
        spec = get_gpu_spec()
        assert spec.warp_size == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
