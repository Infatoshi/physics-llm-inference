import pytest
import torch

from .attention_memory import (
    attention_arithmetic_intensity,
    attention_flops,
    attention_memory_bytes,
    naive_attention,
)
from .flash_attention import (
    FlashAttentionConfig,
    flash_attention_forward,
    flash_attention_memory_bytes,
)
from .online_softmax import (
    online_softmax,
    online_softmax_with_output,
    standard_softmax,
)


class TestAttentionMemory:
    def test_memory_bytes_calculation(self):
        stats = attention_memory_bytes(
            batch_size=1,
            num_heads=8,
            seq_len=1024,
            head_dim=64,
            dtype_bytes=2,
        )
        expected_qk = 1 * 8 * 1024 * 1024 * 2
        assert stats.qk_bytes == expected_qk
        assert stats.total_bytes > 0

    def test_memory_scales_quadratically(self):
        stats_small = attention_memory_bytes(1, 8, 1024, 64, 2)
        stats_large = attention_memory_bytes(1, 8, 2048, 64, 2)
        ratio = stats_large.qk_bytes / stats_small.qk_bytes
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_flops_calculation(self):
        flops = attention_flops(
            batch_size=1,
            num_heads=8,
            seq_len=1024,
            head_dim=64,
        )
        assert flops > 0
        min_flops = 2 * 1024 * 1024 * 64
        assert flops >= min_flops

    def test_arithmetic_intensity_increases_with_seq_len(self):
        ai_small = attention_arithmetic_intensity(512, 64)
        ai_large = attention_arithmetic_intensity(4096, 64)
        assert ai_large > ai_small


class TestNaiveAttention:
    def test_output_shape(self):
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)
        out = naive_attention(q, k, v)
        assert out.shape == (B, H, N, D)

    def test_attention_is_normalized(self):
        B, H, N, D = 1, 1, 32, 16
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.ones(B, H, N, D)
        out = naive_attention(q, k, v)
        torch.testing.assert_close(out, torch.ones_like(out), rtol=1e-4, atol=1e-4)


class TestOnlineSoftmax:
    def test_standard_softmax_sums_to_one(self):
        x = torch.randn(4, 8, 32)
        result = standard_softmax(x)
        sums = result.sum(dim=-1)
        expected = torch.ones_like(sums)
        torch.testing.assert_close(sums, expected, rtol=1e-4, atol=1e-4)

    def test_online_softmax_matches_standard(self):
        x = torch.randn(4, 8, 64)
        standard = standard_softmax(x)
        online = online_softmax(x)
        torch.testing.assert_close(online, standard, rtol=1e-4, atol=1e-4)

    def test_online_softmax_1d(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        standard = standard_softmax(x)
        online = online_softmax(x)
        torch.testing.assert_close(online, standard, rtol=1e-4, atol=1e-4)

    def test_online_softmax_numerical_stability(self):
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        online = online_softmax(x)
        assert not torch.isnan(online).any()
        assert not torch.isinf(online).any()


class TestOnlineSoftmaxWithOutput:
    def test_output_shape(self):
        x = torch.randn(2, 4, 32)
        v = torch.randn(2, 4, 32, 16)
        o, d = online_softmax_with_output(x, v)
        assert o.shape == (2, 4, 16)
        assert d.shape == (2, 4)

    def test_correctness(self):
        x = torch.randn(2, 4, 32)
        v = torch.randn(2, 4, 32, 16)

        attn = standard_softmax(x)
        expected = torch.einsum('...n,...nd->...d', attn, v)

        online_out, _ = online_softmax_with_output(x, v)

        torch.testing.assert_close(online_out, expected, rtol=1e-3, atol=1e-3)


class TestFlashAttentionConfig:
    def test_default_config(self):
        config = FlashAttentionConfig()
        assert config.block_q == 64
        assert config.block_k == 64
        assert config.num_warps == 4

    def test_custom_config(self):
        config = FlashAttentionConfig(block_q=32, block_k=32)
        assert config.block_q == 32
        assert config.block_k == 32


class TestFlashAttentionMemory:
    def test_memory_calculation(self):
        mem = flash_attention_memory_bytes(
            batch_size=1,
            num_heads=8,
            seq_len=1024,
            head_dim=64,
        )
        assert "hbm_bytes" in mem
        assert "sram_bytes_per_block" in mem
        assert mem["hbm_bytes"] > 0

    def test_flash_uses_less_memory_than_naive(self):
        mem = flash_attention_memory_bytes(
            batch_size=1,
            num_heads=8,
            seq_len=1024,
            head_dim=64,
        )
        assert mem["hbm_bytes"] < mem["naive_hbm_bytes"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFlashAttentionForward:
    def test_output_shape(self):
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        out = flash_attention_forward(q, k, v)
        assert out.shape == (B, H, N, D)

    def test_matches_naive_attention(self):
        B, H, N, D = 2, 4, 128, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        naive_out = naive_attention(q, k, v)
        flash_out = flash_attention_forward(q, k, v)

        torch.testing.assert_close(flash_out, naive_out, rtol=0.01, atol=0.01)

    def test_larger_sequence(self):
        B, H, N, D = 1, 8, 512, 64
        q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        naive_out = naive_attention(q, k, v)
        flash_out = flash_attention_forward(q, k, v)

        torch.testing.assert_close(flash_out, naive_out, rtol=0.02, atol=0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
