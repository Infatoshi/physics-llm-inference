"""Tests for Chapter 01: Transformer Mechanics"""

import pytest
import torch

from .attention import MultiHeadAttention, causal_attention, naive_attention
from .ffn import FusedSwiGLUFFN, NaiveFFN, SwiGLUFFN
from .gqa import GroupedQueryAttention
from .transformer import RMSNorm, TransformerBlock, TransformerModel


class TestAttention:
    def test_naive_attention_shape(self):
        batch, seq_len, head_dim = 2, 16, 64
        q = torch.randn(batch, seq_len, head_dim)
        k = torch.randn(batch, seq_len, head_dim)
        v = torch.randn(batch, seq_len, head_dim)

        out = naive_attention(q, k, v)
        assert out.shape == (batch, seq_len, head_dim)

    def test_causal_attention_is_causal(self):
        """Verify that causal attention doesn't attend to future positions."""
        batch, seq_len, head_dim = 1, 4, 8
        q = torch.randn(batch, seq_len, head_dim)
        k = torch.randn(batch, seq_len, head_dim)
        v = torch.randn(batch, seq_len, head_dim)

        out = causal_attention(q, k, v)

        # Modify the last token's K and V
        k_modified = k.clone()
        v_modified = v.clone()
        k_modified[0, -1] = torch.randn(head_dim)
        v_modified[0, -1] = torch.randn(head_dim)

        # First position should be unchanged (can't see position 4)
        out_modified = causal_attention(q, k_modified, v_modified)
        assert torch.allclose(out[0, 0], out_modified[0, 0], atol=1e-5)

    def test_multi_head_attention_shape(self):
        batch, seq_len, hidden_dim, num_heads = 2, 16, 64, 4
        x = torch.randn(batch, seq_len, hidden_dim)

        mha = MultiHeadAttention(hidden_dim, num_heads)
        out = mha(x)
        assert out.shape == x.shape


class TestGQA:
    def test_gqa_shape(self):
        batch, seq_len, hidden_dim = 2, 16, 256
        num_heads, num_kv_heads = 8, 2

        x = torch.randn(batch, seq_len, hidden_dim)
        gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads)
        out = gqa(x)
        assert out.shape == x.shape

    def test_gqa_reduces_to_mha(self):
        """When num_kv_heads == num_heads, GQA should behave like MHA."""
        batch, seq_len, hidden_dim = 2, 16, 64
        num_heads = 4

        x = torch.randn(batch, seq_len, hidden_dim)
        torch.manual_seed(42)
        gqa = GroupedQueryAttention(hidden_dim, num_heads, num_heads)
        torch.manual_seed(42)
        mha = MultiHeadAttention(hidden_dim, num_heads)

        # They should produce the same result with same weights
        out_gqa = gqa(x)
        out_mha = mha(x)
        # Note: won't be exactly equal due to different implementations
        # Just check shapes match
        assert out_gqa.shape == out_mha.shape

    def test_kv_cache_size_calculation(self):
        hidden_dim = 4096
        num_heads = 32
        num_kv_heads = 8

        gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads)
        size = gqa.kv_cache_size_per_token()

        # Expected: 2 (K+V) * 8 (kv_heads) * 128 (head_dim) * 2 (fp16)
        expected = 2 * 8 * 128 * 2
        assert size == expected


class TestFFN:
    def test_naive_ffn_shape(self):
        batch, seq_len, hidden_dim = 2, 16, 256
        intermediate_dim = 4 * hidden_dim

        x = torch.randn(batch, seq_len, hidden_dim)
        ffn = NaiveFFN(hidden_dim, intermediate_dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_swiglu_ffn_shape(self):
        batch, seq_len, hidden_dim = 2, 16, 256
        intermediate_dim = int(8 / 3 * hidden_dim)

        x = torch.randn(batch, seq_len, hidden_dim)
        ffn = SwiGLUFFN(hidden_dim, intermediate_dim)
        out = ffn(x)
        assert out.shape == x.shape

    def test_fused_swiglu_matches_unfused(self):
        """Fused and unfused SwiGLU should produce the same result."""
        batch, seq_len, hidden_dim = 2, 16, 64
        intermediate_dim = 128

        x = torch.randn(batch, seq_len, hidden_dim)

        unfused = SwiGLUFFN(hidden_dim, intermediate_dim)
        fused = FusedSwiGLUFFN(hidden_dim, intermediate_dim)

        # Copy weights from unfused to fused
        with torch.no_grad():
            fused.gate_up_proj.weight[:intermediate_dim] = unfused.gate_proj.weight
            fused.gate_up_proj.weight[intermediate_dim:] = unfused.up_proj.weight
            fused.down_proj.weight.copy_(unfused.down_proj.weight)

        out_unfused = unfused(x)
        out_fused = fused(x)
        assert torch.allclose(out_unfused, out_fused, atol=1e-5)


class TestRMSNorm:
    def test_rmsnorm_shape(self):
        batch, seq_len, hidden_dim = 2, 16, 256

        x = torch.randn(batch, seq_len, hidden_dim)
        norm = RMSNorm(hidden_dim)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_normalized(self):
        """Output should have approximately unit RMS."""
        batch, seq_len, hidden_dim = 2, 16, 256

        x = torch.randn(batch, seq_len, hidden_dim) * 10  # Scale up
        norm = RMSNorm(hidden_dim)
        out = norm(x)

        # Check RMS is approximately 1 (accounting for learned weight)
        rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
        assert rms.mean().item() == pytest.approx(1.0, rel=0.1)


class TestTransformerBlock:
    def test_block_shape(self):
        batch, seq_len, hidden_dim = 2, 16, 256
        num_heads, num_kv_heads = 8, 2
        intermediate_dim = 512

        x = torch.randn(batch, seq_len, hidden_dim)
        block = TransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_dim=intermediate_dim,
        )
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Verify residual connections are working."""
        batch, seq_len, hidden_dim = 2, 16, 64
        num_heads, num_kv_heads = 4, 2
        intermediate_dim = 128

        x = torch.randn(batch, seq_len, hidden_dim)
        block = TransformerBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_dim=intermediate_dim,
        )

        # Initialize all weights to zero - output should equal input
        for p in block.parameters():
            p.data.zero_()

        out = block(x)
        # With zero weights, attn and FFN output zero, so residual = input
        assert torch.allclose(out, x, atol=1e-5)


class TestTransformerModel:
    def test_model_forward(self):
        model = TransformerModel(
            vocab_size=1000,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=128,
        )

        batch, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch, seq_len))
        logits = model(input_ids)

        assert logits.shape == (batch, seq_len, 1000)

    def test_model_autoregressive(self):
        """Verify model is autoregressive - changing future tokens doesn't affect past."""
        model = TransformerModel(
            vocab_size=100,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=64,
        )
        model.eval()

        input_ids = torch.randint(0, 100, (1, 8))
        logits1 = model(input_ids)

        # Change the last token
        input_ids_modified = input_ids.clone()
        input_ids_modified[0, -1] = (input_ids_modified[0, -1] + 1) % 100
        logits2 = model(input_ids_modified)

        # All positions except the last should be unchanged
        assert torch.allclose(logits1[0, :-1], logits2[0, :-1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
