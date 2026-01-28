"""Tests for Chapter 02: The Generation Loop"""

import sys

import torch

sys.path.insert(0, '..')

from ch01.transformer import TransformerModel

from .cached_generation import (
    CachedGQA,
    CachedTransformerModel,
    LayerKVCache,
    cached_generate,
)
from .generation import naive_generate
from .kv_cache import GQAWithCache, KVCache, calculate_kv_cache_size


class TestKVCache:
    """Tests for KV cache implementations."""

    def test_kv_cache_create(self):
        """Test KVCache creation with correct shapes."""
        cache = KVCache.create(
            batch_size=2,
            max_seq_len=100,
            num_kv_heads=4,
            head_dim=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert cache.k_cache.shape == (2, 100, 4, 64)
        assert cache.v_cache.shape == (2, 100, 4, 64)
        assert cache.seq_len == 0

    def test_kv_cache_update(self):
        """Test KVCache update appends correctly."""
        cache = KVCache.create(
            batch_size=2,
            max_seq_len=100,
            num_kv_heads=4,
            head_dim=64,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # First update: 10 tokens
        k1 = torch.randn(2, 10, 4, 64)
        v1 = torch.randn(2, 10, 4, 64)
        k_full, v_full = cache.update(k1, v1)

        assert cache.seq_len == 10
        assert k_full.shape == (2, 10, 4, 64)
        assert torch.allclose(k_full, k1)

        # Second update: 1 token (decode step)
        k2 = torch.randn(2, 1, 4, 64)
        v2 = torch.randn(2, 1, 4, 64)
        k_full, v_full = cache.update(k2, v2)

        assert cache.seq_len == 11
        assert k_full.shape == (2, 11, 4, 64)
        # First 10 tokens should match k1
        assert torch.allclose(k_full[:, :10], k1)
        # Last token should match k2
        assert torch.allclose(k_full[:, 10:11], k2)

    def test_kv_cache_memory_calculation(self):
        """Test memory size calculation is correct."""
        cache = KVCache.create(
            batch_size=1,
            max_seq_len=1024,
            num_kv_heads=8,
            head_dim=128,
            device=torch.device("cpu"),
            dtype=torch.float16,
        )
        # 2 tensors * 1 * 1024 * 8 * 128 * 2 bytes = 4MB
        expected = 2 * 1 * 1024 * 8 * 128 * 2
        assert cache.memory_bytes() == expected


class TestLayerKVCache:
    """Tests for LayerKVCache used in cached generation."""

    def test_layer_kv_cache_update(self):
        """Test LayerKVCache update mechanism."""
        k = torch.zeros(2, 100, 4, 64)
        v = torch.zeros(2, 100, 4, 64)
        cache = LayerKVCache(k=k, v=v, seq_len=0)

        # Add 10 tokens
        k_new = torch.randn(2, 10, 4, 64)
        v_new = torch.randn(2, 10, 4, 64)
        k_full, v_full = cache.update(k_new, v_new)

        assert cache.seq_len == 10
        assert k_full.shape == (2, 10, 4, 64)


class TestCalculateKVCacheSize:
    """Tests for KV cache size calculations."""

    def test_single_layer_single_batch(self):
        """Test basic calculation."""
        sizes = calculate_kv_cache_size(
            batch_size=1,
            max_seq_len=1024,
            num_layers=1,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
        )
        # Per token per layer: 2 * 8 * 128 * 2 = 4096 bytes
        assert sizes["per_token_per_layer_bytes"] == 4096
        # Total: 4096 * 1024 * 1 = 4MB
        assert sizes["total_bytes"] == 4096 * 1024

    def test_scaling_with_batch(self):
        """Test that batch size scales linearly."""
        sizes_b1 = calculate_kv_cache_size(
            batch_size=1, max_seq_len=1024, num_layers=32,
            num_kv_heads=8, head_dim=128,
        )
        sizes_b8 = calculate_kv_cache_size(
            batch_size=8, max_seq_len=1024, num_layers=32,
            num_kv_heads=8, head_dim=128,
        )
        assert sizes_b8["total_bytes"] == 8 * sizes_b1["total_bytes"]

    def test_gqa_reduces_memory(self):
        """Test that GQA (fewer KV heads) reduces memory."""
        sizes_mha = calculate_kv_cache_size(
            batch_size=1, max_seq_len=1024, num_layers=32,
            num_kv_heads=32, head_dim=128,  # MHA: 32 KV heads
        )
        sizes_gqa = calculate_kv_cache_size(
            batch_size=1, max_seq_len=1024, num_layers=32,
            num_kv_heads=8, head_dim=128,  # GQA: 8 KV heads
        )
        # GQA should use 1/4 the memory
        assert sizes_gqa["total_bytes"] == sizes_mha["total_bytes"] // 4


class TestGQAWithCache:
    """Tests for GQA module with cache support."""

    def test_gqa_with_cache_shapes(self):
        """Test output shapes are correct."""
        attn = GQAWithCache(hidden_dim=256, num_heads=8, num_kv_heads=2)
        x = torch.randn(2, 10, 256)
        out, _ = attn(x, kv_cache=None)
        assert out.shape == (2, 10, 256)

    def test_gqa_decode_single_token(self):
        """Test decode with single token input."""
        attn = GQAWithCache(hidden_dim=256, num_heads=8, num_kv_heads=2)
        x = torch.randn(2, 1, 256)
        out, _ = attn(x, kv_cache=None)
        assert out.shape == (2, 1, 256)


class TestCachedGQA:
    """Tests for CachedGQA module."""

    def test_cached_gqa_prefill(self):
        """Test prefill phase populates cache correctly."""
        attn = CachedGQA(hidden_dim=256, num_heads=8, num_kv_heads=2)
        head_dim = 256 // 8

        # Create cache
        k = torch.zeros(2, 100, 2, head_dim)
        v = torch.zeros(2, 100, 2, head_dim)
        cache = LayerKVCache(k=k, v=v, seq_len=0)

        # Prefill with 10 tokens
        x = torch.randn(2, 10, 256)
        out = attn(x, cache=cache, start_pos=0)

        assert out.shape == (2, 10, 256)
        assert cache.seq_len == 10

    def test_cached_gqa_decode(self):
        """Test decode phase uses cache correctly."""
        attn = CachedGQA(hidden_dim=256, num_heads=8, num_kv_heads=2)
        head_dim = 256 // 8

        # Create and populate cache
        k = torch.zeros(2, 100, 2, head_dim)
        v = torch.zeros(2, 100, 2, head_dim)
        cache = LayerKVCache(k=k, v=v, seq_len=0)

        # Prefill
        x_prefill = torch.randn(2, 10, 256)
        attn(x_prefill, cache=cache, start_pos=0)

        # Decode single token
        x_decode = torch.randn(2, 1, 256)
        out = attn(x_decode, cache=cache, start_pos=10)

        assert out.shape == (2, 1, 256)
        assert cache.seq_len == 11


class TestCachedTransformerModel:
    """Tests for full cached transformer model."""

    def test_model_forward_no_cache(self):
        """Test forward pass without cache."""
        model = CachedTransformerModel(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            num_kv_heads=2,
            intermediate_dim=512,
        )
        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids, caches=None)
        assert logits.shape == (2, 10, 1000)

    def test_model_forward_with_cache(self):
        """Test forward pass with cache."""
        model = CachedTransformerModel(
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
            num_kv_heads=2,
            intermediate_dim=512,
        )
        caches = model.create_caches(
            batch_size=2, max_seq_len=100,
            device=torch.device("cpu"), dtype=torch.float32
        )

        # Prefill
        input_ids = torch.randint(0, 1000, (2, 10))
        logits = model(input_ids, caches=caches, start_pos=0)
        assert logits.shape == (2, 10, 1000)
        assert all(c.seq_len == 10 for c in caches)

        # Decode
        next_token = torch.randint(0, 1000, (2, 1))
        logits = model(next_token, caches=caches, start_pos=10)
        assert logits.shape == (2, 1, 1000)
        assert all(c.seq_len == 11 for c in caches)


class TestCachedGenerate:
    """Tests for cached generation function."""

    def test_cached_generate_output_length(self):
        """Test that generation produces correct number of tokens."""
        model = CachedTransformerModel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=256,
        )
        input_ids = torch.randint(0, 1000, (1, 5))
        max_new_tokens = 10

        tokens, timings = cached_generate(model, input_ids, max_new_tokens)

        assert tokens.shape == (1, 5 + max_new_tokens)
        assert "prefill_ms" in timings
        assert "decode_ms" in timings
        assert len(timings["decode_ms"]) == max_new_tokens - 1

    def test_cached_generate_preserves_prompt(self):
        """Test that generated output starts with original prompt."""
        model = CachedTransformerModel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=256,
        )
        input_ids = torch.randint(0, 1000, (1, 5))

        tokens, _ = cached_generate(model, input_ids, max_new_tokens=5)

        assert torch.equal(tokens[:, :5], input_ids)


class TestNaiveGenerate:
    """Tests for naive generation (for comparison)."""

    def test_naive_generate_output_length(self):
        """Test naive generation produces correct length."""
        model = TransformerModel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=256,
        )
        input_ids = torch.randint(0, 1000, (1, 5))

        tokens = naive_generate(model, input_ids, max_new_tokens=10)

        assert tokens.shape == (1, 15)

    def test_naive_generate_preserves_prompt(self):
        """Test naive generation preserves prompt."""
        model = TransformerModel(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_dim=256,
        )
        input_ids = torch.randint(0, 1000, (1, 5))

        tokens = naive_generate(model, input_ids, max_new_tokens=5)

        assert torch.equal(tokens[:, :5], input_ids)
