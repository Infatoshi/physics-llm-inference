"""Benchmarks for Chapter 02: The Generation Loop

Compares naive generation vs cached generation to demonstrate
the O(n^2) vs O(n) complexity difference.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '..')

from ch01.transformer import TransformerModel

from .cached_generation import CachedTransformerModel, cached_generate


def benchmark_naive_generation(
    model: TransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Benchmark naive generation (recompute everything each step)."""
    model.eval()
    generated = input_ids.clone()
    times = []

    with torch.no_grad():
        for i in range(max_new_tokens):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            logits = model(generated)
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    return {
        "times_ms": times,
        "total_ms": sum(times),
        "mean_ms": sum(times) / len(times),
        "first_token_ms": times[0],
        "last_token_ms": times[-1],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Model configuration (same for both)
    config = {
        "vocab_size": 1000,
        "hidden_dim": 512,
        "num_layers": 8,
        "num_heads": 8,
        "num_kv_heads": 2,
        "intermediate_dim": 1024,
    }

    results = {
        "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "config": config,
        "benchmarks": [],
    }

    print(f"Generation Loop Benchmarks (device={device})")
    print(f"Model: {config['num_layers']} layers, {config['hidden_dim']} dim")
    print("=" * 70)

    # Test different prompt lengths
    prompt_lengths = [10, 50, 100]
    max_new_tokens = 50

    for prompt_len in prompt_lengths:
        print(f"\nPrompt length: {prompt_len}, Generate: {max_new_tokens} tokens")
        print("-" * 70)

        input_ids = torch.randint(0, config["vocab_size"], (1, prompt_len), device=device)

        # Naive generation
        naive_model = TransformerModel(**config).to(device, dtype)
        naive_model.eval()

        # Warmup
        with torch.no_grad():
            _ = naive_model(input_ids)

        naive_times = benchmark_naive_generation(
            naive_model, input_ids, max_new_tokens, device
        )

        print("Naive generation:")
        print(f"  First token: {naive_times['first_token_ms']:.2f} ms")
        print(f"  Last token:  {naive_times['last_token_ms']:.2f} ms")
        print(f"  Total:       {naive_times['total_ms']:.2f} ms")
        print(f"  Slowdown:    {naive_times['last_token_ms'] / naive_times['first_token_ms']:.2f}x")

        # Cached generation
        cached_model = CachedTransformerModel(**config).to(device, dtype)
        cached_model.eval()

        # Warmup
        with torch.no_grad():
            warmup_caches = cached_model.create_caches(1, prompt_len + 10, torch.device(device), dtype)
            _ = cached_model(input_ids, warmup_caches)

        input_ids_fresh = torch.randint(0, config["vocab_size"], (1, prompt_len), device=device)
        _, cached_times = cached_generate(cached_model, input_ids_fresh, max_new_tokens)

        decode_times = cached_times["decode_ms"]
        print("\nCached generation:")
        print(f"  Prefill:     {cached_times['prefill_ms']:.2f} ms ({prompt_len} tokens)")
        print(f"  First decode:{decode_times[0]:.2f} ms")
        print(f"  Last decode: {decode_times[-1]:.2f} ms")
        print(f"  Decode avg:  {sum(decode_times) / len(decode_times):.2f} ms/token")
        print(f"  Total:       {cached_times['total_ms']:.2f} ms")

        speedup = naive_times["total_ms"] / cached_times["total_ms"]
        print(f"\nSpeedup: {speedup:.2f}x")

        results["benchmarks"].append({
            "prompt_len": prompt_len,
            "max_new_tokens": max_new_tokens,
            "naive": naive_times,
            "cached": {
                "prefill_ms": cached_times["prefill_ms"],
                "decode_times_ms": decode_times,
                "total_ms": cached_times["total_ms"],
            },
            "speedup": speedup,
        })

        # Clean up
        del naive_model, cached_model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Key insights
    print("\n" + "=" * 70)
    print("\nKey Insights:")
    print("-" * 70)
    print("1. Naive generation time INCREASES with sequence length")
    print("   - Each step recomputes attention for ALL previous tokens")
    print("   - O(n) per token -> O(n^2) total")
    print()
    print("2. Cached generation time is CONSTANT per decode step")
    print("   - KV cache stores previous K,V values")
    print("   - Each decode step only processes 1 new token")
    print("   - O(1) per token -> O(n) total")
    print()
    print("3. Prefill vs Decode are fundamentally different:")
    print("   - Prefill: GEMM (matrix-matrix), compute-bound")
    print("   - Decode: GEMV (matrix-vector), memory-bound")

    # Memory analysis
    print("\n" + "=" * 70)
    print("\nKV Cache Memory Analysis:")
    print("-" * 70)

    from .kv_cache import calculate_kv_cache_size

    for name, kv_heads in [("MHA (32 heads)", 32), ("GQA (8 heads)", 8), ("GQA (2 heads)", 2)]:
        sizes = calculate_kv_cache_size(
            batch_size=1,
            max_seq_len=4096,
            num_layers=config["num_layers"],
            num_kv_heads=kv_heads,
            head_dim=config["hidden_dim"] // config["num_heads"],
        )
        print(f"{name:20s}: {sizes['total_mb']:.1f} MB for 4096 tokens")

    # Output JSON
    print("\n" + "=" * 70)
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
