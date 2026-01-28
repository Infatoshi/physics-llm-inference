"""Benchmarks for Chapter 01: Transformer Mechanics"""

import json
import time
from collections.abc import Callable

import torch

from .attention import MultiHeadAttention
from .ffn import FusedSwiGLUFFN, SwiGLUFFN
from .gqa import GroupedQueryAttention
from .transformer import TransformerBlock


def benchmark_fn(fn: Callable, warmup: int = 10, iterations: int = 100) -> dict:
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e6)  # Convert to microseconds

    return {
        "mean_us": sum(times) / len(times),
        "min_us": min(times),
        "max_us": max(times),
        "std_us": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    results = {
        "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
        "benchmarks": [],
    }

    # Test configurations
    batch_size = 8
    seq_len = 512
    hidden_dim = 1024
    num_heads = 16
    num_kv_heads = 4
    intermediate_dim = int(8 / 3 * hidden_dim)

    print(f"Running benchmarks on {results['hardware']}")
    print(f"Config: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}")
    print("-" * 60)

    # Benchmark MHA vs GQA
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    mha = MultiHeadAttention(hidden_dim, num_heads).to(device, dtype)
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads).to(device, dtype)

    mha_times = benchmark_fn(lambda: mha(x))
    gqa_times = benchmark_fn(lambda: gqa(x))

    print(f"MHA ({num_heads} heads):     {mha_times['mean_us']:.1f} us")
    print(f"GQA ({num_kv_heads} kv heads): {gqa_times['mean_us']:.1f} us")

    results["benchmarks"].append({
        "name": "attention_mha",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_heads,
        **mha_times,
    })
    results["benchmarks"].append({
        "name": "attention_gqa",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        **gqa_times,
    })

    # Benchmark FFN variants
    swiglu = SwiGLUFFN(hidden_dim, intermediate_dim).to(device, dtype)
    fused_swiglu = FusedSwiGLUFFN(hidden_dim, intermediate_dim).to(device, dtype)

    swiglu_times = benchmark_fn(lambda: swiglu(x))
    fused_times = benchmark_fn(lambda: fused_swiglu(x))

    print(f"\nSwiGLU (unfused):  {swiglu_times['mean_us']:.1f} us")
    print(f"SwiGLU (fused):    {fused_times['mean_us']:.1f} us")
    print(f"Speedup:           {swiglu_times['mean_us'] / fused_times['mean_us']:.2f}x")

    results["benchmarks"].append({
        "name": "ffn_swiglu_unfused",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "intermediate_dim": intermediate_dim,
        **swiglu_times,
    })
    results["benchmarks"].append({
        "name": "ffn_swiglu_fused",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "intermediate_dim": intermediate_dim,
        **fused_times,
    })

    # Benchmark full transformer block
    block = TransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_dim=intermediate_dim,
    ).to(device, dtype)

    block_times = benchmark_fn(lambda: block(x))

    print(f"\nTransformer block: {block_times['mean_us']:.1f} us")

    results["benchmarks"].append({
        "name": "transformer_block",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        **block_times,
    })

    # Memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _ = block(x)
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak memory:       {peak_memory_mb:.1f} MB")
        results["peak_memory_mb"] = peak_memory_mb

    # Output JSON
    print("\n" + "=" * 60)
    print(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":
    main()
