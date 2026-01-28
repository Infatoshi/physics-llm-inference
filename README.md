# Physics of LLM Inference - Companion Code

This repository contains runnable code for the book **The Physics of LLM Inference** by Elliot Arledge.

## Overview

The code is organized by chapter, with each folder containing:
- Working implementations of concepts from the book
- Benchmarks to reproduce the numbers shown
- Tests to verify correctness

## Requirements

```bash
uv sync --extra full
```

For GPU benchmarks, you'll need an NVIDIA GPU with CUDA support.

## Structure

```
ch01/   Transformer Mechanics (attention, GQA, FFN)
ch02/   The Generation Loop (KV cache, prefill/decode)
ch03/   Compute Characteristics (GEMM vs GEMV, roofline)
ch04/   Kernel Fundamentals (GPU architecture, CUDA basics)
ch05/   Kernel Optimization (coalescing, shared memory, Triton)
ch06/   FlashAttention (online softmax, tiled attention)
ch07/   Continuous Batching (Orca, radix cache, paged memory)
ch08/   Advanced Scheduling (chunked prefill, CUDA graphs)
ch09/   Mixture of Experts (MoE routing, expert caching)
ch10/   Production Server (FastAPI, tokenizer pool, benchmarks)
```

## Quick Start

```python
# Chapter 1: Basic attention
from ch01 import MultiHeadAttention
attn = MultiHeadAttention(embed_dim=512, num_heads=8)

# Chapter 2: KV Cache
from ch02 import KVCache, CachedTransformer

# Chapter 3: Benchmarks
from ch03 import benchmark_gemv, benchmark_gemm

# Chapter 7: Continuous batching
from ch07 import ContinuousBatcher, RadixCache
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific chapter
uv run pytest ch03/test_ch03.py -v
```

## Running Benchmarks

Each chapter has benchmark scripts:

```bash
# GEMV benchmark (Chapter 3)
uv run python -m ch03.gemv_benchmark

# Attention memory benchmark (Chapter 6)
uv run python -m ch06.attention_memory
```

## License

MIT License - See book for full terms.

## Reference

This code accompanies the mini-sglang reference implementation:
https://github.com/sgl-project/mini-sglang
