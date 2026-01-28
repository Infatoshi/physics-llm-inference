import time
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class MatmulBenchmark:
    m: int
    n: int
    k: int
    triton_us: float
    torch_us: float
    speedup: float


if TRITON_AVAILABLE:
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            mask_m = offs_m[:, None] < M
            mask_n = offs_n[None, :] < N
            mask_k = (offs_k[None, :] + k) < K

            a = tl.load(a_ptrs, mask=mask_m & mask_k, other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & mask_n, other=0.0)

            accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=mask)


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        return torch.matmul(a, b)

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} vs {K2}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    return c


def benchmark_triton_matmul(
    m: int = 1024,
    n: int = 1024,
    k: int = 1024,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> MatmulBenchmark | None:
    if not torch.cuda.is_available():
        return None

    a = torch.randn(m, k, device=device, dtype=torch.float16)
    b = torch.randn(k, n, device=device, dtype=torch.float16)

    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / iterations * 1e6

    if TRITON_AVAILABLE:
        for _ in range(warmup):
            _ = triton_matmul(a, b)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = triton_matmul(a, b)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) / iterations * 1e6
    else:
        triton_time = torch_time

    return MatmulBenchmark(
        m=m,
        n=n,
        k=k,
        triton_us=triton_time,
        torch_us=torch_time,
        speedup=torch_time / triton_time,
    )


def triton_matmul_explained() -> str:
    return """
Triton Matrix Multiplication

The kernel computes C = A @ B using tiled computation:

1. Grid Launch:
   - Each program computes one BLOCK_M x BLOCK_N tile of C
   - Grid size = (ceil(M/BLOCK_M), ceil(N/BLOCK_N))

2. Tiled Loop:
   - Load BLOCK_M x BLOCK_K tile of A
   - Load BLOCK_K x BLOCK_N tile of B
   - Compute partial dot product
   - Accumulate in registers
   - Advance pointers by BLOCK_K

3. Tile Sizes:
   - BLOCK_M, BLOCK_N: output tile dimensions
   - BLOCK_K: reduction dimension tile
   - Trade-off: larger tiles = more reuse, but more registers/shared memory

4. Key Optimizations:
   - tl.dot uses tensor cores when available
   - Accumulation in fp32, output in fp16
   - Mask handles boundary conditions
"""


if __name__ == "__main__":
    print(triton_matmul_explained())

    if not torch.cuda.is_available():
        print("CUDA not available")
    else:
        print("\nBenchmark Results:")
        print("=" * 60)
        print(f"Triton available: {TRITON_AVAILABLE}")

        for size in [512, 1024, 2048, 4096]:
            result = benchmark_triton_matmul(m=size, n=size, k=size)
            if result:
                print(f"{size}x{size}: Triton {result.triton_us:.1f} us, "
                      f"Torch {result.torch_us:.1f} us, "
                      f"Speedup {result.speedup:.2f}x")
