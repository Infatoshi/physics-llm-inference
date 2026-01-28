from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class GraphConfig:
    batch_sizes: list[int] = None
    max_seq_len: int = 2048
    warmup_iterations: int = 3

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32]


class CUDAGraphRunner:
    def __init__(
        self,
        config: GraphConfig,
        model_fn: Callable | None = None,
    ):
        self.config = config
        self.model_fn = model_fn
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self.static_outputs: dict[int, torch.Tensor] = {}

    def capture_graph(
        self,
        batch_size: int,
        input_shape: tuple,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> bool:
        if not torch.cuda.is_available():
            return False

        if self.model_fn is None:
            return False

        static_input = torch.zeros(
            batch_size, *input_shape,
            dtype=dtype,
            device=device,
        )

        for _ in range(self.config.warmup_iterations):
            _ = self.model_fn(static_input)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            static_output = self.model_fn(static_input)

        self.graphs[batch_size] = graph
        self.static_inputs[batch_size] = {"input": static_input}
        self.static_outputs[batch_size] = static_output

        return True

    def run_graph(
        self,
        batch_size: int,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        if batch_size not in self.graphs:
            return None

        self.static_inputs[batch_size]["input"].copy_(input_tensor)

        self.graphs[batch_size].replay()

        return self.static_outputs[batch_size].clone()

    def has_graph(self, batch_size: int) -> bool:
        return batch_size in self.graphs

    def get_captured_batch_sizes(self) -> list[int]:
        return list(self.graphs.keys())


def explain_cuda_graphs() -> str:
    return """
CUDA Graphs

Problem: Kernel launch overhead
  - Each CUDA kernel has ~5-10us launch overhead
  - Decode phase has many small kernels per token
  - At 100 tokens/second, overhead becomes significant

Solution: CUDA Graphs
  - Record sequence of GPU operations
  - Replay entire sequence with single CPU launch
  - Amortize overhead across all kernels

How it works:
  1. Capture phase: run model once, record all GPU ops
  2. Replay phase: launch recorded ops with new inputs
  3. Only input/output copy happens on CPU

Requirements:
  - Fixed tensor shapes (batch size, sequence length)
  - No dynamic control flow
  - No CPU-GPU synchronization during capture

For LLM decode:
  - Capture graph for each supported batch size
  - During decode: select graph matching current batch
  - Copy new tokens into static input buffer
  - Replay graph
  - Copy output from static buffer

Benefits:
  - 2-5x lower latency for decode
  - Better GPU utilization
  - Essential for low-latency serving

Limitations:
  - Cannot use for variable-length prefill
  - Memory overhead from static buffers
  - Must pre-capture for each batch size
"""


def benchmark_graph_vs_eager(
    model_fn: Callable,
    input_shape: tuple,
    batch_size: int = 1,
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cuda",
) -> dict | None:
    if not torch.cuda.is_available():
        return None

    import time

    x = torch.randn(batch_size, *input_shape, device=device)

    for _ in range(warmup):
        _ = model_fn(x)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model_fn(x)
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - start) / iterations * 1e6

    static_input = torch.zeros_like(x)
    for _ in range(warmup):
        _ = model_fn(static_input)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model_fn(static_input)

    for _ in range(warmup):
        static_input.copy_(x)
        graph.replay()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        static_input.copy_(x)
        graph.replay()
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - start) / iterations * 1e6

    return {
        "batch_size": batch_size,
        "input_shape": input_shape,
        "eager_us": eager_time,
        "graph_us": graph_time,
        "speedup": eager_time / graph_time,
    }


if __name__ == "__main__":
    print(explain_cuda_graphs())

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("CUDA Graph Benchmark")
        print("-" * 60)

        def simple_model(x):
            x = torch.relu(x)
            x = x * 2.0
            x = torch.sigmoid(x)
            return x

        for batch_size in [1, 4, 16]:
            result = benchmark_graph_vs_eager(
                simple_model,
                input_shape=(1024,),
                batch_size=batch_size,
                iterations=1000,
            )
            if result:
                print(f"Batch {batch_size:3d}: "
                      f"Eager={result['eager_us']:.1f}us, "
                      f"Graph={result['graph_us']:.1f}us, "
                      f"Speedup={result['speedup']:.2f}x")
    else:
        print("\nCUDA not available for benchmarking")
