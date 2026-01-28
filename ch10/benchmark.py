import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    num_requests: int = 100
    prompt_len: int = 128
    output_len: int = 128
    concurrency: int = 1
    warmup_requests: int = 5


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    total_tokens: int

    ttft_mean_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float

    latency_mean_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float

    throughput_requests_per_sec: float
    throughput_tokens_per_sec: float

    per_request_metrics: list[RequestMetrics] = field(default_factory=list)

    def summary(self) -> str:
        return f"""
Benchmark Results
=================
Requests: {self.successful_requests}/{self.total_requests} successful
Total time: {self.total_time_sec:.2f}s
Total tokens: {self.total_tokens}

Time to First Token (TTFT):
  Mean: {self.ttft_mean_ms:.2f}ms
  P50:  {self.ttft_p50_ms:.2f}ms
  P90:  {self.ttft_p90_ms:.2f}ms
  P99:  {self.ttft_p99_ms:.2f}ms

End-to-End Latency:
  Mean: {self.latency_mean_ms:.2f}ms
  P50:  {self.latency_p50_ms:.2f}ms
  P90:  {self.latency_p90_ms:.2f}ms
  P99:  {self.latency_p99_ms:.2f}ms

Throughput:
  Requests/sec: {self.throughput_requests_per_sec:.2f}
  Tokens/sec:   {self.throughput_tokens_per_sec:.2f}
"""


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def run_benchmark(
    config: BenchmarkConfig,
    generate_fn: Callable[[list[int], int], dict],
) -> BenchmarkResult:
    metrics: list[RequestMetrics] = []

    for i in range(config.warmup_requests):
        prompt = list(range(config.prompt_len))
        _ = generate_fn(prompt, config.output_len)

    start_time = time.perf_counter()

    if config.concurrency == 1:
        for i in range(config.num_requests):
            prompt = list(range(config.prompt_len))
            result = generate_fn(prompt, config.output_len)

            metrics.append(RequestMetrics(
                request_id=i,
                prompt_tokens=result.get("prompt_tokens", config.prompt_len),
                completion_tokens=result.get("completion_tokens", config.output_len),
                time_to_first_token_ms=result.get("ttft_ms", 0),
                total_time_ms=result.get("total_ms", 0),
                tokens_per_second=result.get("tokens_per_sec", 0),
            ))
    else:
        results_lock = threading.Lock()

        def worker(request_id: int):
            prompt = list(range(config.prompt_len))
            result = generate_fn(prompt, config.output_len)

            with results_lock:
                metrics.append(RequestMetrics(
                    request_id=request_id,
                    prompt_tokens=result.get("prompt_tokens", config.prompt_len),
                    completion_tokens=result.get("completion_tokens", config.output_len),
                    time_to_first_token_ms=result.get("ttft_ms", 0),
                    total_time_ms=result.get("total_ms", 0),
                    tokens_per_second=result.get("tokens_per_sec", 0),
                ))

        threads = []
        for i in range(config.num_requests):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

            if len(threads) >= config.concurrency:
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                threads = []

        for t in threads:
            t.start()
        for t in threads:
            t.join()

    total_time = time.perf_counter() - start_time

    ttft_values = [m.time_to_first_token_ms for m in metrics]
    latency_values = [m.total_time_ms for m in metrics]
    total_tokens = sum(m.completion_tokens for m in metrics)

    return BenchmarkResult(
        config=config,
        total_requests=config.num_requests,
        successful_requests=len(metrics),
        failed_requests=config.num_requests - len(metrics),
        total_time_sec=total_time,
        total_tokens=total_tokens,
        ttft_mean_ms=statistics.mean(ttft_values) if ttft_values else 0,
        ttft_p50_ms=percentile(ttft_values, 50),
        ttft_p90_ms=percentile(ttft_values, 90),
        ttft_p99_ms=percentile(ttft_values, 99),
        latency_mean_ms=statistics.mean(latency_values) if latency_values else 0,
        latency_p50_ms=percentile(latency_values, 50),
        latency_p90_ms=percentile(latency_values, 90),
        latency_p99_ms=percentile(latency_values, 99),
        throughput_requests_per_sec=len(metrics) / total_time if total_time > 0 else 0,
        throughput_tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        per_request_metrics=metrics,
    )


def explain_benchmarking() -> str:
    return """
LLM Inference Benchmarking

Key metrics:

1. Time to First Token (TTFT)
   - Time from request to first generated token
   - Dominated by prefill time
   - Critical for perceived responsiveness

2. Time per Output Token (TPOT)
   - Average time between tokens
   - Determines streaming speed
   - Related to inter-token latency

3. End-to-End Latency
   - Total time from request to complete response
   - TTFT + (output_tokens * TPOT)
   - Important for non-streaming use

4. Throughput
   - Requests per second (RPS)
   - Tokens per second (TPS)
   - Measure of system capacity

Methodology:
  - Warmup: ignore first N requests
  - Fixed prompt/output lengths for consistency
  - Measure percentiles (P50, P90, P99)
  - Test at various concurrency levels

Factors affecting performance:
  - Batch size: higher = more throughput, higher latency
  - Sequence length: longer = more memory, slower
  - Model size: larger = slower, more accurate
  - Hardware: GPU memory, compute, memory bandwidth
"""


if __name__ == "__main__":
    print(explain_benchmarking())

    print("\n" + "=" * 60)
    print("Benchmark Demo")
    print("-" * 60)

    def mock_generate(prompt: list[int], max_tokens: int) -> dict:
        time.sleep(0.01)
        return {
            "prompt_tokens": len(prompt),
            "completion_tokens": max_tokens,
            "ttft_ms": 5.0,
            "total_ms": 10.0 + max_tokens * 0.5,
            "tokens_per_sec": max_tokens / (0.01 + max_tokens * 0.0005),
        }

    config = BenchmarkConfig(
        num_requests=20,
        prompt_len=64,
        output_len=32,
        warmup_requests=2,
    )

    result = run_benchmark(config, mock_generate)
    print(result.summary())
