import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class TokenizeRequest:
    request_id: int
    text: str


@dataclass
class TokenizeResult:
    request_id: int
    tokens: list[int]
    num_tokens: int


class TokenizerWorker:
    def __init__(
        self,
        worker_id: int,
        tokenize_fn: Callable[[str], list[int]],
    ):
        self.worker_id = worker_id
        self.tokenize_fn = tokenize_fn
        self.requests_processed = 0

    def tokenize(self, text: str) -> list[int]:
        self.requests_processed += 1
        return self.tokenize_fn(text)


class TokenizerPool:
    def __init__(
        self,
        num_workers: int = 4,
        tokenize_fn: Callable[[str], list[int]] | None = None,
    ):
        self.num_workers = num_workers

        if tokenize_fn is None:
            tokenize_fn = self._dummy_tokenize

        self.workers = [
            TokenizerWorker(i, tokenize_fn)
            for i in range(num_workers)
        ]

        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.next_worker = 0
        self.lock = threading.Lock()

    def _dummy_tokenize(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def _get_next_worker(self) -> TokenizerWorker:
        with self.lock:
            worker = self.workers[self.next_worker]
            self.next_worker = (self.next_worker + 1) % self.num_workers
            return worker

    def tokenize(self, text: str) -> list[int]:
        worker = self._get_next_worker()
        return worker.tokenize(text)

    def tokenize_batch(self, texts: list[str]) -> list[list[int]]:
        futures = []
        for text in texts:
            worker = self._get_next_worker()
            future = self.executor.submit(worker.tokenize, text)
            futures.append(future)

        return [f.result() for f in futures]

    def tokenize_async(
        self,
        text: str,
        callback: Callable[[list[int]], None],
    ) -> None:
        worker = self._get_next_worker()

        def task():
            tokens = worker.tokenize(text)
            callback(tokens)

        self.executor.submit(task)

    def get_stats(self) -> dict:
        total_processed = sum(w.requests_processed for w in self.workers)
        per_worker = [w.requests_processed for w in self.workers]

        return {
            "num_workers": self.num_workers,
            "total_requests": total_processed,
            "per_worker": per_worker,
            "balance": min(per_worker) / max(per_worker) if max(per_worker) > 0 else 1.0,
        }

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


def explain_tokenizer_pool() -> str:
    return """
Tokenizer Worker Pool

Problem: Tokenization is CPU-bound
  - GPU is fast, but we need to tokenize before inference
  - Single-threaded tokenization becomes bottleneck
  - Python GIL limits CPU parallelism

Solution: Thread pool for tokenization
  - Multiple tokenizer instances
  - Round-robin or load-balanced dispatch
  - Parallel tokenization of batched requests

Design:
  1. Pool of N tokenizer workers
  2. Each worker has its own tokenizer instance
  3. ThreadPoolExecutor for parallelism
  4. Lock-free request dispatch

Async tokenization:
  - Don't block on tokenization
  - Submit request, get callback when done
  - Allows prefetching next batch tokens

Batch tokenization:
  - Submit multiple texts at once
  - Process in parallel across workers
  - Return results in order

Detokenization:
  - Same pattern for output decoding
  - Can be done async while generating

Process vs Thread pool:
  - Threads: simpler, share memory
  - Processes: bypass GIL, true parallelism
  - For tokenization, threads often sufficient
  - vLLM uses multiprocessing for maximum throughput
"""


if __name__ == "__main__":
    print(explain_tokenizer_pool())

    print("\n" + "=" * 60)
    print("Tokenizer Pool Demo")
    print("-" * 60)

    pool = TokenizerPool(num_workers=4)

    texts = [
        "Hello, world!",
        "What is machine learning?",
        "Explain quantum computing.",
        "How does attention work?",
    ]

    results = pool.tokenize_batch(texts)

    for text, tokens in zip(texts, results):
        print(f"Text: {text[:30]:30s} -> {len(tokens)} tokens")

    print(f"\nStats: {pool.get_stats()}")

    pool.shutdown()
