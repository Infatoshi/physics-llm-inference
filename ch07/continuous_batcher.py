import time
from dataclasses import dataclass, field
from enum import Enum


class RequestState(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Request:
    request_id: int
    prompt_tokens: list[int]
    max_tokens: int = 2048
    state: RequestState = RequestState.WAITING
    arrival_time: float = field(default_factory=time.time)
    start_time: float | None = None
    finish_time: float | None = None
    output_tokens: list[int] = field(default_factory=list)
    kv_cache_slot: int | None = None

    @property
    def num_generated(self) -> int:
        return len(self.output_tokens)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    def time_to_first_token(self) -> float | None:
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    def generation_time(self) -> float | None:
        if self.start_time is None or self.finish_time is None:
            return None
        return self.finish_time - self.start_time


@dataclass
class ContinuousBatcherConfig:
    max_batch_size: int = 32
    max_total_tokens: int = 8192
    prefill_chunk_size: int = 512


class ContinuousBatcher:
    def __init__(self, config: ContinuousBatcherConfig):
        self.config = config
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.finished: list[Request] = []
        self.next_id = 0
        self.total_tokens_in_batch = 0

    def add_request(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 2048,
    ) -> int:
        request_id = self.next_id
        self.next_id += 1

        request = Request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        self.waiting.append(request)

        return request_id

    def can_add_to_batch(self, request: Request) -> bool:
        if len(self.running) >= self.config.max_batch_size:
            return False

        new_total = self.total_tokens_in_batch + len(request.prompt_tokens)
        if new_total > self.config.max_total_tokens:
            return False

        return True

    def schedule_iteration(self) -> dict:
        now = time.time()

        finished_this_iter = []
        for req in self.running:
            if req.num_generated >= req.max_tokens:
                req.state = RequestState.FINISHED
                req.finish_time = now
                finished_this_iter.append(req)
                self.total_tokens_in_batch -= req.total_tokens

        for req in finished_this_iter:
            self.running.remove(req)
            self.finished.append(req)

        newly_added = []
        while self.waiting and self.can_add_to_batch(self.waiting[0]):
            req = self.waiting.pop(0)
            req.state = RequestState.RUNNING
            req.start_time = now
            self.running.append(req)
            self.total_tokens_in_batch += len(req.prompt_tokens)
            newly_added.append(req)

        prefill_tokens = sum(
            len(r.prompt_tokens) for r in newly_added
        )
        decode_tokens = len(self.running) - len(newly_added)

        return {
            "running": [r.request_id for r in self.running],
            "prefill_requests": [r.request_id for r in newly_added],
            "decode_requests": [r.request_id for r in self.running if r not in newly_added],
            "finished_requests": [r.request_id for r in finished_this_iter],
            "total_tokens": self.total_tokens_in_batch,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
        }

    def step(self, new_tokens: dict[int, int]) -> None:
        for req in self.running:
            if req.request_id in new_tokens:
                req.output_tokens.append(new_tokens[req.request_id])
                self.total_tokens_in_batch += 1

    def get_stats(self) -> dict:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
            "total_tokens_in_batch": self.total_tokens_in_batch,
        }


def explain_continuous_batching() -> str:
    return """
Continuous Batching (Orca)

Instead of processing a fixed batch until all sequences complete,
continuous batching allows sequences to enter and exit dynamically.

Key insight: LLM inference generates one token at a time.
Each iteration, we can:
1. Remove finished sequences (free their slots)
2. Add waiting sequences (fill available slots)
3. Generate one token for all running sequences

Iteration-level scheduling:
  - Requests join as soon as there's capacity
  - Requests leave immediately when done
  - No waiting for the longest sequence

Benefits:
  - Higher GPU utilization (always filling batch)
  - Lower latency for short requests
  - Better throughput (no padding waste)

Implementation:
  - Maintain waiting/running/finished queues
  - Each iteration: schedule() -> forward() -> update()
  - Memory management via paged KV cache

Prefill vs Decode:
  - New requests need prefill (process full prompt)
  - Existing requests need decode (generate next token)
  - Can mix prefill and decode in same batch
"""


if __name__ == "__main__":
    print(explain_continuous_batching())

    print("\n" + "=" * 60)
    print("Continuous Batcher Simulation")
    print("-" * 60)

    config = ContinuousBatcherConfig(max_batch_size=4, max_total_tokens=2048)
    batcher = ContinuousBatcher(config)

    batcher.add_request([1, 2, 3, 4, 5], max_tokens=3)
    batcher.add_request([1, 2, 3], max_tokens=5)
    batcher.add_request([1] * 100, max_tokens=2)

    for i in range(10):
        schedule = batcher.schedule_iteration()
        print(f"\nIteration {i}:")
        print(f"  Running: {schedule['running']}")
        print(f"  Prefill: {schedule['prefill_requests']}")
        print(f"  Decode:  {schedule['decode_requests']}")

        new_tokens = {rid: 100 + i for rid in schedule['running']}
        batcher.step(new_tokens)

        stats = batcher.get_stats()
        if stats['running'] == 0 and stats['waiting'] == 0:
            print("\nAll requests complete!")
            break

    print(f"\nFinal stats: {batcher.get_stats()}")
