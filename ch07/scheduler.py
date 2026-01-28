from dataclasses import dataclass
from enum import Enum


class SchedulePolicy(Enum):
    FCFS = "fcfs"
    SHORTEST_FIRST = "shortest_first"
    PRIORITY = "priority"


@dataclass
class SchedulerConfig:
    max_running_requests: int = 32
    max_waiting_requests: int = 1024
    max_tokens_per_batch: int = 8192
    policy: SchedulePolicy = SchedulePolicy.FCFS


@dataclass
class SchedulerRequest:
    request_id: int
    prompt_len: int
    generated_len: int = 0
    max_tokens: int = 2048
    priority: int = 0
    num_blocks: int = 0

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_len

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.generated_len)


@dataclass
class SchedulerOutput:
    prefill_requests: list[SchedulerRequest]
    decode_requests: list[SchedulerRequest]
    preempted_requests: list[SchedulerRequest]
    total_tokens: int
    prefill_tokens: int
    decode_tokens: int


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting: list[SchedulerRequest] = []
        self.running: dict[int, SchedulerRequest] = {}
        self.preempted: list[SchedulerRequest] = []

    def add_request(
        self,
        request_id: int,
        prompt_len: int,
        max_tokens: int = 2048,
        priority: int = 0,
    ) -> None:
        req = SchedulerRequest(
            request_id=request_id,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            priority=priority,
        )
        self.waiting.append(req)
        self._sort_waiting()

    def _sort_waiting(self) -> None:
        if self.config.policy == SchedulePolicy.FCFS:
            pass
        elif self.config.policy == SchedulePolicy.SHORTEST_FIRST:
            self.waiting.sort(key=lambda r: r.prompt_len)
        elif self.config.policy == SchedulePolicy.PRIORITY:
            self.waiting.sort(key=lambda r: -r.priority)

    def _get_available_tokens(self) -> int:
        used = sum(r.total_len for r in self.running.values())
        return self.config.max_tokens_per_batch - used

    def schedule(self) -> SchedulerOutput:
        prefill_requests = []
        decode_requests = list(self.running.values())
        preempted = []

        available_slots = self.config.max_running_requests - len(self.running)
        available_tokens = self._get_available_tokens()

        for req in decode_requests:
            available_tokens -= 1

        to_admit = []
        for req in self.waiting[:]:
            if available_slots <= 0:
                break

            tokens_needed = req.prompt_len
            if tokens_needed <= available_tokens:
                to_admit.append(req)
                available_slots -= 1
                available_tokens -= tokens_needed

        for req in to_admit:
            self.waiting.remove(req)
            self.running[req.request_id] = req
            prefill_requests.append(req)

        prefill_tokens = sum(r.prompt_len for r in prefill_requests)
        decode_tokens = len(decode_requests)
        total_tokens = prefill_tokens + decode_tokens

        return SchedulerOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=preempted,
            total_tokens=total_tokens,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        )

    def update(
        self,
        finished_ids: set[int],
        generated_tokens: dict[int, int],
    ) -> None:
        for req_id in finished_ids:
            if req_id in self.running:
                del self.running[req_id]

        for req_id, num_tokens in generated_tokens.items():
            if req_id in self.running:
                self.running[req_id].generated_len += num_tokens

    def preempt(self, request_ids: list[int]) -> None:
        for req_id in request_ids:
            if req_id in self.running:
                req = self.running.pop(req_id)
                self.preempted.append(req)

    def get_running_count(self) -> int:
        return len(self.running)

    def get_waiting_count(self) -> int:
        return len(self.waiting)


def explain_scheduler_design() -> str:
    return """
Scheduler Design

The scheduler decides which requests run each iteration.

Key decisions:
1. Admission: which waiting requests to start
2. Preemption: which running requests to pause
3. Ordering: how to prioritize requests

Admission Policy:
  - FCFS: first-come first-served (fair)
  - Shortest First: minimize average latency
  - Priority: user-defined importance

Constraints:
  - Max batch size (GPU memory for activations)
  - Max total tokens (KV cache memory)
  - Max waiting queue (backpressure)

Preemption:
  - When memory pressure is high
  - Save KV cache to CPU (swap)
  - Or discard and recompute later

Each schedule() call returns:
  - Prefill requests (new, process full prompt)
  - Decode requests (continuing, generate one token)
  - Preempted requests (paused for memory)
"""


if __name__ == "__main__":
    print(explain_scheduler_design())

    print("\n" + "=" * 60)
    print("Scheduler Simulation")
    print("-" * 60)

    config = SchedulerConfig(
        max_running_requests=4,
        max_tokens_per_batch=1024,
    )
    scheduler = Scheduler(config)

    scheduler.add_request(0, prompt_len=100, max_tokens=50)
    scheduler.add_request(1, prompt_len=200, max_tokens=100)
    scheduler.add_request(2, prompt_len=50, max_tokens=200)
    scheduler.add_request(3, prompt_len=150, max_tokens=75)
    scheduler.add_request(4, prompt_len=80, max_tokens=100)

    for iteration in range(5):
        output = scheduler.schedule()

        print(f"\nIteration {iteration}:")
        print(f"  Prefill: {[r.request_id for r in output.prefill_requests]}")
        print(f"  Decode:  {[r.request_id for r in output.decode_requests]}")
        print(f"  Tokens:  {output.total_tokens} (prefill={output.prefill_tokens}, decode={output.decode_tokens})")

        finished = set()
        generated = {}
        for req in output.decode_requests + output.prefill_requests:
            generated[req.request_id] = 1
            if req.generated_len + 1 >= req.max_tokens:
                finished.add(req.request_id)

        scheduler.update(finished, generated)

        print(f"  Finished: {finished}")
        print(f"  Running: {scheduler.get_running_count()}, Waiting: {scheduler.get_waiting_count()}")
