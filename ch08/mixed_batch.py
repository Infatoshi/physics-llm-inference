from dataclasses import dataclass
from enum import Enum


class RequestPhase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class MixedRequest:
    request_id: int
    prompt_len: int
    generated: int = 0
    max_tokens: int = 2048
    phase: RequestPhase = RequestPhase.PREFILL


@dataclass
class MixedBatch:
    prefill_requests: list[MixedRequest]
    decode_requests: list[MixedRequest]
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int

    @property
    def prefill_fraction(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return self.prefill_tokens / self.total_tokens


@dataclass
class MixedBatchConfig:
    max_batch_tokens: int = 8192
    max_prefill_tokens: int = 4096
    max_decode_batch: int = 256
    prefill_priority: float = 0.3


class MixedBatchScheduler:
    def __init__(self, config: MixedBatchConfig):
        self.config = config
        self.waiting_prefill: list[MixedRequest] = []
        self.waiting_decode: list[MixedRequest] = []
        self.next_id = 0

    def add_request(self, prompt_len: int, max_tokens: int = 2048) -> int:
        request_id = self.next_id
        self.next_id += 1

        req = MixedRequest(
            request_id=request_id,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            phase=RequestPhase.PREFILL,
        )
        self.waiting_prefill.append(req)

        return request_id

    def schedule(self) -> MixedBatch:
        prefill_requests = []
        decode_requests = []
        prefill_tokens = 0
        decode_tokens = 0

        prefill_budget = int(self.config.max_batch_tokens * self.config.prefill_priority)
        prefill_budget = min(prefill_budget, self.config.max_prefill_tokens)

        remaining_prefill = []
        for req in self.waiting_prefill:
            if prefill_tokens + req.prompt_len <= prefill_budget:
                prefill_requests.append(req)
                prefill_tokens += req.prompt_len
                req.phase = RequestPhase.DECODE
            else:
                remaining_prefill.append(req)
        self.waiting_prefill = remaining_prefill

        remaining_budget = self.config.max_batch_tokens - prefill_tokens
        max_decode = min(self.config.max_decode_batch, remaining_budget)

        decode_count = 0
        remaining_decode = []
        for req in self.waiting_decode:
            if decode_count < max_decode:
                decode_requests.append(req)
                decode_tokens += 1
                decode_count += 1
            else:
                remaining_decode.append(req)
        self.waiting_decode = remaining_decode

        total_tokens = prefill_tokens + decode_tokens

        return MixedBatch(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            total_tokens=total_tokens,
        )

    def complete_prefill(self, request_ids: list[int]) -> None:
        for req_id in request_ids:
            for req in self.waiting_prefill:
                if req.request_id == req_id:
                    self.waiting_prefill.remove(req)
                    req.phase = RequestPhase.DECODE
                    self.waiting_decode.append(req)
                    break

    def add_to_decode(self, requests: list[MixedRequest]) -> None:
        self.waiting_decode.extend(requests)

    def remove_finished(self, request_ids: set[int]) -> None:
        self.waiting_decode = [
            r for r in self.waiting_decode
            if r.request_id not in request_ids
        ]

    def get_stats(self) -> dict:
        return {
            "waiting_prefill": len(self.waiting_prefill),
            "waiting_decode": len(self.waiting_decode),
            "prefill_tokens_waiting": sum(r.prompt_len for r in self.waiting_prefill),
        }


def explain_mixed_batches() -> str:
    return """
Mixed Prefill/Decode Batches

Traditional approach: separate prefill and decode batches
  - Prefill: compute-bound, high arithmetic intensity
  - Decode: memory-bound, low arithmetic intensity
  - Different optimal batch sizes

Problem: separation causes inefficiency
  - Prefill batches may underutilize memory bandwidth
  - Decode batches may underutilize compute
  - Context switching overhead between phases

Solution: mix prefill and decode in same batch
  - Prefill tokens use compute
  - Decode tokens use memory bandwidth
  - Better overall GPU utilization

Scheduling:
  1. Budget total tokens for iteration
  2. Allocate portion to prefill (new requests)
  3. Fill remaining with decode (ongoing requests)
  4. Process all in single forward pass

Attention complexity:
  - Prefill: self-attention within prompt
  - Decode: cross-attention to existing KV cache
  - FlashAttention handles both cases

Benefits:
  - Better GPU utilization
  - Smoother latency distribution
  - Higher overall throughput
"""


if __name__ == "__main__":
    print(explain_mixed_batches())

    print("\n" + "=" * 60)
    print("Mixed Batch Scheduler Demo")
    print("-" * 60)

    config = MixedBatchConfig(
        max_batch_tokens=1024,
        max_prefill_tokens=512,
        max_decode_batch=64,
        prefill_priority=0.3,
    )
    scheduler = MixedBatchScheduler(config)

    scheduler.add_request(prompt_len=200)
    scheduler.add_request(prompt_len=150)
    scheduler.add_request(prompt_len=100)

    for i in range(5):
        batch = scheduler.schedule()

        print(f"\nIteration {i}:")
        print(f"  Prefill requests: {[r.request_id for r in batch.prefill_requests]}")
        print(f"  Decode requests:  {[r.request_id for r in batch.decode_requests]}")
        print(f"  Tokens: prefill={batch.prefill_tokens}, decode={batch.decode_tokens}")
        print(f"  Prefill fraction: {batch.prefill_fraction:.1%}")

        scheduler.add_to_decode(batch.prefill_requests)

        print(f"  Stats: {scheduler.get_stats()}")
