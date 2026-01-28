import time
from dataclasses import dataclass


@dataclass
class StaticBatcherConfig:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    timeout_ms: float = 50.0


@dataclass
class BatchedRequest:
    request_id: int
    tokens: list[int]
    arrival_time: float


@dataclass
class StaticBatch:
    requests: list[BatchedRequest]
    padded_length: int
    padding_tokens: int
    efficiency: float


class StaticBatcher:
    def __init__(self, config: StaticBatcherConfig):
        self.config = config
        self.pending: list[BatchedRequest] = []
        self.next_id = 0

    def add_request(self, tokens: list[int]) -> int:
        request_id = self.next_id
        self.next_id += 1

        self.pending.append(BatchedRequest(
            request_id=request_id,
            tokens=tokens,
            arrival_time=time.time(),
        ))

        return request_id

    def form_batch(self) -> StaticBatch | None:
        if not self.pending:
            return None

        batch_requests = self.pending[:self.config.max_batch_size]
        self.pending = self.pending[self.config.max_batch_size:]

        max_len = max(len(r.tokens) for r in batch_requests)
        padded_length = min(max_len, self.config.max_seq_len)

        total_tokens = sum(len(r.tokens) for r in batch_requests)
        total_padded = len(batch_requests) * padded_length
        padding_tokens = total_padded - total_tokens
        efficiency = total_tokens / total_padded if total_padded > 0 else 0.0

        return StaticBatch(
            requests=batch_requests,
            padded_length=padded_length,
            padding_tokens=padding_tokens,
            efficiency=efficiency,
        )

    def pending_count(self) -> int:
        return len(self.pending)


def analyze_static_batching_waste(
    request_lengths: list[int],
    max_gen_lengths: list[int],
    batch_size: int,
) -> dict:
    total_requests = len(request_lengths)
    batches = []

    for i in range(0, total_requests, batch_size):
        batch_prompts = request_lengths[i:i+batch_size]
        batch_gens = max_gen_lengths[i:i+batch_size]

        max_prompt = max(batch_prompts)
        max_total = max(p + g for p, g in zip(batch_prompts, batch_gens))

        useful_tokens = sum(p + g for p, g in zip(batch_prompts, batch_gens))
        padded_tokens = len(batch_prompts) * max_total

        batches.append({
            "size": len(batch_prompts),
            "max_prompt": max_prompt,
            "max_total": max_total,
            "useful": useful_tokens,
            "padded": padded_tokens,
            "waste": padded_tokens - useful_tokens,
            "efficiency": useful_tokens / padded_tokens,
        })

    total_useful = sum(b["useful"] for b in batches)
    total_padded = sum(b["padded"] for b in batches)

    return {
        "num_batches": len(batches),
        "total_useful_tokens": total_useful,
        "total_padded_tokens": total_padded,
        "total_waste": total_padded - total_useful,
        "overall_efficiency": total_useful / total_padded if total_padded > 0 else 0,
        "batches": batches,
    }


def explain_static_batching_problem() -> str:
    return """
The Static Batching Problem

Static batching pads all sequences to the longest in the batch:

Example batch:
  Request 1: "Hi" (2 tokens) -> generates 50 tokens
  Request 2: "Tell me about..." (20 tokens) -> generates 500 tokens
  Request 3: "What is 2+2?" (5 tokens) -> generates 10 tokens

With static batching:
  - All requests padded to 520 tokens (20 + 500)
  - Request 1 wastes 468 slots (90% waste)
  - Request 3 wastes 505 slots (97% waste)

Two types of waste:
1. Prompt padding: shorter prompts padded to longest prompt
2. Generation padding: requests that finish early still occupy slots

Throughput impact:
  - GPU computes on padded positions (wasted FLOPS)
  - Memory bandwidth wasted on padding
  - Batch completes when longest request finishes
  - Short requests have high latency waiting for long ones

The insight: treat each token generation as independent
Instead of "run batch until all done", run "generate one token per request"
Requests can join and leave the batch at any iteration.
"""


if __name__ == "__main__":
    print(explain_static_batching_problem())

    print("\n" + "=" * 60)
    print("Static Batching Waste Analysis")
    print("-" * 60)

    prompt_lens = [10, 50, 100, 200, 30, 15, 80, 150]
    gen_lens = [100, 50, 200, 100, 300, 20, 150, 50]

    analysis = analyze_static_batching_waste(prompt_lens, gen_lens, batch_size=4)

    print(f"Total requests: {len(prompt_lens)}")
    print("Batch size: 4")
    print(f"Number of batches: {analysis['num_batches']}")
    print(f"Overall efficiency: {analysis['overall_efficiency']:.1%}")
    print(f"Total waste: {analysis['total_waste']} tokens")

    print("\nPer-batch breakdown:")
    for i, batch in enumerate(analysis["batches"]):
        print(f"  Batch {i+1}: {batch['efficiency']:.1%} efficient, "
              f"{batch['waste']} tokens wasted")
