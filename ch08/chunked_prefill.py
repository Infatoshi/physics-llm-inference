from dataclasses import dataclass


@dataclass
class ChunkConfig:
    chunk_size: int = 512
    max_chunks_per_iteration: int = 4


@dataclass
class PrefillChunk:
    request_id: int
    chunk_index: int
    start_pos: int
    end_pos: int
    is_last_chunk: bool


@dataclass
class ChunkedRequest:
    request_id: int
    prompt_tokens: list[int]
    max_tokens: int
    chunks_completed: int = 0
    prefill_complete: bool = False

    @property
    def total_chunks(self) -> int:
        chunk_size = 512
        return (len(self.prompt_tokens) + chunk_size - 1) // chunk_size

    def get_next_chunk(self, chunk_size: int) -> PrefillChunk | None:
        if self.prefill_complete:
            return None

        start = self.chunks_completed * chunk_size
        end = min(start + chunk_size, len(self.prompt_tokens))

        if start >= len(self.prompt_tokens):
            self.prefill_complete = True
            return None

        is_last = end >= len(self.prompt_tokens)

        return PrefillChunk(
            request_id=self.request_id,
            chunk_index=self.chunks_completed,
            start_pos=start,
            end_pos=end,
            is_last_chunk=is_last,
        )


class ChunkedPrefillScheduler:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.pending: list[ChunkedRequest] = []
        self.in_prefill: dict[int, ChunkedRequest] = {}
        self.ready_for_decode: list[ChunkedRequest] = []
        self.next_id = 0

    def add_request(
        self,
        prompt_tokens: list[int],
        max_tokens: int = 2048,
    ) -> int:
        request_id = self.next_id
        self.next_id += 1

        request = ChunkedRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        self.pending.append(request)

        return request_id

    def schedule_chunks(self) -> list[PrefillChunk]:
        chunks = []
        tokens_scheduled = 0

        for req_id, req in list(self.in_prefill.items()):
            if len(chunks) >= self.config.max_chunks_per_iteration:
                break

            chunk = req.get_next_chunk(self.config.chunk_size)
            if chunk:
                chunks.append(chunk)
                tokens_scheduled += chunk.end_pos - chunk.start_pos
                req.chunks_completed += 1

                if chunk.is_last_chunk:
                    req.prefill_complete = True
                    self.ready_for_decode.append(req)
                    del self.in_prefill[req_id]

        while (self.pending and
               len(chunks) < self.config.max_chunks_per_iteration):
            req = self.pending.pop(0)
            self.in_prefill[req.request_id] = req

            chunk = req.get_next_chunk(self.config.chunk_size)
            if chunk:
                chunks.append(chunk)
                req.chunks_completed += 1

                if chunk.is_last_chunk:
                    req.prefill_complete = True
                    self.ready_for_decode.append(req)
                    del self.in_prefill[req.request_id]

        return chunks

    def get_decode_ready(self) -> list[ChunkedRequest]:
        ready = self.ready_for_decode
        self.ready_for_decode = []
        return ready

    def get_stats(self) -> dict:
        return {
            "pending": len(self.pending),
            "in_prefill": len(self.in_prefill),
            "ready_for_decode": len(self.ready_for_decode),
        }


def explain_chunked_prefill() -> str:
    return """
Chunked Prefill

Problem: Long prompts (10K+ tokens) cause latency spikes
  - Full prefill blocks all decode requests
  - Users waiting for generation see high TTFT
  - Throughput suffers from head-of-line blocking

Solution: Break long prefills into chunks
  - Process 512-2048 tokens per iteration
  - Interleave prefill chunks with decode tokens
  - Bound the maximum latency per iteration

Trade-offs:
  - Pro: More consistent latency
  - Pro: Better tail latency for short requests
  - Con: Slightly higher total time for long prefills
  - Con: More scheduling complexity

Implementation:
  1. Track prefill progress per request
  2. Each iteration: schedule N tokens of prefill + M decode tokens
  3. When prefill completes, move request to decode phase

Chunk size selection:
  - Too small: more iterations, more overhead
  - Too large: latency spikes return
  - Typical: 512-2048 tokens per chunk
"""


if __name__ == "__main__":
    print(explain_chunked_prefill())

    print("\n" + "=" * 60)
    print("Chunked Prefill Demo")
    print("-" * 60)

    config = ChunkConfig(chunk_size=100, max_chunks_per_iteration=2)
    scheduler = ChunkedPrefillScheduler(config)

    scheduler.add_request([1] * 350, max_tokens=50)
    scheduler.add_request([2] * 150, max_tokens=30)
    scheduler.add_request([3] * 50, max_tokens=20)

    for iteration in range(10):
        chunks = scheduler.schedule_chunks()
        ready = scheduler.get_decode_ready()

        if not chunks and not scheduler.in_prefill:
            print(f"\nIteration {iteration}: All prefills complete")
            break

        print(f"\nIteration {iteration}:")
        for chunk in chunks:
            print(f"  Chunk: req={chunk.request_id}, "
                  f"pos={chunk.start_pos}-{chunk.end_pos}, "
                  f"last={chunk.is_last_chunk}")

        if ready:
            print(f"  Ready for decode: {[r.request_id for r in ready]}")

        print(f"  Stats: {scheduler.get_stats()}")
