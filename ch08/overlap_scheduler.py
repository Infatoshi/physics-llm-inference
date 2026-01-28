from dataclasses import dataclass
from enum import Enum


class StreamType(Enum):
    COMPUTE = "compute"
    TRANSFER = "transfer"
    PREFETCH = "prefetch"


@dataclass
class OverlapConfig:
    enable_prefetch: bool = True
    enable_async_transfer: bool = True
    prefetch_blocks: int = 2
    max_concurrent_transfers: int = 2


@dataclass
class ScheduledOp:
    op_type: str
    request_id: int
    stream: StreamType
    start_time: float = 0.0
    end_time: float = 0.0


class OverlapScheduler:
    def __init__(self, config: OverlapConfig):
        self.config = config
        self.pending_compute: list[ScheduledOp] = []
        self.pending_transfer: list[ScheduledOp] = []
        self.completed: list[ScheduledOp] = []
        self.current_time = 0.0

    def schedule_compute(self, request_id: int, duration: float) -> ScheduledOp:
        op = ScheduledOp(
            op_type="compute",
            request_id=request_id,
            stream=StreamType.COMPUTE,
            start_time=self.current_time,
            end_time=self.current_time + duration,
        )
        self.pending_compute.append(op)
        return op

    def schedule_transfer(self, request_id: int, duration: float) -> ScheduledOp:
        op = ScheduledOp(
            op_type="transfer",
            request_id=request_id,
            stream=StreamType.TRANSFER,
            start_time=self.current_time,
            end_time=self.current_time + duration,
        )
        self.pending_transfer.append(op)
        return op

    def schedule_prefetch(self, request_id: int, duration: float) -> ScheduledOp:
        op = ScheduledOp(
            op_type="prefetch",
            request_id=request_id,
            stream=StreamType.PREFETCH,
            start_time=self.current_time,
            end_time=self.current_time + duration,
        )
        self.pending_transfer.append(op)
        return op

    def advance_time(self, delta: float) -> list[ScheduledOp]:
        self.current_time += delta

        newly_completed = []

        for op in self.pending_compute[:]:
            if op.end_time <= self.current_time:
                self.pending_compute.remove(op)
                self.completed.append(op)
                newly_completed.append(op)

        for op in self.pending_transfer[:]:
            if op.end_time <= self.current_time:
                self.pending_transfer.remove(op)
                self.completed.append(op)
                newly_completed.append(op)

        return newly_completed

    def get_timeline(self) -> dict:
        all_ops = self.completed + self.pending_compute + self.pending_transfer

        compute_ops = [op for op in all_ops if op.stream == StreamType.COMPUTE]
        transfer_ops = [op for op in all_ops if op.stream in (StreamType.TRANSFER, StreamType.PREFETCH)]

        return {
            "compute_stream": compute_ops,
            "transfer_stream": transfer_ops,
            "total_time": self.current_time,
            "compute_busy": sum(op.end_time - op.start_time for op in compute_ops),
            "transfer_busy": sum(op.end_time - op.start_time for op in transfer_ops),
        }

    def compute_overlap_ratio(self) -> float:
        timeline = self.get_timeline()
        if timeline["total_time"] == 0:
            return 0.0

        compute_busy = timeline["compute_busy"]
        transfer_busy = timeline["transfer_busy"]

        max_sequential = compute_busy + transfer_busy
        actual_time = timeline["total_time"]

        if max_sequential == 0:
            return 0.0

        overlap = max(0, max_sequential - actual_time)
        return overlap / max_sequential


def explain_overlap_scheduling() -> str:
    return """
Overlap Scheduling

Goal: Hide latency by overlapping operations on different resources

Key insight: GPU has multiple independent units
  - Compute SMs (tensor cores, CUDA cores)
  - Copy engines (DMA for CPU<->GPU transfers)
  - Memory controllers

Operations that can overlap:
  1. Compute and H2D transfer (next batch input)
  2. Compute and D2H transfer (previous batch output)
  3. Compute and NVLink/PCIe communication
  4. Multiple copy operations on different engines

Implementation with CUDA streams:
  - Default stream: compute operations
  - Transfer stream: memory copies
  - Use events to synchronize when needed

For LLM inference:
  - While computing current batch, prefetch KV cache blocks
  - While generating, transfer outputs to CPU
  - Overlap all-reduce with independent computation

Double/triple buffering:
  - Buffer A: currently processing
  - Buffer B: being filled (prefetch)
  - Buffer C: results being transferred out

Benefits:
  - Hide memory transfer latency
  - Better GPU utilization
  - Lower effective latency
"""


def simulate_overlap_vs_sequential(
    num_iterations: int = 10,
    compute_time: float = 1.0,
    transfer_time: float = 0.3,
) -> dict:
    sequential_time = num_iterations * (compute_time + transfer_time)

    overlap_scheduler = OverlapScheduler(OverlapConfig())

    for i in range(num_iterations):
        overlap_scheduler.schedule_compute(i, compute_time)
        overlap_scheduler.schedule_transfer(i, transfer_time)
        overlap_scheduler.advance_time(compute_time)

    overlap_scheduler.advance_time(transfer_time)

    overlap_time = overlap_scheduler.current_time

    return {
        "sequential_time": sequential_time,
        "overlap_time": overlap_time,
        "speedup": sequential_time / overlap_time,
        "overlap_ratio": overlap_scheduler.compute_overlap_ratio(),
    }


if __name__ == "__main__":
    print(explain_overlap_scheduling())

    print("\n" + "=" * 60)
    print("Overlap Scheduling Simulation")
    print("-" * 60)

    result = simulate_overlap_vs_sequential(
        num_iterations=10,
        compute_time=1.0,
        transfer_time=0.3,
    )

    print(f"Sequential time: {result['sequential_time']:.1f} units")
    print(f"Overlap time:    {result['overlap_time']:.1f} units")
    print(f"Speedup:         {result['speedup']:.2f}x")
    print(f"Overlap ratio:   {result['overlap_ratio']:.1%}")

    print("\nVarying transfer/compute ratio:")
    for transfer_ratio in [0.1, 0.2, 0.5, 1.0]:
        result = simulate_overlap_vs_sequential(
            num_iterations=10,
            compute_time=1.0,
            transfer_time=transfer_ratio,
        )
        print(f"  Transfer={transfer_ratio:.1f}: "
              f"Speedup={result['speedup']:.2f}x, "
              f"Overlap={result['overlap_ratio']:.1%}")
