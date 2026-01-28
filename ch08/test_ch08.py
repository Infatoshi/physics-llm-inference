import pytest
import torch

from .chunked_prefill import (
    ChunkConfig,
    ChunkedPrefillScheduler,
    ChunkedRequest,
    PrefillChunk,
)
from .cuda_graph import (
    CUDAGraphRunner,
    GraphConfig,
)
from .mixed_batch import (
    MixedBatch,
    MixedBatchConfig,
    MixedBatchScheduler,
    MixedRequest,
    RequestPhase,
)
from .overlap_scheduler import (
    OverlapConfig,
    OverlapScheduler,
    StreamType,
    simulate_overlap_vs_sequential,
)


class TestChunkedPrefill:
    def test_chunk_config_defaults(self):
        config = ChunkConfig()
        assert config.chunk_size == 512
        assert config.max_chunks_per_iteration == 4

    def test_chunked_request_properties(self):
        req = ChunkedRequest(
            request_id=0,
            prompt_tokens=[1] * 1000,
            max_tokens=100,
        )

        assert req.total_chunks == 2

    def test_get_next_chunk(self):
        req = ChunkedRequest(
            request_id=0,
            prompt_tokens=[1] * 600,
            max_tokens=100,
        )

        chunk1 = req.get_next_chunk(chunk_size=512)
        assert chunk1 is not None
        assert chunk1.start_pos == 0
        assert chunk1.end_pos == 512
        assert not chunk1.is_last_chunk

        req.chunks_completed = 1

        chunk2 = req.get_next_chunk(chunk_size=512)
        assert chunk2 is not None
        assert chunk2.start_pos == 512
        assert chunk2.end_pos == 600
        assert chunk2.is_last_chunk

    def test_scheduler_add_request(self):
        config = ChunkConfig(chunk_size=100)
        scheduler = ChunkedPrefillScheduler(config)

        id1 = scheduler.add_request([1] * 250, max_tokens=50)
        id2 = scheduler.add_request([2] * 100, max_tokens=30)

        assert id1 == 0
        assert id2 == 1

        stats = scheduler.get_stats()
        assert stats["pending"] == 2

    def test_scheduler_schedules_chunks(self):
        config = ChunkConfig(chunk_size=100, max_chunks_per_iteration=2)
        scheduler = ChunkedPrefillScheduler(config)

        scheduler.add_request([1] * 250, max_tokens=50)

        chunks = scheduler.schedule_chunks()
        assert len(chunks) == 1
        assert chunks[0].start_pos == 0
        assert chunks[0].end_pos == 100


class TestMixedBatch:
    def test_mixed_request_creation(self):
        req = MixedRequest(
            request_id=0,
            prompt_len=100,
            max_tokens=50,
        )

        assert req.phase == RequestPhase.PREFILL
        assert req.generated == 0

    def test_mixed_batch_properties(self):
        batch = MixedBatch(
            prefill_requests=[],
            decode_requests=[],
            prefill_tokens=100,
            decode_tokens=50,
            total_tokens=150,
        )

        assert batch.prefill_fraction == pytest.approx(100/150)

    def test_scheduler_add_request(self):
        config = MixedBatchConfig()
        scheduler = MixedBatchScheduler(config)

        id1 = scheduler.add_request(prompt_len=100)
        id2 = scheduler.add_request(prompt_len=200)

        assert id1 == 0
        assert id2 == 1

        stats = scheduler.get_stats()
        assert stats["waiting_prefill"] == 2

    def test_scheduler_schedules_mixed_batch(self):
        config = MixedBatchConfig(
            max_batch_tokens=1000,
            prefill_priority=0.5,
        )
        scheduler = MixedBatchScheduler(config)

        scheduler.add_request(prompt_len=200)
        scheduler.add_request(prompt_len=100)

        batch = scheduler.schedule()

        assert batch.prefill_tokens > 0
        assert batch.total_tokens > 0


class TestCUDAGraph:
    def test_graph_config_defaults(self):
        config = GraphConfig()
        assert config.warmup_iterations == 3
        assert 1 in config.batch_sizes

    def test_graph_runner_initialization(self):
        config = GraphConfig(batch_sizes=[1, 2, 4])
        runner = CUDAGraphRunner(config)

        assert runner.config.batch_sizes == [1, 2, 4]
        assert len(runner.graphs) == 0

    def test_has_graph(self):
        config = GraphConfig()
        runner = CUDAGraphRunner(config)

        assert not runner.has_graph(1)
        assert not runner.has_graph(4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_capture_and_run_graph(self):
        def simple_model(x):
            return x * 2.0

        config = GraphConfig()
        runner = CUDAGraphRunner(config, model_fn=simple_model)

        success = runner.capture_graph(
            batch_size=2,
            input_shape=(32,),
            dtype=torch.float32,
        )

        assert success
        assert runner.has_graph(2)

        input_tensor = torch.ones(2, 32, device="cuda", dtype=torch.float32)
        output = runner.run_graph(2, input_tensor)

        assert output is not None
        expected = torch.ones(2, 32, device="cuda", dtype=torch.float32) * 2.0
        torch.testing.assert_close(output, expected)


class TestOverlapScheduler:
    def test_overlap_config_defaults(self):
        config = OverlapConfig()
        assert config.enable_prefetch == True
        assert config.prefetch_blocks == 2

    def test_schedule_compute(self):
        config = OverlapConfig()
        scheduler = OverlapScheduler(config)

        op = scheduler.schedule_compute(request_id=0, duration=1.0)

        assert op.op_type == "compute"
        assert op.stream == StreamType.COMPUTE
        assert op.end_time == 1.0

    def test_schedule_transfer(self):
        config = OverlapConfig()
        scheduler = OverlapScheduler(config)

        op = scheduler.schedule_transfer(request_id=0, duration=0.5)

        assert op.op_type == "transfer"
        assert op.stream == StreamType.TRANSFER

    def test_advance_time(self):
        config = OverlapConfig()
        scheduler = OverlapScheduler(config)

        scheduler.schedule_compute(request_id=0, duration=1.0)
        scheduler.schedule_transfer(request_id=0, duration=0.5)

        completed = scheduler.advance_time(0.6)

        assert len(completed) == 1
        assert completed[0].op_type == "transfer"

    def test_timeline(self):
        config = OverlapConfig()
        scheduler = OverlapScheduler(config)

        scheduler.schedule_compute(request_id=0, duration=1.0)
        scheduler.schedule_transfer(request_id=0, duration=0.5)
        scheduler.advance_time(1.0)

        timeline = scheduler.get_timeline()

        assert len(timeline["compute_stream"]) == 1
        assert len(timeline["transfer_stream"]) == 1

    def test_simulate_overlap_improvement(self):
        result = simulate_overlap_vs_sequential(
            num_iterations=5,
            compute_time=1.0,
            transfer_time=0.3,
        )

        assert result["speedup"] > 1.0
        assert result["overlap_time"] < result["sequential_time"]


class TestPrefillChunk:
    def test_chunk_creation(self):
        chunk = PrefillChunk(
            request_id=0,
            chunk_index=1,
            start_pos=512,
            end_pos=1024,
            is_last_chunk=False,
        )

        assert chunk.request_id == 0
        assert chunk.chunk_index == 1
        assert chunk.end_pos - chunk.start_pos == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
