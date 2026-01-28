import pytest

from .continuous_batcher import (
    ContinuousBatcher,
    ContinuousBatcherConfig,
    Request,
    RequestState,
)
from .paged_memory import (
    BlockTable,
    PagedKVCache,
)
from .radix_cache import (
    RadixCache,
)
from .scheduler import (
    SchedulePolicy,
    Scheduler,
    SchedulerConfig,
)
from .static_batcher import (
    StaticBatcher,
    StaticBatcherConfig,
    analyze_static_batching_waste,
)


class TestStaticBatcher:
    def test_add_request(self):
        config = StaticBatcherConfig(max_batch_size=4)
        batcher = StaticBatcher(config)

        id1 = batcher.add_request([1, 2, 3])
        id2 = batcher.add_request([4, 5])

        assert id1 == 0
        assert id2 == 1
        assert batcher.pending_count() == 2

    def test_form_batch(self):
        config = StaticBatcherConfig(max_batch_size=2)
        batcher = StaticBatcher(config)

        batcher.add_request([1, 2, 3, 4, 5])
        batcher.add_request([1, 2])

        batch = batcher.form_batch()

        assert batch is not None
        assert len(batch.requests) == 2
        assert batch.padded_length == 5
        assert batch.efficiency < 1.0

    def test_efficiency_calculation(self):
        config = StaticBatcherConfig()
        batcher = StaticBatcher(config)

        batcher.add_request([1, 2, 3, 4, 5])
        batcher.add_request([1, 2, 3, 4, 5])

        batch = batcher.form_batch()
        assert batch.efficiency == 1.0

    def test_analyze_waste(self):
        prompt_lens = [10, 50]
        gen_lens = [20, 100]

        analysis = analyze_static_batching_waste(
            prompt_lens, gen_lens, batch_size=2
        )

        assert analysis["num_batches"] == 1
        assert analysis["overall_efficiency"] < 1.0
        assert analysis["total_waste"] > 0


class TestContinuousBatcher:
    def test_add_request(self):
        config = ContinuousBatcherConfig(max_batch_size=4)
        batcher = ContinuousBatcher(config)

        id1 = batcher.add_request([1, 2, 3], max_tokens=10)

        assert id1 == 0
        stats = batcher.get_stats()
        assert stats["waiting"] == 1
        assert stats["running"] == 0

    def test_schedule_admits_requests(self):
        config = ContinuousBatcherConfig(max_batch_size=4, max_total_tokens=1000)
        batcher = ContinuousBatcher(config)

        batcher.add_request([1, 2, 3], max_tokens=5)
        batcher.add_request([4, 5], max_tokens=5)

        schedule = batcher.schedule_iteration()

        assert len(schedule["prefill_requests"]) == 2
        assert batcher.get_stats()["running"] == 2

    def test_request_finishes(self):
        config = ContinuousBatcherConfig(max_batch_size=4)
        batcher = ContinuousBatcher(config)

        batcher.add_request([1, 2], max_tokens=2)
        batcher.schedule_iteration()

        batcher.step({0: 100})
        batcher.step({0: 101})
        batcher.schedule_iteration()

        stats = batcher.get_stats()
        assert stats["finished"] == 1
        assert stats["running"] == 0


class TestRequest:
    def test_request_properties(self):
        req = Request(
            request_id=0,
            prompt_tokens=[1, 2, 3],
            max_tokens=10,
        )

        assert req.num_generated == 0
        assert req.total_tokens == 3
        assert not req.is_finished

    def test_request_state_transitions(self):
        req = Request(
            request_id=0,
            prompt_tokens=[1, 2, 3],
        )

        assert req.state == RequestState.WAITING

        req.state = RequestState.RUNNING
        assert not req.is_finished

        req.state = RequestState.FINISHED
        assert req.is_finished


class TestScheduler:
    def test_add_and_schedule(self):
        config = SchedulerConfig(max_running_requests=2)
        scheduler = Scheduler(config)

        scheduler.add_request(0, prompt_len=100)
        scheduler.add_request(1, prompt_len=200)
        scheduler.add_request(2, prompt_len=50)

        output = scheduler.schedule()

        assert len(output.prefill_requests) == 2
        assert scheduler.get_waiting_count() == 1

    def test_update_finishes_requests(self):
        config = SchedulerConfig(max_running_requests=4)
        scheduler = Scheduler(config)

        scheduler.add_request(0, prompt_len=50, max_tokens=10)
        scheduler.schedule()

        scheduler.update(finished_ids={0}, generated_tokens={})

        assert scheduler.get_running_count() == 0

    def test_shortest_first_policy(self):
        config = SchedulerConfig(
            max_running_requests=1,
            policy=SchedulePolicy.SHORTEST_FIRST,
        )
        scheduler = Scheduler(config)

        scheduler.add_request(0, prompt_len=200)
        scheduler.add_request(1, prompt_len=50)
        scheduler.add_request(2, prompt_len=100)

        output = scheduler.schedule()

        assert len(output.prefill_requests) == 1
        assert output.prefill_requests[0].prompt_len == 50


class TestRadixCache:
    def test_insert_and_match(self):
        cache = RadixCache()

        tokens = [1, 2, 3, 4, 5]
        kv_indices = [10, 20, 30, 40, 50]
        cache.insert(tokens, kv_indices)

        matched, indices = cache.match_prefix([1, 2, 3])
        assert matched == 3

    def test_no_match(self):
        cache = RadixCache()

        cache.insert([1, 2, 3], [10, 20, 30])

        matched, _ = cache.match_prefix([5, 6, 7])
        assert matched == 0

    def test_partial_match(self):
        cache = RadixCache()

        cache.insert([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])

        matched, _ = cache.match_prefix([1, 2, 3, 10, 11])
        assert matched == 3

    def test_cache_hit_rate(self):
        cache = RadixCache()

        cache.insert([1, 2, 3, 4], [10, 20, 30, 40])

        queries = [
            [1, 2, 3, 4],
            [1, 2, 5, 6],
            [5, 6, 7, 8],
        ]

        hit_rate = cache.get_cache_hit_rate(queries)
        assert 0.0 < hit_rate < 1.0


class TestPagedKVCache:
    def test_allocate_blocks(self):
        cache = PagedKVCache(
            num_blocks=100,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            device="cpu",
        )

        table = cache.allocate_blocks(request_id=1, num_tokens=50)

        assert table.request_id == 1
        assert table.num_tokens == 50
        assert len(table.block_indices) == 4

    def test_extend_blocks(self):
        cache = PagedKVCache(
            num_blocks=100,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            device="cpu",
        )

        cache.allocate_blocks(request_id=1, num_tokens=20)
        initial_blocks = len(cache.block_tables[1].block_indices)

        cache.extend_blocks(request_id=1, new_tokens=20)

        assert cache.block_tables[1].num_tokens == 40
        assert len(cache.block_tables[1].block_indices) >= initial_blocks

    def test_free_blocks(self):
        cache = PagedKVCache(
            num_blocks=100,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            device="cpu",
        )

        initial_free = cache.get_num_free_blocks()
        cache.allocate_blocks(request_id=1, num_tokens=50)

        after_alloc = cache.get_num_free_blocks()
        assert after_alloc < initial_free

        cache.free_blocks_for_request(request_id=1)

        assert cache.get_num_free_blocks() == initial_free

    def test_memory_usage(self):
        cache = PagedKVCache(
            num_blocks=100,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            device="cpu",
        )

        usage = cache.get_memory_usage()

        assert usage["total_blocks"] == 100
        assert usage["free_blocks"] == 100
        assert usage["utilization"] == 0.0

    def test_allocation_failure(self):
        cache = PagedKVCache(
            num_blocks=2,
            block_size=16,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            device="cpu",
        )

        with pytest.raises(RuntimeError):
            cache.allocate_blocks(request_id=1, num_tokens=100)


class TestBlockTable:
    def test_block_table_creation(self):
        table = BlockTable(
            request_id=1,
            block_indices=[0, 1, 2],
            num_tokens=40,
        )

        assert table.request_id == 1
        assert table.num_blocks() == 3
        assert table.num_tokens == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
