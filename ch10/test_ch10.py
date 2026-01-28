
import pytest

from .api_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    StreamChoice,
    StreamDelta,
    UsageStats,
)
from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    percentile,
    run_benchmark,
)
from .engine import (
    EngineConfig,
    GenerationRequest,
    GenerationResult,
    InferenceEngine,
)
from .tokenizer_pool import (
    TokenizerPool,
    TokenizerWorker,
)


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"


class TestUsageStats:
    def test_creation(self):
        usage = UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30

    def test_to_dict(self):
        usage = UsageStats(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        d = usage.to_dict()
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 20


class TestChatCompletionRequest:
    def test_creation(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert request.model == "test-model"
        assert len(request.messages) == 1

    def test_from_dict(self):
        data = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = ChatCompletionRequest.from_dict(data)
        assert request.model == "test-model"
        assert request.max_tokens == 100


class TestChatCompletionResponse:
    def test_create(self):
        response = ChatCompletionResponse.create(
            model="test-model",
            content="Hello!",
            prompt_tokens=5,
            completion_tokens=10,
        )
        assert response.model == "test-model"
        assert len(response.choices) == 1
        assert response.usage.total_tokens == 15

    def test_to_dict(self):
        response = ChatCompletionResponse.create(
            model="test-model",
            content="Hello!",
            prompt_tokens=5,
            completion_tokens=10,
        )
        d = response.to_dict()
        assert "id" in d
        assert "choices" in d
        assert "usage" in d


class TestStreamTypes:
    def test_stream_delta(self):
        delta = StreamDelta(content="Hello")
        d = delta.to_dict()
        assert d["content"] == "Hello"
        assert "role" not in d

    def test_stream_choice(self):
        choice = StreamChoice(
            index=0,
            delta=StreamDelta(content="Hi"),
        )
        d = choice.to_dict()
        assert d["index"] == 0

    def test_chunk_to_sse(self):
        chunk = ChatCompletionChunk(
            id="test-id",
            object="chat.completion.chunk",
            created=12345,
            model="test",
            choices=[StreamChoice(index=0, delta=StreamDelta(content="Hi"))],
        )
        sse = chunk.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")


class TestTokenizerWorker:
    def test_tokenize(self):
        def dummy_tokenize(text):
            return [1, 2, 3]

        worker = TokenizerWorker(worker_id=0, tokenize_fn=dummy_tokenize)
        result = worker.tokenize("hello")

        assert result == [1, 2, 3]
        assert worker.requests_processed == 1


class TestTokenizerPool:
    def test_pool_creation(self):
        pool = TokenizerPool(num_workers=4)
        assert pool.num_workers == 4
        pool.shutdown()

    def test_single_tokenize(self):
        pool = TokenizerPool(num_workers=2)
        result = pool.tokenize("hello")
        assert len(result) > 0
        pool.shutdown()

    def test_batch_tokenize(self):
        pool = TokenizerPool(num_workers=2)
        texts = ["hello", "world", "test"]
        results = pool.tokenize_batch(texts)
        assert len(results) == 3
        pool.shutdown()

    def test_stats(self):
        pool = TokenizerPool(num_workers=2)
        pool.tokenize("test1")
        pool.tokenize("test2")
        stats = pool.get_stats()
        assert stats["total_requests"] == 2
        pool.shutdown()


class TestEngineConfig:
    def test_default_config(self):
        config = EngineConfig()
        assert config.max_batch_size == 32
        assert config.vocab_size == 32000


class TestGenerationRequest:
    def test_creation(self):
        request = GenerationRequest(
            request_id=0,
            prompt_tokens=[1, 2, 3],
            max_tokens=100,
        )
        assert request.request_id == 0
        assert len(request.prompt_tokens) == 3


class TestGenerationResult:
    def test_tokens_per_second(self):
        result = GenerationResult(
            request_id=0,
            output_tokens=[1, 2, 3, 4, 5],
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            time_to_first_token_ms=10.0,
            total_time_ms=100.0,
        )
        assert result.tokens_per_second == 50.0


class TestInferenceEngine:
    def test_engine_creation(self):
        config = EngineConfig()
        engine = InferenceEngine(config)
        assert engine.config == config

    def test_submit_request(self):
        config = EngineConfig()
        engine = InferenceEngine(config)
        request_id = engine.submit_request([1, 2, 3], max_tokens=10)
        assert request_id == 0

    def test_generate(self):
        config = EngineConfig(vocab_size=100)
        engine = InferenceEngine(config)

        request = GenerationRequest(
            request_id=0,
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )

        result = engine.generate(request)

        assert result.request_id == 0
        assert result.completion_tokens <= 5
        assert result.prompt_tokens == 3

    def test_stats(self):
        config = EngineConfig(vocab_size=100)
        engine = InferenceEngine(config)

        request = GenerationRequest(
            request_id=0,
            prompt_tokens=[1, 2, 3],
            max_tokens=5,
        )
        engine.generate(request)

        stats = engine.get_stats()
        assert stats["total_requests"] == 1


class TestBenchmarkConfig:
    def test_default_config(self):
        config = BenchmarkConfig()
        assert config.num_requests == 100
        assert config.concurrency == 1


class TestPercentile:
    def test_percentile_50(self):
        data = [1, 2, 3, 4, 5]
        assert percentile(data, 50) == 3

    def test_percentile_empty(self):
        assert percentile([], 50) == 0.0


class TestRunBenchmark:
    def test_benchmark_runs(self):
        def mock_generate(prompt, max_tokens):
            return {
                "prompt_tokens": len(prompt),
                "completion_tokens": max_tokens,
                "ttft_ms": 5.0,
                "total_ms": 10.0,
                "tokens_per_sec": 100.0,
            }

        config = BenchmarkConfig(
            num_requests=5,
            warmup_requests=1,
        )

        result = run_benchmark(config, mock_generate)

        assert result.successful_requests == 5
        assert result.ttft_mean_ms > 0


class TestBenchmarkResult:
    def test_summary(self):
        config = BenchmarkConfig(num_requests=10)
        result = BenchmarkResult(
            config=config,
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            total_time_sec=1.0,
            total_tokens=100,
            ttft_mean_ms=5.0,
            ttft_p50_ms=4.0,
            ttft_p90_ms=8.0,
            ttft_p99_ms=10.0,
            latency_mean_ms=50.0,
            latency_p50_ms=45.0,
            latency_p90_ms=80.0,
            latency_p99_ms=100.0,
            throughput_requests_per_sec=10.0,
            throughput_tokens_per_sec=100.0,
        )

        summary = result.summary()
        assert "10/10 successful" in summary
        assert "TTFT" in summary
        assert "Throughput" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
