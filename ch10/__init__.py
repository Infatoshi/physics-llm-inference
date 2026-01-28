"""
Chapter 10: Production Server

This chapter covers:
- OpenAI-compatible API design
- FastAPI server implementation
- Tokenizer worker pools
- Putting the inference engine together
- Benchmarking and metrics
"""

from .api_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageStats,
)
from .benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)
from .engine import (
    EngineConfig,
    GenerationResult,
    InferenceEngine,
)
from .tokenizer_pool import (
    TokenizerPool,
    TokenizerWorker,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "UsageStats",
    "TokenizerPool",
    "TokenizerWorker",
    "InferenceEngine",
    "EngineConfig",
    "GenerationResult",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmark",
]
