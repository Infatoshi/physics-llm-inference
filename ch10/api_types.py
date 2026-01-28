import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class UsageStats:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatCompletionRequest:
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: list[str] | None = None
    user: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatCompletionRequest":
        messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in data.get("messages", [])
        ]
        return cls(
            model=data.get("model", ""),
            messages=messages,
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 1.0),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            user=data.get("user"),
        )


@dataclass
class ChatCompletionChoice:
    index: int
    message: ChatMessage
    finish_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats

    @classmethod
    def create(
        cls,
        model: str,
        content: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
    ) -> "ChatCompletionResponse":
        return cls(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage": self.usage.to_dict(),
        }


@dataclass
class StreamDelta:
    role: str | None = None
    content: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        if self.role is not None:
            d["role"] = self.role
        if self.content is not None:
            d["content"] = self.content
        return d


@dataclass
class StreamChoice:
    index: int
    delta: StreamDelta
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "index": self.index,
            "delta": self.delta.to_dict(),
        }
        if self.finish_reason is not None:
            d["finish_reason"] = self.finish_reason
        return d


@dataclass
class ChatCompletionChunk:
    id: str
    object: str
    created: int
    model: str
    choices: list[StreamChoice]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
        }

    def to_sse(self) -> str:
        import json
        return f"data: {json.dumps(self.to_dict())}\n\n"


def explain_api_design() -> str:
    return """
OpenAI-Compatible API Design

The OpenAI Chat Completions API has become a de facto standard.
Implementing compatibility allows drop-in replacement for applications.

Key endpoints:
  POST /v1/chat/completions - Main inference endpoint
  GET /v1/models - List available models
  GET /health - Health check

Request format:
  - model: which model to use
  - messages: conversation history (role + content)
  - max_tokens: maximum tokens to generate
  - temperature: sampling temperature
  - stream: whether to stream tokens

Response format:
  - id: unique request ID
  - choices: array of completions
  - usage: token counts

Streaming (SSE):
  - Each token sent as server-sent event
  - Format: data: {json}\n\n
  - Final message: data: [DONE]\n\n

Error handling:
  - HTTP 400: Bad request (invalid params)
  - HTTP 401: Unauthorized (bad API key)
  - HTTP 429: Rate limited
  - HTTP 500: Internal server error
"""


if __name__ == "__main__":
    print(explain_api_design())

    print("\n" + "=" * 60)
    print("API Types Demo")
    print("-" * 60)

    request = ChatCompletionRequest(
        model="qwen-30b",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is 2+2?"),
        ],
        max_tokens=100,
    )
    print(f"Request model: {request.model}")
    print(f"Messages: {len(request.messages)}")

    response = ChatCompletionResponse.create(
        model="qwen-30b",
        content="2+2 equals 4.",
        prompt_tokens=20,
        completion_tokens=5,
    )
    print(f"\nResponse ID: {response.id}")
    print(f"Usage: {response.usage.to_dict()}")
